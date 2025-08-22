
import os
import json
import time
import pandas as pd
import duckdb
import streamlit as st

# ========== CONFIG ==========
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
SYSTEM_PROMPT = r'''
You are “Telco Churn Data Scientist Agent.”

Mission
- Translate the user's natural-language question into a single, valid DuckDB SQL query that runs against a table named `data`.
- Answer only using columns that exist in the currently loaded CSV.
- Never invent columns or values. If information is unavailable, say so and suggest a close alternative using existing columns.

Data Access Contract
- The app will re-read the CSV on every question. Assume no state between questions.
- The only table you can reference is `data`.

Output Format (STRICT)
Return a single JSON object inside one ```json fenced code block with exactly these keys:
{
  "thinking_notes": "brief internal notes; keep very short",
  "sql": "DuckDB-compatible SQL using only table `data` and existing columns",
  "narrative": "1–3 sentence TL;DR in plain business language",
  "chart_suggestion": {"type": "bar|line|none", "x": "col or null", "y": "col or null"}
}
If you cannot produce safe SQL using only the provided schema, set "sql" to "" and explain why in "narrative".

Style & Constraints
- Be precise and auditable (counts, percentages, grouped stats).
- No causal claims—descriptive/associational language only.
- Use exact column names (case-sensitive); do not alias nonexistent columns.
- If the user asks for an unsupported metric, offer the closest available fields.

'''

# Optional OpenAI SDK
try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except Exception:
    _OPENAI_AVAILABLE = False

st.set_page_config(page_title="Data Scientist Agent (DuckDB + Streamlit)", layout="wide")
st.title("📊 Data Scientist Agent — CSV → DuckDB → SQL → Insight")

# ---------- SESSION ----------
if "user_q" not in st.session_state:
    st.session_state.user_q = ""
if "ask_from_preset" not in st.session_state:
    st.session_state.ask_from_preset = False

# ---------- SIDEBAR ----------
with st.sidebar:
    st.header("⚙️ Setup")
    st.write("The agent reloads the CSV **every time you ask a question**.")
    uploaded = st.file_uploader("Upload CSV", type=["csv"], accept_multiple_files=False)
    model = st.text_input("OpenAI model", value=DEFAULT_MODEL, help="e.g., gpt-4o-mini")
    api_key = st.text_input("OpenAI API Key", value=OPENAI_API_KEY, type="password")
    st.caption("Tip: set env vars OPENAI_API_KEY and OPENAI_MODEL to avoid typing each time.")
    show_schema = st.checkbox("Show inferred schema", value=True)

    st.markdown("---")
    st.subheader("📈 Quick Data Dashboard")
    if uploaded is not None:
        try:
            df_side = pd.read_csv(uploaded)
            st.metric("Rows", f"{len(df_side):,}")
            st.metric("Columns", f"{len(df_side.columns):,}")
            cols = set(df_side.columns)
            if "Churn" in cols:
                churn_rate = (df_side["Churn"].astype(str).str.strip().str.lower().eq("yes")).mean()
                st.metric("Churn Rate", f"{churn_rate*100:.1f}%")
            if "MonthlyCharges" in cols:
                st.metric("Avg Monthly Charges", f"{pd.to_numeric(df_side['MonthlyCharges'], errors='coerce').mean():.2f}")
            if "tenure" in cols:
                st.metric("Avg Tenure (months)", f"{pd.to_numeric(df_side['tenure'], errors='coerce').mean():.1f}")
            if "Contract" in cols:
                st.write("**Top Contract Types**")
                for k, v in df_side["Contract"].astype(str).value_counts().head(3).items():
                    st.write(f"- {k}: {v:,}")
            if "InternetService" in cols:
                st.write("**Top Internet Service**")
                for k, v in df_side["InternetService"].astype(str).value_counts().head(3).items():
                    st.write(f"- {k}: {v:,}")
        except Exception as e:
            st.warning(f"Dashboard preview failed: {e}")

st.divider()

# ---------- PRESET QUESTIONS ----------
def build_presets(df: pd.DataFrame | None):
    base = [
        "Show total rows and columns.",
        "List columns with % missing values (descending).",
    ]
    if df is None:
        return base
    cols = set(df.columns)
    telco_presets = []
    if {"Churn", "Contract"}.issubset(cols):
        telco_presets.append("What is churn rate by Contract?")
    if {"Churn", "InternetService"}.issubset(cols):
        telco_presets.append("What is churn rate by InternetService?")
    if {"Churn", "PaymentMethod"}.issubset(cols):
        telco_presets.append("Top 5 PaymentMethod by churn rate and count.")
    if {"MonthlyCharges", "tenure"}.issubset(cols):
        telco_presets.append("Average MonthlyCharges by tenure buckets (0-12, 13-24, 25-48, 49+).")
    if {"Churn", "SeniorCitizen"}.issubset(cols):
        telco_presets.append("Churn rate by SeniorCitizen.")
    return base + telco_presets[:6]

df_for_presets = None
if uploaded is not None:
    try:
        df_for_presets = pd.read_csv(uploaded, nrows=2000)
    except Exception:
        pass

preset_cols = st.columns(3)
presets = build_presets(df_for_presets)
for i, q in enumerate(presets):
    with preset_cols[i % 3]:
        if st.button(q, key=f"preset_{i}"):
            st.session_state.user_q = q
            st.session_state.ask_from_preset = True

# ---------- QUESTION INPUT + CHART CONTROLS ----------
st.subheader("Ask a question")
user_q = st.text_area(
    "Ask about your data…",
    key="user_q",
    height=100,
    placeholder="e.g., What's the churn rate by contract type?",
)

left, right = st.columns([1,1])
with left:
    chart_type = st.selectbox("Chart type", ["auto (from model)", "bar", "line", "area", "scatter", "none"])
with right:
    ask_clicked = st.button("Ask the Agent", type="primary", use_container_width=True)

interpret_verbose = st.checkbox("Generate detailed interpretation (LLM)", value=True)

ask = ask_clicked or st.session_state.ask_from_preset

# ---------- HELPERS ----------
def infer_schema(df: pd.DataFrame, sample_rows: int = 5) -> str:
    lines = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        sample_vals = df[col].dropna().unique()[:sample_rows]
        sample_list = ", ".join([repr(v)[:60] for v in sample_vals])
        lines.append(f"- {col} ({dtype})  samples: {sample_list}")
    return "\\n".join(lines)

def call_llm(schema_text: str, question: str, model_name: str, key: str) -> str:
    if not _OPENAI_AVAILABLE:
        raise RuntimeError("OpenAI SDK not installed. Run: pip install openai")
    if not key:
        raise RuntimeError("Missing OPENAI_API_KEY. Provide it in the sidebar.")
    client = OpenAI(api_key=key)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            "Return a SINGLE JSON object in a ```json fenced block with keys: thinking_notes, sql, narrative, chart_suggestion.\\n"
            "Use ONLY the schema below (table name: data).\\n\\n"
            f"Schema for `data`:\\n{schema_text}\\n\\n"
            f"User question:\\n{question}"
        )},
    ]
    resp = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.1,
    )
    return resp.choices[0].message.content

def extract_json_block(text: str) -> str:
    # 1) Try ```json fenced
    import re
    if not text:
        return ""
    m = re.search(r"```json\\s*(\\{.*?\\})\\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # 2) Fallback: first top-level {...}
    brace = re.search(r"\\{.*\\}", text, flags=re.DOTALL)
    return brace.group(0).strip() if brace else ""

def coerce_json_obj(obj):
    try:
        if isinstance(obj, (dict, list)):
            json.dumps(obj)  # validate serializable
            return obj
        if isinstance(obj, str) and obj.strip():
            return json.loads(obj)  # try load string
    except Exception:
        pass
    # fallback to a simple dict with a note
    return {"note": "Non-JSON data; showing string preview.", "preview": str(obj)[:500]}

def safe_run_sql(df: pd.DataFrame, sql: str):
    con = duckdb.connect(database=":memory:")
    con.register("data", df)
    return con.execute(sql).df()

def llm_interpret_table(df_out: pd.DataFrame, narrative_hint: str, model_name: str, key: str) -> str:
    if not _OPENAI_AVAILABLE or not key:
        return ""
    client = OpenAI(api_key=key)
    max_rows = min(50, len(df_out))
    csv_sample = df_out.head(max_rows).to_csv(index=False)
    prompt = (
        "Interpret ONLY the result table (CSV sample). Be concise, neutral, and avoid speculation. "
        f"Optional hint: {narrative_hint}\\n\\n"
        f"Result table (up to {max_rows} rows):\\n{csv_sample}"
    )
    r = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "Be concise, precise, neutral."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return r.choices[0].message.content.strip()

# ---------- RUN ----------
diag = {"schema_text": "", "raw_model_response": "", "parsed_json": None, "sql": "", "error": "", "timing": {}}

if ask:
    st.session_state.ask_from_preset = False
    status = st.status("⏳ Querying the data… (the agent re-reads your CSV and plans a SQL query)", state="running")
    t0 = time.time()
    try:
        if uploaded is None:
            raise RuntimeError("Please upload a CSV in the sidebar.")
        t_csv0 = time.time()
        df = pd.read_csv(uploaded)  # re-read EVERY time
        t_csv1 = time.time()

        schema_txt = infer_schema(df)
        diag["schema_text"] = schema_txt
        if show_schema:
            with st.expander("📜 Inferred Schema", expanded=False):
                st.code(schema_txt, language="markdown")

        status.update(label="🧠 Planning SQL with the model…", state="running")
        t_llm0 = time.time()
        raw = call_llm(schema_txt, user_q, model, api_key)
        t_llm1 = time.time()
        diag["raw_model_response"] = raw

        jtxt = extract_json_block(raw)
        if not jtxt:
            raise RuntimeError("Model did not return a valid JSON block. Check 'Raw model response' below.")
        try:
            parsed = json.loads(jtxt)
        except Exception as je:
            raise RuntimeError(f"JSON parsing failed: {je}. First 200 chars: {jtxt[:200]}")
        diag["parsed_json"] = parsed
        sql = (parsed.get("sql") or "").strip()
        diag["sql"] = sql

        if not isinstance(parsed.get("chart_suggestion"), dict):
            parsed["chart_suggestion"] = {}

        if not sql:
            status.update(label="ℹ️ The agent couldn't form a safe SQL with given columns.", state="complete")
            st.info(parsed.get("narrative", "The agent couldn't form a safe SQL with given columns."))
        else:
            status.update(label="🏃 Running SQL on your data…", state="running")
            t_sql0 = time.time()
            try:
                out = safe_run_sql(df, sql)
                t_sql1 = time.time()

                status.update(label="✅ Query complete", state="complete")
                st.subheader("TL;DR")
                st.write(parsed.get("narrative", ""))

                st.subheader("Results")
                st.dataframe(out, use_container_width=True)

                st.subheader("Visualization")
                cs = parsed.get("chart_suggestion") or {}
                chosen = chart_type
                if chosen == "auto (from model)":
                    chosen = cs.get("type", "none")
                    x = cs.get("x")
                    y = cs.get("y")
                else:
                    x = cs.get("x")
                    y = cs.get("y")
                    if (not x or x not in out.columns) and len(out.columns) >= 2:
                        x, y = out.columns[0], out.columns[1]

                if chosen in {"bar", "line", "area"} and x in out.columns and y in out.columns:
                    if chosen == "bar":
                        st.bar_chart(out.set_index(x)[y])
                    elif chosen == "line":
                        st.line_chart(out.set_index(x)[y])
                    elif chosen == "area":
                        st.area_chart(out.set_index(x)[y])
                elif chosen == "scatter" and x in out.columns and y in out.columns:
                    st.scatter_chart(out, x=x, y=y)
                else:
                    st.caption("No chart rendered (type 'none' or insufficient columns).")

                if interpret_verbose:
                    st.subheader("Detailed Interpretation")
                    try:
                        interp = llm_interpret_table(out, parsed.get("narrative", ""), model, api_key)
                        st.write(interp or "Interpretation skipped (no API key / SDK).")
                    except Exception as e:
                        st.write(f"Interpretation failed: {e}")

                st.subheader("Evidence: Executed SQL")
                st.code(sql, language="sql")

            except Exception as exec_err:
                status.update(label="❌ SQL execution failed", state="error")
                diag["error"] = f"{type(exec_err).__name__}: {exec_err}"
                st.error("SQL execution failed. See Diagnostics for details.")

        diag["timing"] = {
            "csv_read_s": round(t_csv1 - t_csv0, 3),
            "llm_s": round(t_llm1 - t_llm0, 3),
            "sql_exec_s": round((t_sql1 - t_sql0), 3) if 't_sql1' in locals() else None,
            "total_s": round(time.time() - t0, 3),
        }

    except Exception as e:
        status.update(label="❌ Error encountered", state="error")
        diag["error"] = f"{type(e).__name__}: {e}"
        st.error("The agent hit an error. Open Diagnostics to inspect.")

    with st.expander("🛠️ Diagnostics (raw model output, timings, errors)"):
        st.write("**Timings (seconds):**")
        st.json(coerce_json_obj(diag.get("timing", {})))
        st.write("**Raw model response:**")
        st.code(diag.get("raw_model_response", ""), language="markdown")
        st.write("**Parsed JSON:**")
        st.json(coerce_json_obj(diag.get("parsed_json", {})))
        st.write("**SQL:**")
        st.code(diag.get("sql", ""), language="sql")
        if diag.get("error"):
            st.write("**Error / Trace:**")
            st.code(diag["error"], language="text")

else:
    st.info("Upload a CSV, click a preset or type a question, and click **Ask the Agent**. The app will re-read your CSV every time and show a live status while querying.")
