
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

**Mission**
- Translate the user's natural-language question into a single, valid **DuckDB SQL** query that runs against a table named **data**.
- Answer **only** using columns that exist in the currently loaded CSV.
- Never invent columns or values. If information is unavailable, say so and suggest a close alternative using existing columns.

**Hard Rules**
1) Only reference the provided schema and column names exactly as given (case-sensitive).
2) The only table name you may use is `data`.
3) Prefer precise, auditable metrics (counts, %, grouped stats). Avoid causal language.
4) Return JSON in a fenced code block with exactly these keys:
   {
     "thinking_notes": "brief chain-of-thought style notes for yourself; keep short",
     "sql": "STRICT SQL for DuckDB using the table `data`",
     "narrative": "1-3 sentence TL;DR in business language (no SQL)",
     "chart_suggestion": {"type": "bar|line|none", "x": "col or null", "y": "col or null"}
   }
   If you cannot produce safe SQL strictly using the schema, set "sql" to "" and explain why in "narrative".
5) Always put the JSON inside a single ```json code block.
'''

try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except Exception:
    _OPENAI_AVAILABLE = False

st.set_page_config(page_title="Data Scientist Agent (DuckDB + Streamlit)", layout="wide")
st.title("📊 Telco Churn Data Scientist Agent")

with st.sidebar:
    st.header("⚙️ Setup")
    st.write("The agent reloads the CSV **every time you ask a question**.")
    uploaded = st.file_uploader("Upload CSV", type=["csv"], accept_multiple_files=False)
    model = st.text_input("OpenAI model", value=DEFAULT_MODEL, help="e.g., gpt-4o-mini")
    api_key = st.text_input("OpenAI API Key", value=OPENAI_API_KEY, type="password")
    st.caption("Tip: set env vars OPENAI_API_KEY and OPENAI_MODEL to avoid typing each time.")
    show_schema = st.checkbox("Show inferred schema", value=True)

st.divider()

user_q = st.text_area("Ask a question about your data…", height=100, placeholder="e.g., What's the churn rate by contract type?")
ask = st.button("Ask the Agent", type="primary", use_container_width=True)

diag = {
    "schema_text": "",
    "raw_model_response": "",
    "parsed_json": None,
    "sql": "",
    "error": "",
    "timing": {},
}

def infer_schema(df: pd.DataFrame, sample_rows: int = 5) -> str:
    lines = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        sample_vals = df[col].dropna().unique()[:sample_rows]
        sample_list = ", ".join([repr(v)[:60] for v in sample_vals])
        lines.append(f"- {col} ({dtype})  samples: {sample_list}")
    return "\n".join(lines)

def call_llm(schema_text: str, question: str, model_name: str, key: str) -> str:
    if not _OPENAI_AVAILABLE:
        raise RuntimeError("OpenAI SDK not installed. Run: pip install openai")
    if not key:
        raise RuntimeError("Missing OPENAI_API_KEY. Provide it in the sidebar.")
    client = OpenAI(api_key=key)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            "You must respond with a single JSON object in a ```json fenced code block.\n\n"
            "Here is the table schema for `data`:\n"
            f"{schema_text}\n\n"
            "User question:\n"
            f"{question}"
        )},
    ]
    resp = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.1,
    )
    return resp.choices[0].message.content

def extract_json_block(text: str) -> str:
    if not text:
        return ""
    import re
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    return m.group(1).strip() if m else ""

def safe_run_sql(df: pd.DataFrame, sql: str):
    con = duckdb.connect(database=":memory:")
    con.register("data", df)
    return con.execute(sql).df()

if ask:
    status = st.status("⏳ Querying the data… (the agent reads your CSV and plans a SQL query)", state="running")
    t0 = time.time()
    try:
        if uploaded is None:
            raise RuntimeError("Please upload a CSV in the sidebar.")
        t_csv0 = time.time()
        df = pd.read_csv(uploaded)
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
            raise RuntimeError("Model did not return a valid ```json block. See Diagnostics.")
        parsed = json.loads(jtxt)
        diag["parsed_json"] = parsed
        sql = (parsed.get("sql") or "").strip()
        diag["sql"] = sql

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

                cs = parsed.get("chart_suggestion") or {}
                if isinstance(cs, dict) and cs.get("type", "none") != "none":
                    ct = cs.get("type")
                    x = cs.get("x")
                    y = cs.get("y")
                    if ct in {"bar", "line"} and x in out.columns and y in out.columns:
                        st.subheader("Quick Chart")
                        if ct == "bar":
                            st.bar_chart(out.set_index(x)[y])
                        elif ct == "line":
                            st.line_chart(out.set_index(x)[y])

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
        st.json(diag.get("timing", {}))
        st.write("**Raw model response:**")
        st.code(diag.get("raw_model_response", ""), language="markdown")
        st.write("**Parsed JSON:**")
        st.json(diag.get("parsed_json", {}))
        st.write("**SQL:**")
        st.code(diag.get("sql", ""), language="sql")
        if diag.get("error"):
            st.write("**Error / Trace:**")
            st.code(diag["error"], language="text")

else:
    st.info("Upload a CSV, type a question, and click **Ask the Agent**. The app will re-read your CSV every time and show a live status while querying.")
