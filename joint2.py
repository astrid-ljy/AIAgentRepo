
import os
import json
import pandas as pd
import duckdb
import streamlit as st

# ===================== CONFIG =====================
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Try OpenAI SDK (optional)
try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except Exception:
    _OPENAI_AVAILABLE = False

# ===================== SYSTEM PROMPTS =====================
SYSTEM_BA_FIN = r"""
You are the “Financial Business Analyst Agent (BA).”
You must produce BOTH the business-facing answer and the precise DuckDB SQL that backs it.
Assume a single table named `data` is already registered in DuckDB.

Goals:
- Focus on financial analysis (revenue, sales, amount, price, cost, discount, tax, margin, profit, ARPU, LTV, retention value) using ONLY existing columns.
- Use the user’s question and the SCHEMA to choose the right metrics, filters, and group-bys.
- Prefer time-series and segment breakdowns when date/period or categorical columns exist.
- Be auditable: the SQL must exactly match what you describe in the narrative.
- If a requested metric is impossible with the current schema, return an empty SQL string and explain briefly in the narrative.

Rules:
- Table name MUST be `data`.
- Use ONLY columns that exist in the schema. Do not invent columns.
- Derived metrics are allowed ONLY if all needed columns exist (e.g., revenue = price*quantity; margin = revenue - cost; margin_rate = margin/revenue when revenue>0).
- If date-like columns exist, consider grouping by day/week/month (using DuckDB functions like date_trunc('month', col)).
- Keep the answer short and business-oriented—no causal claims.

Return your answer inside ONE and only ONE ```json block with the following keys:
{
  "sql": "DuckDB SQL or empty string if not possible",
  "narrative": "concise business summary of what this query returns",
  "chart_suggestion": {"type": "bar|line|area|scatter|none", "x": "col or null", "y": "col or null"},
  "followups": ["next question 1", "next question 2"]
}
"""

SYSTEM_BA_REVISE = r"""
You are the “Financial BA (Reviser).”
You will receive:
- The SCHEMA,
- The prior BA JSON (with sql/narrative),
- The prior RESULT sample (CSV),
- The USER FEEDBACK.

Task:
- Revise/improve the previous SQL and narrative strictly based on the feedback and schema.
- Keep `data` as the table name and use only existing columns.
- If feedback requires changes that are impossible with the schema, return sql="" and explain briefly.

Return ONE ```json block with the same keys as before (sql, narrative, chart_suggestion, followups).
"""

# ===================== UI SETUP =====================
st.set_page_config(page_title="💼 Financial BA Agent (SQL + Feedback Chat)", layout="wide")
st.title("💼 Financial BA Agent — Chat • SQL • Feedback")

with st.sidebar:
    st.header("⚙️ Setup")
    uploaded = st.file_uploader("Upload CSV", type=["csv"], accept_multiple_files=False)
    model = st.text_input("OpenAI model", value=DEFAULT_MODEL)
    api_key = st.text_input("OpenAI API Key", value=OPENAI_API_KEY, type="password")
    show_schema = st.checkbox("Show inferred schema", value=True)
    show_sql = st.checkbox("Show SQL in replies", value=True)

# ===================== HELPERS =====================
def infer_schema(df: pd.DataFrame, sample_rows: int = 5) -> str:
    lines = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        sample_vals = df[col].dropna().unique()[:sample_rows]
        sample_list = ", ".join([repr(v)[:60] for v in sample_vals])
        lines.append(f"- {col} ({dtype})  samples: {sample_list}")
    return "\\n".join(lines)

def openai_chat(system_prompt: str, user_payload: str, model_name: str, key: str) -> str:
    if not _OPENAI_AVAILABLE:
        raise RuntimeError("OpenAI SDK not installed. Run: pip install openai")
    if not key:
        raise RuntimeError("Missing OPENAI_API_KEY. Provide it in the sidebar.")
    client = OpenAI(api_key=key)
    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_payload},
        ],
        temperature=0.1,
    )
    return resp.choices[0].message.content

def extract_json_block(text: str) -> str:
    if not text:
        return ""
    import re
    m = re.search(r"```json\\s*(\\{.*?\\})\\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""

def run_duckdb(df: pd.DataFrame, sql: str) -> pd.DataFrame:
    con = duckdb.connect(database=":memory:")
    con.register("data", df)
    return con.execute(sql).df()

def df_to_snapshot(df: pd.DataFrame, limit: int = 50):
    return df.head(min(limit, len(df))).to_dict(orient="list")

def snapshot_to_df(snap: dict) -> pd.DataFrame:
    try:
        return pd.DataFrame(snap)
    except Exception:
        return pd.DataFrame()

def render_assistant_turn(turn: dict):
    """Render an assistant turn: narrative, optional SQL, table, and chart."""
    with st.chat_message("assistant"):
        if turn.get("narrative"):
            st.write(turn["narrative"])
        if turn.get("sql") and st.session_state.get("show_sql_flag", True):
            st.code(turn["sql"], language="sql")
        if turn.get("result_snapshot"):
            st.dataframe(snapshot_to_df(turn["result_snapshot"]), use_container_width=True)
        cs = turn.get("chart_suggestion") or {}
        out_df = snapshot_to_df(turn.get("result_snapshot") or {})
        if isinstance(cs, dict) and cs.get("type", "none") != "none" and not out_df.empty:
            x = cs.get("x"); y = cs.get("y")
            if x in out_df.columns and y in out_df.columns:
                st.subheader("Quick Chart")
                if cs["type"] == "bar":
                    st.bar_chart(out_df.set_index(x)[y])
                elif cs["type"] == "line":
                    st.line_chart(out_df.set_index(x)[y])
                elif cs["type"] == "area":
                    st.area_chart(out_df.set_index(x)[y])
                elif cs["type"] == "scatter":
                    st.scatter_chart(out_df, x=x, y=y)
        if turn.get("followups"):
            st.caption("Follow-ups: " + " • ".join(turn["followups"]))

# ===================== SESSION STATE =====================
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of dicts: {"role": "user"/"assistant", "content": str, ...}
if "df" not in st.session_state:
    st.session_state.df = None
if "schema_txt" not in st.session_state:
    st.session_state.schema_txt = ""
if "last_ba_json" not in st.session_state:
    st.session_state.last_ba_json = {}
if "show_sql_flag" not in st.session_state:
    st.session_state.show_sql_flag = True

# Bind checkbox to session flag
st.session_state.show_sql_flag = show_sql

# ===================== LOAD DATA / SCHEMA =====================
if uploaded is not None:
    st.session_state.df = pd.read_csv(uploaded)
    st.session_state.schema_txt = infer_schema(st.session_state.df)

if st.session_state.df is None:
    st.info("⬆️ Please upload a CSV to get started.")
else:
    if show_schema:
        with st.expander("📜 Inferred Schema", expanded=False):
            st.code(st.session_state.schema_txt, language="markdown")

# ===================== RENDER CHAT LOG =====================
for turn in st.session_state.messages:
    if turn["role"] == "user":
        with st.chat_message("user"):
            st.markdown(turn["content"])
    else:
        render_assistant_turn(turn)

# ===================== INPUT AREA =====================
def build_condensed_history(max_pairs: int = 3) -> str:
    """Return last N user/assistant pairs as plain text (for light context)."""
    strs = []
    count_pairs = 0
    # Walk from end to start
    for t in reversed(st.session_state.messages):
        if t["role"] == "assistant":
            # assistant first, then the preceding user, to form a pair
            a_txt = t.get("narrative") or t.get("content") or ""
            strs.append(f"ASSISTANT: {a_txt}")
            # find previous user turn
            for t2 in reversed(st.session_state.messages[:st.session_state.messages.index(t)]):
                if t2["role"] == "user":
                    strs.append(f"USER: {t2.get('content','')}")
                    break
            count_pairs += 1
            if count_pairs >= max_pairs:
                break
    return "\\n".join(reversed(strs)) if strs else ""

def ask_ba_and_execute(question: str):
    """Call BA with schema + question, run SQL, append assistant turn."""
    if st.session_state.df is None:
        st.error("Please upload a CSV first.")
        return

    history = build_condensed_history(max_pairs=3)
    payload = f"""USER QUESTION:
{question}

SCHEMA for `data`:
{st.session_state.schema_txt}

CHAT LOG (most recent pairs):
{history if history else '(none)'}
"""
    raw = openai_chat(SYSTEM_BA_FIN, payload, model, api_key)
    jtxt = extract_json_block(raw)
    if not jtxt:
        st.error("BA did not return valid JSON.")
        return
    try:
        ba = json.loads(jtxt)
    except Exception:
        st.error("Could not parse BA JSON.")
        return

    sql = (ba.get("sql") or "").strip()
    out_df = pd.DataFrame()
    if sql:
        try:
            out_df = run_duckdb(st.session_state.df, sql)
        except Exception as e:
            st.warning(f"SQL execution failed: {e}")
    # Build assistant turn
    turn = {
        "role": "assistant",
        "content": ba.get("narrative",""),
        "narrative": ba.get("narrative",""),
        "sql": sql if show_sql else "",
        "result_snapshot": df_to_snapshot(out_df) if not out_df.empty else {},
        "chart_suggestion": ba.get("chart_suggestion") or {},
        "followups": ba.get("followups") or []
    }
    st.session_state.messages.append({"role":"user","content": question})
    st.session_state.messages.append(turn)
    st.session_state.last_ba_json = ba

def revise_with_feedback(feedback: str):
    """Send feedback to BA to revise the last answer, re-run SQL, and append a new assistant turn."""
    if not st.session_state.messages or st.session_state.df is None:
        st.error("Nothing to revise yet.")
        return
    # Find last assistant turn
    last_assistant = None
    for t in reversed(st.session_state.messages):
        if t["role"] == "assistant":
            last_assistant = t
            break
    if last_assistant is None:
        st.error("No assistant answer to revise.")
        return

    prior_json = json.dumps(st.session_state.last_ba_json or {}, ensure_ascii=False)
    prior_result_csv = ""
    if last_assistant.get("result_snapshot"):
        prior_result_csv = snapshot_to_df(last_assistant["result_snapshot"]).to_csv(index=False)

    payload = f"""SCHEMA for `data`:
{st.session_state.schema_txt}

PRIOR BA JSON:
{prior_json}

PRIOR RESULT (CSV sample):
{prior_result_csv}

USER FEEDBACK:
{feedback}
"""
    raw = openai_chat(SYSTEM_BA_REVISE, payload, model, api_key)
    jtxt = extract_json_block(raw)
    if not jtxt:
        st.error("BA (Reviser) did not return valid JSON.")
        return
    try:
        ba2 = json.loads(jtxt)
    except Exception:
        st.error("Could not parse revised BA JSON.")
        return

    sql2 = (ba2.get("sql") or "").strip()
    out_df2 = pd.DataFrame()
    if sql2:
        try:
            out_df2 = run_duckdb(st.session_state.df, sql2)
        except Exception as e:
            st.warning(f"Revised SQL execution failed: {e}")
    # Append a new assistant turn
    turn2 = {
        "role": "assistant",
        "content": ba2.get("narrative",""),
        "narrative": ba2.get("narrative",""),
        "sql": sql2 if show_sql else "",
        "result_snapshot": df_to_snapshot(out_df2) if not out_df2.empty else {},
        "chart_suggestion": ba2.get("chart_suggestion") or {},
        "followups": ba2.get("followups") or []
    }
    st.session_state.messages.append(turn2)
    st.session_state.last_ba_json = ba2

# Main input controls
col1, col2 = st.columns([3,2])
with col1:
    question = st.chat_input("Ask a financial analysis question about your data…")
    if question:
        ask_ba_and_execute(question)

with col2:
    st.subheader("Feedback")
    fb = st.text_area("Tell the agent how to improve the previous answer (e.g., change grouping, filter dates, use margin rate).", height=100)
    if st.button("Improve last answer", type="secondary", use_container_width=True):
        if not fb.strip():
            st.warning("Please enter some feedback first.")
        else:
            revise_with_feedback(fb.strip())
