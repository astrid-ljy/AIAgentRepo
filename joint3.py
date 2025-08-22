import os
import json
import re
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
SYSTEM_DECIDER = r"""
You are a router for a Financial BA chat app.
You will receive:
- SCHEMA,
- CHAT HISTORY (chronological list of USER/ASSISTANT exchanges),
- LAST_BA_JSON (sql/narrative of the latest answer),
- LAST_RESULT_CSV (sample of last query output),
- NEW_INPUT (the user's latest message).

Decide whether NEW_INPUT is:
- "feedback": instructions to change or refine the **immediately previous** answer/SQL,
- "new_question": a fresh business question to analyze from scratch.

Heuristics for "feedback": mentions like "instead", "add", "filter", "by month", "group", "use margin rate", "previous", "that chart/table", column names used in the last answer, or short imperative edits.
If the user asks for different metrics, segments, or a new time range not clearly referring to the last answer, it is "new_question".

Return ONLY a json object with:
{
  "mode": "feedback" | "new_question",
  "normalized_question": "string (if mode=new_question else empty)",
  "normalized_feedback": "string (if mode=feedback else empty)"
}
"""

SYSTEM_BA_FIN = r"""
You are the “Financial Business Analyst Agent (BA).”
You must produce BOTH the business-facing answer and the precise DuckDB SQL that backs it.
Assume a single table named `data` is already registered in DuckDB.

Goals:
- Focus on financial analysis (revenue, sales, amount, price, cost, discount, tax, margin, profit, ARPU, LTV, retention value) using ONLY existing columns.
- Use the user’s question, the SCHEMA, and the CHAT HISTORY for context to choose the right metrics, filters, and group-bys.
- Prefer time-series and segment breakdowns when date/period or categorical columns exist.
- Be auditable: the SQL must exactly match what you describe in the narrative.
- If a requested metric is impossible with the current schema, return an empty SQL string and explain briefly in the narrative.

Rules:
- Table name MUST be `data`.
- Use ONLY columns that exist in the schema. Do not invent columns.
- Derived metrics are allowed ONLY if all needed columns exist (e.g., revenue = price*quantity; margin = revenue - cost; margin_rate = margin/revenue when revenue>0).
- If date-like columns exist, consider grouping by day/week/month (DuckDB: date_trunc('month', col)).
- Keep the answer short and business-oriented—no causal claims.

Return ONLY a json object (no backticks, no markdown, no extra text) with the following keys:
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
- The CHAT HISTORY,
- The prior BA JSON (with sql/narrative),
- The prior RESULT sample (CSV),
- The USER FEEDBACK.

Task:
- Revise/improve the previous SQL and narrative strictly based on the feedback and schema.
- Keep `data` as the table name and use only existing columns.
- If feedback requires changes that are impossible with the schema, return sql="" and explain briefly.

Return ONLY a json object (no backticks, no markdown, no extra text) with the same keys as before (sql, narrative, chart_suggestion, followups).
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
    debug_raw = st.checkbox("Debug: show raw model replies on parse errors", value=False)

# ===================== HELPERS =====================
def infer_schema(df: pd.DataFrame, sample_rows: int = 5) -> str:
    lines = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        sample_vals = df[col].dropna().unique()[:sample_rows]
        sample_list = ", ".join([repr(v)[:60] for v in sample_vals])
        lines.append(f"- {col} ({dtype})  samples: {sample_list}")
    return "\n".join(lines)

def openai_chat(system_prompt: str, user_payload: str, model_name: str, key: str, require_json: bool = False) -> str:
    if not _OPENAI_AVAILABLE:
        raise RuntimeError("OpenAI SDK not installed. Run: pip install openai")
    if not key:
        raise RuntimeError("Missing OPENAI_API_KEY. Provide it in the sidebar.")
    client = OpenAI(api_key=key)
    kwargs = dict(model=model_name, messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_payload},
        ], temperature=0.1)
    # Try to force json_object; if it fails, fallback without
    if require_json:
        try:
            kwargs["response_format"] = {"type": "json_object"}
            resp = client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content
        except Exception:
            kwargs.pop("response_format", None)
            resp = client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content
    else:
        resp = client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content

def extract_json_block(text: str) -> str:
    if not text:
        return ""
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""

def parse_ba_json(text: str):
    """Robust parsing: direct JSON -> fenced block -> greedy braces slice."""
    if not text:
        return None
    # 1) try direct parse
    try:
        return json.loads(text)
    except Exception:
        pass
    # 2) try fenced
    fenced = extract_json_block(text)
    if fenced:
        try:
            return json.loads(fenced)
        except Exception:
            pass
    # 3) greedy slice between first { and last }
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end+1]
        try:
            return json.loads(candidate)
        except Exception:
            return None
    return None

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

# ===================== CHAT HISTORY UTILS =====================
def compile_chat_history(max_chars: int = 12000) -> str:
    """Full chronological chat history (user prompts + assistant narratives), truncated to max_chars."""
    parts = []
    for t in st.session_state.messages:
        if t["role"] == "user":
            parts.append(f"USER: {t.get('content','')}")
        else:
            # Use concise narrative for history (not the whole table/SQL)
            parts.append(f"ASSISTANT: {t.get('narrative') or t.get('content','')}")
    text = "\n".join(parts)
    if len(text) <= max_chars:
        return text
    # Truncate from the front to keep the most recent context
    return text[-max_chars:]

# ===================== CORE LOGIC (unified input) =====================
def handle_user_input(user_text: str):
    if st.session_state.df is None:
        st.error("Please upload a CSV first.")
        return

    # 1) Decide whether it's feedback or a new question
    chat_hist = compile_chat_history()
    prior_json = json.dumps(st.session_state.last_ba_json or {}, ensure_ascii=False)
    # Find last assistant result snapshot to CSV
    last_assistant = None
    for t in reversed(st.session_state.messages):
        if t["role"] == "assistant":
            last_assistant = t; break
    prior_result_csv = ""
    if last_assistant and last_assistant.get("result_snapshot"):
        prior_result_csv = snapshot_to_df(last_assistant["result_snapshot"]).to_csv(index=False)

    decider_payload = f"""SCHEMA:
{st.session_state.schema_txt}

CHAT HISTORY:
{chat_hist if chat_hist else '(none)'}

LAST_BA_JSON:
{prior_json}

LAST_RESULT_CSV:
{prior_result_csv}

NEW_INPUT:
{user_text}
"""
    raw_mode = openai_chat(SYSTEM_DECIDER, decider_payload, model, api_key, require_json=True)
    mode_obj = parse_ba_json(raw_mode)
    if mode_obj is None:
        # Fall back: if there is a previous assistant turn, treat as feedback when short; else new question
        is_feedback = bool(last_assistant) and len(user_text) < 140
        mode_obj = {
            "mode": "feedback" if is_feedback else "new_question",
            "normalized_question": user_text if not is_feedback else "",
            "normalized_feedback": user_text if is_feedback else ""
        }

    # 2) Run BA or Reviser
    if mode_obj.get("mode") == "feedback" and last_assistant is not None:
        payload = f"""SCHEMA for `data`:
{st.session_state.schema_txt}

CHAT HISTORY:
{chat_hist if chat_hist else '(none)'}

PRIOR BA JSON:
{prior_json}

PRIOR RESULT (CSV sample):
{prior_result_csv}

USER FEEDBACK:
{mode_obj.get('normalized_feedback','')}
"""
        raw = openai_chat(SYSTEM_BA_REVISE, payload, model, api_key, require_json=True)
    else:
        payload = f"""USER QUESTION:
{mode_obj.get('normalized_question') or user_text}

SCHEMA for `data`:
{st.session_state.schema_txt}

CHAT HISTORY (chronological):
{chat_hist if chat_hist else '(none)'}
"""
        raw = openai_chat(SYSTEM_BA_FIN, payload, model, api_key, require_json=True)

    ba = parse_ba_json(raw)
    if ba is None:
        st.error("BA did not return valid JSON.")
        if debug_raw:
            with st.expander("Raw model reply"):
                st.write(raw)
        return

    sql = (ba.get("sql") or "").strip()
    out_df = pd.DataFrame()
    if sql:
        try:
            out_df = run_duckdb(st.session_state.df, sql)
        except Exception as e:
            st.warning(f"SQL execution failed: {e}")

    # Append turns
    st.session_state.messages.append({"role":"user","content": user_text})
    turn = {
        "role": "assistant",
        "content": ba.get("narrative",""),
        "narrative": ba.get("narrative",""),
        "sql": sql if show_sql else "",
        "result_snapshot": df_to_snapshot(out_df) if not out_df.empty else {},
        "chart_suggestion": ba.get("chart_suggestion") or {},
        "followups": ba.get("followups") or []
    }
    st.session_state.messages.append(turn)
    st.session_state.last_ba_json = ba

# ===================== RENDER CHAT LOG =====================
for turn in st.session_state.messages:
    if turn["role"] == "user":
        with st.chat_message("user"):
            st.markdown(turn["content"])
    else:
        render_assistant_turn(turn)

# ===================== INPUT (single box) =====================
user_msg = st.chat_input("Ask a financial question… or type feedback like 'group by month and show margin rate'.")
if user_msg:
    handle_user_input(user_msg)
