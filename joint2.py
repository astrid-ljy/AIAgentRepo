import os
import json
import time
import pandas as pd
import duckdb
import streamlit as st

# =============================
# Cloud-friendly config
# =============================
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
DEFAULT_MODEL  = st.secrets.get("OPENAI_MODEL",  os.getenv("OPENAI_MODEL",  "gpt-4o-mini"))

# Optional OpenAI SDK (app degrades with clear error if missing)
try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except Exception:
    _OPENAI_AVAILABLE = False

# =============================
# Agent Prompts
# =============================
SYSTEM_BA_PLANNER = r'''
You are “Business Analyst Agent (Planner).”
Task: Read USER QUESTION and SCHEMA. Clarify the business intent and produce a concrete analytic SPEC for the DS to implement.
Rules:
- Use ONLY columns that exist; if a requested concept is absent (e.g., Region), propose the closest exact column(s) from the schema (e.g., State or Country) and state the substitution.
- Define metric(s), group-bys, filters, and sort order when appropriate.
Output ONE ```json block:
{
  "spec": "plain English spec the DS should implement",
  "tl_dr": "1-2 sentence expected takeaway to aim for"
}
'''

SYSTEM_DS = r'''
You are “Data Scientist Agent.”
Task: Given a BA SPEC and the SCHEMA (single table `data`), return a SINGLE DuckDB SQL that answers the spec.
Rules:
- Only reference columns that exist; table name MUST be `data`.
- Prefer precise, auditable metrics (counts, %, grouped stats). No causal claims.
- If spec is impossible with the schema, set "sql" = "" and explain briefly in "narrative".
Output ONE ```json block:
{
  "sql": "DuckDB SQL or empty string",
  "narrative": "1-2 sentences describing the result",
  "chart_suggestion": {"type": "bar|line|area|scatter|none", "x": "col or null", "y": "col or null"}
}
'''

SYSTEM_BA_INTERPRET = r'''
You are “Business Analyst Agent (Interpreter).”
Task: Given RESULT TABLE (CSV sample) and DS narrative, write a concise, business-oriented interpretation with 3-6 bullets (findings, segments, risks).
Do not speculate beyond the table; no causal claims.
Output ONE ```json block:
{
  "insight_bullets": ["...", "...", "..."]
}
'''

SYSTEM_BA_REVIEW = r'''
You are “BA Reviewer.”
Task: Briefly review the DS plan/SQL for business alignment with the BA SPEC.
Return ONE line: either "approve" OR "revise: <1-sentence instruction>".
'''

# =============================
# Streamlit page setup
# =============================
st.set_page_config(page_title="Dual-Agent Market Simulator (Baby-BGI)", layout="wide")
st.title("🍼 Baby-BGI: Dual-Agent Market Simulator")

with st.sidebar:
    st.header("⚙️ Setup")
    uploaded = st.file_uploader("Upload CSV", type=["csv"], accept_multiple_files=False)
    model = st.text_input("OpenAI model", value=DEFAULT_MODEL)
    api_key = st.text_input("OpenAI API Key", value=OPENAI_API_KEY, type="password")
    feedback_loop = st.checkbox("Enable 1 feedback loop (BA critiques DS → DS revises)", value=True)
    show_schema = st.checkbox("Show inferred schema", value=True)

    st.markdown("---")
    st.subheader("🎯 Simulation")
    # Initialize sim state
    if "sim" not in st.session_state:
        st.session_state.sim = {
            "difficulty": 1,
            "round": 1,
            "score": 0,
            "history": [],  # list of turns with artifacts and feedback
            "memory": {
                "preferred_dims": set(),
                "preferred_metrics": set(),
                "bad_joins": set(),
                "chart_prefs": {}
            }
        }
    sim = st.session_state.sim
    st.metric("Difficulty", sim["difficulty"])
    st.metric("Round", sim["round"])
    st.metric("Score", sim["score"])

# =============================
# Helpers
# =============================
def ensure_openai():
    if not _OPENAI_AVAILABLE:
        raise RuntimeError("OpenAI SDK not installed. Add 'openai' to requirements.txt")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY (set in Streamlit Secrets or env).")
    return OpenAI(api_key=api_key)

def infer_schema(df: pd.DataFrame, sample_rows: int = 5) -> str:
    lines = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        sample_vals = df[col].dropna().unique()[:sample_rows]
        sample_list = ", ".join([repr(v)[:60] for v in sample_vals])
        lines.append(f"- {col} ({dtype})  samples: {sample_list}")
    return "\n".join(lines)

def openai_chat(system_prompt: str, user_payload: str) -> str:
    client = ensure_openai()
    resp = client.chat.completions.create(
        model=model,
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
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""

def run_duckdb(df: pd.DataFrame, sql: str) -> pd.DataFrame:
    con = duckdb.connect(database=":memory:")
    con.register("data", df)
    return con.execute(sql).df()

def add_msg(role, content, artifacts=None):
    if "chat" not in st.session_state:
        st.session_state.chat = []
    st.session_state.chat.append({"role": role, "content": content, "artifacts": artifacts or {}})

def render_chat():
    if "chat" not in st.session_state:
        st.session_state.chat = []
    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.write(m["content"])
            if m["artifacts"]:
                with st.expander("Artifacts", expanded=False):
                    for k, v in m["artifacts"].items():
                        st.write(f"**{k}**")
                        if isinstance(v, str) and v.strip().upper().startswith("SELECT"):
                            st.code(v, language="sql")
                        else:
                            try:
                                st.json(v)
                            except Exception:
                                st.write(v)

def reflect_memory_from_feedback(turn):
    mem = st.session_state.sim["memory"]
    fb = turn.get("feedback", {})
    if fb.get("up"):
        st.session_state.sim["score"] += 1
    # Learn chart preference
    ds = turn.get("artifacts", {}).get("DS JSON", {})
    cs = (ds or {}).get("chart_suggestion") or {}
    if isinstance(cs, dict) and cs.get("type"):
        mem["chart_prefs"][cs["type"]] = mem["chart_prefs"].get(cs["type"], 0) + 1

def evaluate_and_progress(df_out):
    success = len(df_out) > 0
    msg = "Great! Scenario cleared — next one unlocked." if success else "No luck yet — BA will propose a simpler segmentation next."
    if success:
        st.session_state.sim["difficulty"] += 1
        st.session_state.sim["round"] += 1
    else:
        st.session_state.sim["round"] += 1
    add_msg("system", msg)

# =============================
# Chat area (top) + input
# =============================
st.subheader("Chat")
render_chat()

prompt = st.chat_input("Ask a business question (or describe a decision you want to try)…")

# =============================
# Main loop: BA → DS → Execute → BA Interpret (+ optional feedback loop)
# =============================
if prompt:
    add_msg("user", prompt)
    try:
        if uploaded is None:
            raise RuntimeError("Please upload a CSV in the sidebar.")
        df = pd.read_csv(uploaded)
        schema_txt = infer_schema(df)

        if show_schema:
            with st.expander("📜 Inferred Schema", expanded=False):
                st.code(schema_txt, language="markdown")

        # ---- BA Planner
        status = st.status("🧭 BA Planner drafting a spec…", state="running")
        planner_user = f"USER QUESTION:\n{prompt}\n\nSCHEMA for `data`:\n{schema_txt}"
        raw_plan = openai_chat(SYSTEM_BA_PLANNER, planner_user)
        j_plan = extract_json_block(raw_plan)
        plan = json.loads(j_plan)
        add_msg("assistant", "BA Plan", artifacts={"BA Plan JSON": plan})
        status.update(label="🧪 DS generating SQL…", state="running")

        # ---- DS Agent
        ds_user = f"SPEC from BA:\n{plan.get('spec','')}\n\nSCHEMA for `data`:\n{schema_txt}"
        raw_ds = openai_chat(SYSTEM_DS, ds_user)
        j_ds = extract_json_block(raw_ds)
        ds = json.loads(j_ds)
        sql = (ds.get("sql") or "").strip()
        add_msg("assistant", ds.get("narrative", "DS result narrative."), artifacts={"DS JSON": ds, "SQL": sql})

        # ---- Feedback loop
        if feedback_loop and sql:
            review_payload = f"SPEC:\n{plan.get('spec','')}\n\nDS RESPONSE JSON:\n{j_ds}\n\nReturn 'approve' OR 'revise: <short instruction>'."
            raw_review = openai_chat(SYSTEM_BA_REVIEW, review_payload)
            add_msg("assistant", "BA Review", artifacts={"Review": raw_review})
            if "revise:" in raw_review.lower():
                status.update(label="🛠️ DS revising based on BA feedback…", state="running")
                ds_user2 = f"{ds_user}\n\nBA FEEDBACK:\n{raw_review}\nReturn full JSON again."
                raw_ds2 = openai_chat(SYSTEM_DS, ds_user2)
                j_ds2 = extract_json_block(raw_ds2)
                ds2 = json.loads(j_ds2)
                sql = (ds2.get("sql") or sql).strip()
                add_msg("assistant", "DS Revision Applied", artifacts={"DS Revised JSON": ds2, "SQL": sql})

        # ---- Execute SQL
        if not sql:
            status.update(label="ℹ️ DS could not form safe SQL.", state="complete")
            add_msg("assistant", "DS could not form a safe SQL with given schema.", artifacts={})
        else:
            status.update(label="🏃 Running SQL…", state="running")
            out = run_duckdb(df, sql)
            status.update(label="✅ Execution complete", state="complete")
            add_msg("assistant", "Query Results", artifacts={"Rows": len(out), "Preview": out.head(20).to_dict(orient="records")})

            # ---- BA Interpreter
            sample_csv = out.head(min(50, len(out))).to_csv(index=False)
            interp_user = f"DS narrative: {ds.get('narrative','')}\n\nRESULT TABLE (CSV):\n{sample_csv}"
            raw_interp = openai_chat(SYSTEM_BA_INTERPRET, interp_user)
            j_interp = extract_json_block(raw_interp)
            insights = json.loads(j_interp).get("insight_bullets", [])
            add_msg("assistant", "BA Insights", artifacts={"insights": insights})

            # ---- Feedback controls
            st.markdown("#### Feedback on this turn")
            col1, col2 = st.columns([1,3])
            with col1:
                up = st.button("👍 Helpful", key=f"up_{time.time()}")
                down = st.button("👎 Not helpful", key=f"down_{time.time()}")
            with col2:
                fb = st.text_input("Optional feedback (what to change next?)", key=f"fb_{time.time()}")

            turn = {
                "user": prompt,
                "artifacts": {"BA Plan JSON": plan, "DS JSON": ds, "SQL": sql, "ResultRows": len(out), "Insights": insights},
                "feedback": {"up": bool(up), "down": bool(down), "note": fb} if (up or down or fb) else {}
            }
            st.session_state.sim["history"].append(turn)
            reflect_memory_from_feedback(turn)

            # ---- Scenario progression
            evaluate_and_progress(out)

        status.update(label="✅ Round complete", state="complete")

    except Exception as e:
        add_msg("assistant", f"Error: {type(e).__name__}: {e}")
        st.error(str(e))

with st.expander("🛠️ Diagnostics & Tips", expanded=False):
    st.write("- The app re-reads your CSV every question to keep schema awareness fresh.")
    st.write("- If DS invents a column, use feedback to nudge.")
    st.write("- On Streamlit Cloud, set OPENAI_API_KEY in the app’s Secrets.")
