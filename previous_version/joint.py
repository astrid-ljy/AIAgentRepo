
import os
import json
import time
import pandas as pd
import duckdb
import streamlit as st

# ========== CONFIG ==========
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

SYSTEM_DS = r'''
You are “Telco/Data Scientist Agent.”
Task: Given a user/BA spec and the SCHEMA (table name `data`), output a SINGLE DuckDB SQL that answers the spec.
Rules:
- Use ONLY existing columns; table name MUST be `data`.
- Be precise and auditable (counts, %, grouped stats).
- If something requested is impossible with schema, set sql="" and explain briefly why in narrative.
Output (STRICT) inside ONE ```json block:
{
  "sql": "DuckDB SQL or empty string",
  "narrative": "1-2 sentences on what the SQL returns",
  "chart_suggestion": {"type": "bar|line|area|scatter|none", "x": "col or null", "y": "col or null"}
}
'''

SYSTEM_BA_PLANNER = r'''
You are “Business Analyst Agent (Planner).”
Task: Read USER QUESTION and SCHEMA. Clarify the business intent and produce a concrete analytic SPEC that the DS agent can implement.
- Use ONLY columns that exist; if asked for unavailable concept (e.g., Region), pick the closest valid columns (e.g., State/Country) and state the substitution.
- Define metric(s), group-bys, filters, and sort order when appropriate.
Output inside ONE ```json block:
{
  "spec": "plain English spec the DS should implement",
  "tl_dr": "1-2 sentence expected business takeaway to aim for"
}
'''

SYSTEM_BA_INTERPRET = r'''
You are “Business Analyst Agent (Interpreter).”
Task: Given RESULT TABLE (CSV sample) and the DS narrative, write a concise, business-oriented interpretation with 3-6 bullets (findings, segments, risks).
Do not speculate beyond the table. No causal claims.
Output inside ONE ```json block:
{
  "insight_bullets": ["...", "...", "..."]
}
'''

# Optional OpenAI SDK
try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except Exception:
    _OPENAI_AVAILABLE = False

# ========== UI ==========
st.set_page_config(page_title="Agent Orchestrator: BA ↔ DS", layout="wide")
st.title("🤝 Agent Orchestrator — Business Analyst ↔ Data Scientist")

with st.sidebar:
    st.header("⚙️ Setup")
    uploaded = st.file_uploader("Upload CSV", type=["csv"], accept_multiple_files=False)
    model = st.text_input("OpenAI model", value=DEFAULT_MODEL)
    api_key = st.text_input("OpenAI API Key", value=OPENAI_API_KEY, type="password")
    feedback_loop = st.checkbox("Enable 1 feedback loop (BA critiques DS then DS revises)", value=True)
    show_schema = st.checkbox("Show inferred schema", value=True)

st.divider()

user_q = st.text_area("Business question (for BA Planner)", height=100,
                      placeholder="e.g., What's churn rate by Contract and InternetService?")
run = st.button("Run BA → DS → Execute → BA Interpret", type="primary", use_container_width=True)

# ========== Helpers ==========
def infer_schema(df: pd.DataFrame, sample_rows: int = 5) -> str:
    lines = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        sample_vals = df[col].dropna().unique()[:sample_rows]
        sample_list = ", ".join([repr(v)[:60] for v in sample_vals])
        lines.append(f"- {col} ({dtype})  samples: {sample_list}")
    return "\n".join(lines)

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
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""

def run_duckdb(df: pd.DataFrame, sql: str) -> pd.DataFrame:
    con = duckdb.connect(database=":memory:")
    con.register("data", df)
    return con.execute(sql).df()

# Diagnostics container
diag = {"planner_raw":"", "planner_json":{}, "ds_raw":"", "ds_json":{}, "ds_sql":"", "ds2_raw":"", "ds2_json":{}, "error":""}

# ========== Pipeline ==========
if run:
    status = st.status("🔁 Orchestrating BA ↔ DS …", state="running")
    try:
        if uploaded is None:
            raise RuntimeError("Please upload a CSV first.")
        df = pd.read_csv(uploaded)
        schema_txt = infer_schema(df)
        if show_schema:
            with st.expander("📜 Inferred Schema", expanded=False):
                st.code(schema_txt, language="markdown")

        # ---- 1) BA Planner
        status.update(label="🧭 BA Planner drafting a spec…", state="running")
        planner_user = f"USER QUESTION:\n{user_q}\n\nSCHEMA for `data`:\n{schema_txt}"
        raw_plan = openai_chat(SYSTEM_BA_PLANNER, planner_user, model, api_key)
        diag["planner_raw"] = raw_plan
        j1 = extract_json_block(raw_plan)
        if not j1:
            raise RuntimeError("BA Planner did not return JSON.")
        plan = json.loads(j1)
        diag["planner_json"] = plan

        st.subheader("BA Plan")
        st.write(plan.get("spec","(no spec)"))
        st.caption(plan.get("tl_dr",""))

        # ---- 2) DS generates SQL
        status.update(label="🧪 DS generating SQL…", state="running")
        ds_user = f"SPEC from BA:\n{plan.get('spec','')}\n\nSCHEMA for `data`:\n{schema_txt}"
        raw_ds = openai_chat(SYSTEM_DS, ds_user, model, api_key)
        diag["ds_raw"] = raw_ds
        j2 = extract_json_block(raw_ds)
        if not j2:
            raise RuntimeError("DS did not return JSON.")
        ds = json.loads(j2)
        diag["ds_json"] = ds
        sql = (ds.get("sql") or "").strip()
        diag["ds_sql"] = sql

        # ---- 2.5) Optional Feedback loop
        if feedback_loop and sql:
            status.update(label="🔍 BA quick critique of SQL spec…", state="running")
            critique_prompt = (
                "You are BA. Briefly review this DS plan and SQL for business alignment with the spec. "
                "If revision is needed, provide a single sentence 'revise:' instruction; else 'approve'.\n\n"
                f"SPEC:\n{plan.get('spec','')}\n\nDS RESPONSE:\n{j2}"
            )
            crit_raw = openai_chat(
                "You are a very concise BA reviewer. Respond in ONE paragraph.", critique_prompt, model, api_key
            )
            st.subheader("BA Review")
            st.write(crit_raw)

            if "revise:" in crit_raw.lower():
                status.update(label="🛠️ DS revising based on BA feedback…", state="running")
                ds_user2 = f"{ds_user}\n\nBA FEEDBACK:\n{crit_raw}\nReturn full JSON again."
                raw_ds2 = openai_chat(SYSTEM_DS, ds_user2, model, api_key)
                diag["ds2_raw"] = raw_ds2
                j2b = extract_json_block(raw_ds2)
                if j2b:
                    ds2 = json.loads(j2b)
                    diag["ds2_json"] = ds2
                    sql = (ds2.get('sql') or sql).strip()

        # ---- 3) Execute SQL
        if not sql:
            status.update(label="ℹ️ DS could not form safe SQL.", state="complete")
            st.info(ds.get("narrative","DS could not form a safe SQL with given schema."))
        else:
            status.update(label="🏃 Running SQL…", state="running")
            out = run_duckdb(df, sql)
            status.update(label="✅ Execution complete", state="complete")

            st.subheader("Executed SQL")
            st.code(sql, language="sql")

            st.subheader("Result")
            st.dataframe(out, use_container_width=True)

            # ---- 4) BA interprets results
            status.update(label="🗣️ BA interpreting results…", state="running")
            sample_csv = out.head(min(50, len(out))).to_csv(index=False)
            interp_user = f"DS narrative: {ds.get('narrative','')}\n\nRESULT TABLE (CSV):\n{sample_csv}"
            raw_interp = openai_chat(SYSTEM_BA_INTERPRET, interp_user, model, api_key)
            j3 = extract_json_block(raw_interp)
            insights = []
            if j3:
                insights = json.loads(j3).get("insight_bullets", [])
            st.subheader("BA Insights")
            for b in insights or ["(no insights)"]:
                st.write("- " + b)

            # Quick chart if model suggested
            cs = (diag.get("ds2_json") or ds).get("chart_suggestion") or {}
            if isinstance(cs, dict) and cs.get("type","none") != "none":
                st.subheader("Quick Chart")
                x = cs.get("x"); y = cs.get("y")
                ct = cs.get("type")
                if x in out.columns and y in out.columns:
                    if ct == "bar":
                        st.bar_chart(out.set_index(x)[y])
                    elif ct == "line":
                        st.line_chart(out.set_index(x)[y])
                    elif ct == "area":
                        st.area_chart(out.set_index(x)[y])
                    elif ct == "scatter":
                        st.scatter_chart(out, x=x, y=y)

        # ---- Diagnostics
        with st.expander("🛠️ Orchestrator Diagnostics"):
            st.write("BA Planner (raw):"); st.code(diag["planner_raw"] or "", language="markdown")
            st.write("BA Planner (json):"); st.json(diag["planner_json"] or {})
            st.write("DS (raw):"); st.code(diag["ds_raw"] or "", language="markdown")
            st.write("DS (json):"); st.json(diag["ds_json"] or {})
            if diag.get("ds2_raw"):
                st.write("DS (revised raw):"); st.code(diag["ds2_raw"], language="markdown")
                st.write("DS (revised json):"); st.json(diag["ds2_json"] or {})
    except Exception as e:
        status.update(label="❌ Error", state="error")
        st.error(str(e))
