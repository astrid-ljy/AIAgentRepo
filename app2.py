import os
import io
import json
from typing import Dict, Any, List

import pandas as pd
import duckdb
import streamlit as st

# =====================
# Config / Secrets
# =====================
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
MODEL = st.secrets.get("OPENAI_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

# =====================
# Lightweight LLM wrapper (stub-friendly)
# =====================
try:
    from openai import OpenAI
    _OPENAI = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
except Exception:
    _OPENAI = None


def llm_json(system_prompt: str, user_payload: dict) -> dict:
    """Call the LLM and parse a single JSON object from the response.
    If no API key, return a safe stub to keep UI working.
    """
    payload = json.dumps(user_payload, ensure_ascii=False)
    if not _OPENAI:
        # ---- SAFE STUB (keeps app functional without a key) ----
        return {
            "am_brief": "(stub) Planning next step based on your question.",
            "plan_for_ds": "(stub) Preview tables and run a simple SQL.",
            "goal": "profit",
            "next_action_type": "eda",
            "action_reason": "(stub)",
        }
    try:
        msg = _OPENAI.chat.completions.create(
            model=MODEL,
            temperature=0.2,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": payload},
            ],
        )
        text = msg.choices[0].message.content or "{}"
        # Extract the first JSON object in the content
        start = text.find("{")
        end = text.rfind("}")
        return json.loads(text[start : end + 1]) if start != -1 else {}
    except Exception as e:
        st.error(f"LLM error: {e}")
        return {}

# =====================
# System Prompts
# =====================
SYSTEM_AM = """
You are the Analytics Manager (AM). Plan how to answer the CEO’s business question using the available data.
Pick exactly one next action for the DS: one of [overview, sql, eda, calc, feature_engineering, modeling].
- overview → the CEO asked "what data do we have"; DS must preview first 5 rows per table (no EDA).
- sql/eda/feature_engineering/modeling → specify concrete guidance that uses ONLY existing tables/columns.
Use provided `column_hints` to map business terms to real columns.
Return a single JSON object with fields:
{
  am_brief: str,               # 1-2 sentences paraphrasing the CEO ask
  plan_for_ds: str,            # concrete, column-aware plan for DS
  goal: str,                   # profit intent (e.g., revenue↑, margin↑)
  next_action_type: str,       # one of [overview, sql, eda, calc, feature_engineering, modeling]
  action_reason: str           # short rationale
}
The word "json" appears here to satisfy the downstream parser; respond with a single JSON object only.
"""

SYSTEM_DS = """
You are the Data Scientist (DS). Execute the AM plan using only available columns.
Rules for this step:
- Your `action` MUST equal `am_next_action_type` for the CURRENT step (you still execute it).
- You may ALSO ask clarifying questions and propose what to do in the NEXT step.
- If the AM indicates overview or the CEO asked "what data do we have", your action MUST be "overview" and show first 5 rows of each table.
Inputs you receive:
- ceo_question: the CEO's latest question
- user_clarifications: any answers the CEO provided last turn
- am_plan: AM's guidance for you
- am_next_action_type: the action you MUST perform this step
- tables: mapping of table → list of columns
- column_hints: semantic hints mapping business terms → table.columns
- recent_summaries: last 1-2 AM/DS short messages
Return a single JSON object with fields:
{
  ds_summary: str,
  action: "overview"|"sql"|"eda"|"calc"|"feature_engineering"|"modeling",  # MUST equal am_next_action_type
  duckdb_sql: str | [str],   # SQL to run for this step (list allowed: first N-1 prep, last is base)
  charts: list | null,       # optional chart specs for EDA
  model_plan: {              # when action=="modeling"
     task: "classification"|"regression"|"clustering",
     target: str|null,
     features: [str]|null,
     model_family: str|null,
     n_clusters: int|null
  },
  # Proactivity & interaction
  questions_for_user: [str]|null,
  needs_user_input: bool|null,
  questions_for_am: [str]|null,
  proposed_next_actions: ["overview"|"sql"|"eda"|"feature_engineering"|"modeling"|"finalize"]|null,
  confidence: float|null
}
The word "json" appears here to satisfy the downstream parser; respond with a single JSON object only.
"""

SYSTEM_AM_REVIEW = """
You are the AM Reviewer. Given CEO question, AM plan, DS output, and result meta, write a short plain-language summary for the CEO and critique suitability.
Decide explicitly if the DS output suffices.
Return a JSON object with fields:
{
  summary_for_ceo: str,
  appropriateness_check: str,
  gaps_or_risks: str,
  improvements: [str],
  suggested_next_steps: [str],
  must_revise: bool,
  sufficient_to_answer: bool,
  clarification_needed: bool,
  clarifying_questions: [str],        # questions for the CEO (user)
  clarifications_for_ds: [str],       # answers to DS questions_for_am (if any)
  accept_ds_proposal: bool,           # if true, you accept DS's proposed_next_actions
  next_action_type: str|null          # next action to take (when accepting DS proposal)
}
The word "json" appears here to satisfy the downstream parser; respond with a single JSON object only.
"""

# =====================
# Data I/O helpers
# =====================
@st.cache_data(show_spinner=False)
def run_duckdb_sql(sql: str) -> pd.DataFrame:
    if not sql or not str(sql).strip():
        return pd.DataFrame()
    con = duckdb.connect(database=':memory:')
    # register RAW + FE tables
    for name, df in (st.session_state.get("tables_raw") or {}).items():
        con.register(name, df)
    for name, df in (st.session_state.get("tables_fe") or {}).items():
        con.register(name, df)
    return con.execute(sql).df()


def get_all_tables() -> Dict[str, pd.DataFrame]:
    out = {}
    out.update(st.session_state.get("tables_raw", {}))
    out.update(st.session_state.get("tables_fe", {}))
    return out


# =====================
# UI scaffolding & state
# =====================
if "chat" not in st.session_state:
    st.session_state.chat = []
if "tables_raw" not in st.session_state:
    st.session_state.tables_raw = {}
if "tables_fe" not in st.session_state:
    st.session_state.tables_fe = {}
if "awaiting_user_clarifications" not in st.session_state:
    st.session_state.awaiting_user_clarifications = False


def add_msg(role: str, content: str, artifacts: dict | None = None):
    st.session_state.chat.append({"role": role, "content": content, "artifacts": artifacts or {}})


def render_chat():
    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.write(m["content"])
            if m.get("artifacts"):
                with st.expander("Artifacts", expanded=False):
                    st.json(m["artifacts"]) 


# =====================
# Column hints (simple heuristic)
# =====================
TERM_SYNONYMS = {
    "revenue": ["revenue", "sales", "gross_revenue", "payment_value", "total_price"],
    "cost": ["cost", "cogs", "shipping_cost"],
    "product": ["product", "sku", "item"],
}

def build_column_hints(question: str) -> dict:
    struct = {t: list(df.columns) for t, df in get_all_tables().items()}
    hints = {"term_to_columns": {}, "suggested_features": [], "notes": ""}
    q = (question or "").lower()
    for term, cands in TERM_SYNONYMS.items():
        if term in q:
            found = [f"{t}.{c}" for t, cols in struct.items() for c in cols if c in cands]
            if found:
                hints["term_to_columns"][term] = found[:8]
    return hints


# =====================
# AM / DS / Review orchestration
# =====================
def run_am_plan(ceo_question: str, column_hints: dict) -> dict:
    payload = {"ceo_question": ceo_question, "column_hints": column_hints}
    am_json = llm_json(SYSTEM_AM, payload)
    add_msg("am", am_json.get("am_brief", "(no brief)"), artifacts=am_json)
    return am_json


def run_ds_step(am_json: dict, column_hints: dict) -> dict:
    ds_payload = {
        "ceo_question": st.session_state.get("last_ceo_question", ""),
        "user_clarifications": st.session_state.get("user_clarifications", ""),
        "am_plan": am_json.get("plan_for_ds", ""),
        "am_next_action_type": am_json.get("next_action_type", "eda"),
        "tables": {k: list(v.columns) for k, v in get_all_tables().items()},
        "column_hints": column_hints,
        "recent_summaries": [m.get("content", "") for m in st.session_state.chat[-4:] if m.get("role") in ("am", "ds")][-2:]
    }
    ds_json = llm_json(SYSTEM_DS, ds_payload)

    # guard: enforce allowed action & obey AM for THIS step
    allowed = {"overview", "sql", "eda", "calc", "feature_engineering", "modeling"}
    am_action = (am_json.get("next_action_type") or "").lower()
    ds_action = (ds_json.get("action") or "").lower()
    ds_json["action"] = am_action if am_action in allowed else (ds_action if ds_action in allowed else "eda")

    add_msg("ds", ds_json.get("ds_summary", "(no summary)"), artifacts={
        "action": ds_json.get("action"),
        "duckdb_sql": ds_json.get("duckdb_sql"),
        "model_plan": ds_json.get("model_plan"),
        "questions_for_user": ds_json.get("questions_for_user"),
        "questions_for_am": ds_json.get("questions_for_am"),
        "proposed_next_actions": ds_json.get("proposed_next_actions"),
    })
    return ds_json


def am_review(ceo_question: str, ds_json: dict, meta: dict) -> dict:
    bundle = {"ceo_question": ceo_question, "am_plan": st.session_state.get("last_am_json", {}), "ds_json": ds_json, "meta": meta}
    return llm_json(SYSTEM_AM_REVIEW, bundle)


# =====================
# Meta & Rendering
# =====================
def _sql_first(x):
    if isinstance(x, list):
        return x[0] if x else ""
    return (x or "").strip()


def build_meta(ds_json: dict) -> dict:
    action = (ds_json.get("action") or "").lower()
    if action == "overview":
        return {"type": "overview", "tables": {t: {"rows": len(df), "cols": list(df.columns)[:8]} for t, df in get_all_tables().items()}}
    if action == "eda":
        raw = ds_json.get("duckdb_sql")
        sqls = raw if isinstance(raw, list) else [raw]
        metas = []
        for sql in [s for s in sqls if s]:
            try:
                df = run_duckdb_sql(sql)
                metas.append({"sql": sql, "rows": len(df), "cols": list(df.columns)[:12]})
            except Exception as e:
                metas.append({"sql": sql, "error": str(e)})
        return {"type": "eda", "results": metas}
    if action in {"sql", "feature_engineering", "modeling", "calc"}:
        return {"type": action}
    return {"type": "unknown"}


def render_final(ds_json: dict):
    action = (ds_json.get("action") or "").lower()

    # ---- OVERVIEW ----
    if action == "overview":
        st.markdown("### 📊 Table Previews (first 5 rows)")
        for name, df in get_all_tables().items():
            st.markdown(f"**{name}** — rows: {len(df)}, cols: {len(df.columns)}")
            st.dataframe(df.head(5), use_container_width=True)
        return

    # ---- EDA ----
    if action == "eda":
        raw = ds_json.get("duckdb_sql")
        sqls = raw if isinstance(raw, list) else [raw]
        for i, sql in enumerate([_sql_first(s) for s in sqls][:3]):
            if not sql: 
                continue
            try:
                df = run_duckdb_sql(sql)
                st.markdown(f"### 📈 EDA Result #{i+1} (first 50 rows)")
                st.code(sql, language="sql")
                st.dataframe(df.head(50), use_container_width=True)
                # cache EDA outputs for downstream modeling
                st.session_state.tables_fe[f"eda_result_{i+1}"] = df.copy()
            except Exception as e:
                st.error(f"EDA SQL failed: {e}")
        return

    # ---- SQL ----
    if action == "sql":
        sql = _sql_first(ds_json.get("duckdb_sql"))
        if not sql:
            st.info("No SQL provided.")
            return
        try:
            out = run_duckdb_sql(sql)
            st.markdown("### 🧮 SQL Results (first 25 rows)")
            st.code(sql, language="sql")
            st.dataframe(out.head(25), use_container_width=True)
        except Exception as e:
            st.error(f"SQL failed: {e}")
        return

    # ---- FEATURE ENGINEERING ----
    if action == "feature_engineering":
        sql = _sql_first(ds_json.get("duckdb_sql"))
        base = run_duckdb_sql(sql) if sql else next(iter(get_all_tables().values()), pd.DataFrame()).copy()
        st.markdown("### 🧱 Feature Engineering Base (first 20 rows)")
        st.dataframe(base.head(20), use_container_width=True)
        st.session_state.tables_fe["feature_base"] = base
        return

    # ---- MODELING (prep list allowed, reuse cached frames) ----
    if action == "modeling":
        raw_sql = ds_json.get("duckdb_sql")
        if isinstance(raw_sql, list) and raw_sql:
            for prep_sql in raw_sql[:-1]:
                if prep_sql and str(prep_sql).strip():
                    _ = run_duckdb_sql(prep_sql)
            sql = raw_sql[-1]
        else:
            sql = _sql_first(raw_sql)

        plan = ds_json.get("model_plan") or {}
        task = (plan.get("task") or "classification").lower()
        target = plan.get("target")

        base = run_duckdb_sql(sql) if sql else None
        if base is None or base.empty:
            # Prefer previously cached FE/EDA snapshots
            if "feature_base" in st.session_state.tables_fe:
                base = st.session_state.tables_fe["feature_base"].copy()
            else:
                eda_keys = [k for k in st.session_state.tables_fe.keys() if k.startswith("eda_result_")]
                if eda_keys:
                    base = st.session_state.tables_fe[sorted(eda_keys)[-1]].copy()
        if base is None or base.empty:
            # fall back to any available table
            base = next(iter(get_all_tables().values()), pd.DataFrame()).copy()

        # For demo purposes, show what would be modeled
        st.markdown("### 🤖 Modeling Stub")
        st.write({"task": task, "target": target, "shape": base.shape})
        st.dataframe(base.head(15), use_container_width=True)
        return


# =====================
# Turn runner
# =====================
def run_turn_ceo(new_text: str):
    # If awaiting clarifications, capture them and continue
    if st.session_state.get("awaiting_user_clarifications"):
        st.session_state.user_clarifications = new_text
        st.session_state.awaiting_user_clarifications = False
        add_msg("s", "Got your clarifications. I’ll incorporate them now.")

    # Persist last question for DS visibility
    st.session_state.last_ceo_question = new_text

    # Column hints
    col_hints = build_column_hints(new_text)

    # AM plan
    am_json = run_am_plan(new_text, col_hints)
    st.session_state.last_am_json = am_json

    # If we have an override from a prior review, apply & clear
    if st.session_state.get("next_action_override"):
        am_json["next_action_type"] = st.session_state.pop("next_action_override")

    # DS step
    ds_json = run_ds_step(am_json, col_hints)

    # If DS asked the user, surface and possibly pause
    qs_user = ds_json.get("questions_for_user") or []
    needs_user = bool(ds_json.get("needs_user_input"))
    if qs_user:
        add_msg("d", "I need a quick clarification from you before proceeding:", artifacts={"questions_for_user": qs_user})
        st.session_state.awaiting_user_clarifications = True
        if needs_user:
            return  # wait for the user's next message

    # Build meta & AM review
    meta = build_meta(ds_json)
    review = am_review(new_text, ds_json, meta)

    # Surface reviewer content
    add_msg("am", review.get("summary_for_ceo", ""), artifacts={
        "appropriateness_check": review.get("appropriateness_check"),
        "gaps_or_risks": review.get("gaps_or_risks"),
        "improvements": review.get("improvements"),
        "suggested_next_steps": review.get("suggested_next_steps"),
        "must_revise": review.get("must_revise"),
        "sufficient_to_answer": review.get("sufficient_to_answer"),
    })

    # Answers to DS, if any
    if review.get("clarifications_for_ds"):
        add_msg("am", "Answers to DS questions.", artifacts={"clarifications_for_ds": review.get("clarifications_for_ds")})

    # If AM accepts DS proposal, steer next turn
    if review.get("accept_ds_proposal"):
        st.session_state.next_action_override = review.get("next_action_type")

    # If AM needs clarification from CEO, ask and pause
    if review.get("clarification_needed") and (review.get("clarifying_questions") or []):
        add_msg("am", "Before proceeding, could you clarify:")
        for q in (review.get("clarifying_questions") or [])[:3]:
            add_msg("am", f"• {q}")
        st.session_state.awaiting_user_clarifications = True
        return

    # Render what we have for this step
    render_final(ds_json)


# =====================
# App UI
# =====================
st.set_page_config(page_title="CEO ↔ AM ↔ DS — Profit Assistant", layout="wide")
st.title("🏢 CEO ↔ AM ↔ DS — Profit Assistant")

st.sidebar.header("Data")
uploads = st.sidebar.file_uploader("Upload CSV(s)", type=["csv", "txt"], accept_multiple_files=True)
if uploads:
    for f in uploads:
        try:
            df = pd.read_csv(f)
            key = f.name.split(".")[0]
            # ensure unique name
            base, i, name = key, 1, key
            while name in st.session_state.tables_raw:
                name = f"{base}_{i}"; i += 1
            st.session_state.tables_raw[name] = df
        except Exception as e:
            st.sidebar.error(f"Failed to read {f.name}: {e}")
    st.sidebar.success(f"Loaded {len(uploads)} file(s). Tables: {', '.join(st.session_state.tables_raw.keys())}")

# show current tables
if st.sidebar.checkbox("Show table catalog"):
    for t, df in get_all_tables().items():
        st.sidebar.write(f"• **{t}** — {df.shape}")

st.subheader("Chat")
render_chat()

user_prompt = st.chat_input("You're the CEO. Ask a question (e.g., 'What data do we have?' or 'Can we cluster products?')")
if user_prompt:
    add_msg("user", user_prompt)
    run_turn_ceo(user_prompt)
    render_chat()
