import os
import io
import re
import json
import zipfile
from typing import Dict, Any, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
import duckdb
import streamlit as st


# === DS ARTIFACT CACHE (light) ===
import hashlib

if "ds_cache" not in st.session_state:
    st.session_state.ds_cache = {"profiles": {}, "featprops": {}, "clusters": {}, "colmaps": {}}

def _ds_table_signature(df):
    cols_sig = hashlib.md5(("|".join(map(str, df.columns))).encode()).hexdigest()
    sample = df.head(100)
    try:
        data_sig = hashlib.md5(pd.util.hash_pandas_object(sample, index=False).values).hexdigest()
    except Exception:
        data_sig = hashlib.md5(sample.to_csv(index=False).encode()).hexdigest()
    return f"{cols_sig}:{data_sig}"

def cached_profile(df: 'pd.DataFrame'):
    sig = _ds_table_signature(df)
    store = st.session_state.ds_cache["profiles"]
    if sig in store:
        return pd.DataFrame.from_dict(store[sig])
    prof = profile_columns(df)
    store[sig] = prof.to_dict(orient="list")
    return prof

def cached_propose_features(task: str, df: 'pd.DataFrame', allow_geo: bool=False) -> dict:
    sig = _ds_table_signature(df)
    key = (sig, task, bool(allow_geo))
    store = st.session_state.ds_cache["featprops"]
    if key in store:
        return dict(store[key])
    prof = cached_profile(df)
    prop = propose_features(task, prof, allow_geo=allow_geo)
    store[key] = dict(prop)
    return prop


from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, roc_auc_score, mean_absolute_error,
    mean_squared_error, r2_score, silhouette_score
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ---------- OpenAI setup ----------
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
DEFAULT_MODEL  = st.secrets.get("OPENAI_MODEL",  os.getenv("OPENAI_MODEL",  "gpt-4o-mini"))
try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except Exception:
    _OPENAI_AVAILABLE = False


# ======================
# System Prompts
# ======================
SYSTEM_AM = """
You are the Analytics Manager (AM). Plan how to answer the CEO’s business question using the available data.

**Action classification:** Decide the **granularity** first:
- task_mode: "single" or "multi".
- If task_mode="single" → choose exactly one next_action_type for DS from:
  `overview`, `sql`, `eda`, `calc`, `feature_engineering`, `modeling`, `explain`.
- If task_mode="multi" → propose a short `action_sequence` (2–5 steps) using ONLY those allowed actions.

**Special rule — Data Inventory:** If the CEO asks variations of “what data do we have,” set next_action_type="overview" only and instruct DS to output **first 5 rows for each table**. No EDA/FE/modeling.

**Follow-up rule:** If the CEO’s message is a follow-up asking to *explain/interpret/what features/what changed*, choose **`explain`** and DO NOT rerun modeling/EDA/SQL. Use cached results; treat the central question as context only.

Use provided `column_hints` to resolve business terms strictly to existing columns.

You can see the CEO's **current question**, the **central question of the active thread**, and **all prior questions**.

Output JSON fields:
- am_brief
- plan_for_ds
- goal  (revenue ↑ | cost ↓ | margin ↑)
- task_mode
- next_action_type
- action_sequence
- action_reason
- notes_to_ceo
- need_more_info
- clarifying_questions
Return ONLY a single JSON object. The word "json" is present here to satisfy the API requirement.
"""

SYSTEM_DS = """
You are the Data Scientist (DS). Execute the AM plan using only available columns.
You can see:
- `am_next_action_type` OR `am_action_sequence`
- `current_question`, `central_question`, and `prior_questions`
Use history to understand whether the user is giving feedback/follow-up, and keep your work anchored to the **central question**. If the current message is feedback, explain how you will revise to address it.

**Execution modes:**
- If AM provided `am_action_sequence`, you may return a matching `action_sequence` (2–5 steps). Otherwise, return a single `action` that MUST match `am_next_action_type`.
- Allowed actions: overview, sql, eda, calc, feature_engineering, modeling, explain.
- If a different approach is objectively better, explain briefly in ds_summary, but still conform to the AM’s action(s).

**Special rule — Data Inventory:** If AM indicates overview OR the CEO asked “what data do we have,” your action MUST be "overview" and output previews of the **first 5 rows for each table**.

**Explain rule:** If `action=explain`, DO NOT recompute. Read cached results (prefer clustering > supervised modeling > eda > feature_engineering > sql) and interpret.

**General safety for any theme:**
- Before modeling/clustering, browse available data (schema) and propose features with reasons.
- Dynamically exclude identifier-like, geo/postal-like, datetime, and near-constant features (names + quick stats).
- Run a preflight check; if it fails (e.g., <2 features for clustering), ask a clarifying question rather than running.

Return JSON fields:
- ds_summary
- need_more_info
- clarifying_questions
- action OR action_sequence
- duckdb_sql
- charts
- model_plan: {task, target, features, model_family, n_clusters}
- calc_description
- assumptions
Return ONLY a single JSON object. The word "json" is present here to satisfy the API requirement.
"""

SYSTEM_DS_REVISE = """
You are the Data Scientist (DS). Revise your prior plan/output based on AM critique **and the central question**. When feedback is provided, make the revision address that feedback while still serving the central question.
- Keep to ONLY existing tables/columns.
- Fix suitability issues AM mentioned.
- Keep it concise and executable.
Return JSON with the SAME schema you use normally.
Return ONLY a single JSON object. The word "json" is present here to satisfy the API requirement.
"""

SYSTEM_AM_REVIEW = """
You are the AM Reviewer. Given CEO question(s), AM plan, DS action(s), and lightweight result meta (shapes/samples/metrics),
write a short plain-language summary for the CEO and critique suitability. Be mindful of the central question and whether the latest response was a follow-up.
Return JSON fields:
- summary_for_ceo
- appropriateness_check
- gaps_or_risks
- improvements
- suggested_next_steps
- must_revise
- sufficient_to_answer
- clarification_needed
- clarifying_questions
- revision_notes
Return ONLY a single JSON object. The word "json" is present here to satisfy the API requirement.
"""

SYSTEM_REVIEW = """
You are a Coordinator. Produce a concise revision directive for AM & DS when CEO gives feedback.
Return ONLY a single JSON object. The word "json" is present here to satisfy the API requirement.
"""

SYSTEM_INTENT = """
Classify the CEO's new input relative to context.
Inputs you receive:
- previous_question
- central_question
- prior_questions
- new_text

Decide two things:
1) intent: new_request | feedback | answers_to_clarifying
2) related_to_central: true|false

Return ONLY a single JSON object like {"intent": "...", "related_to_central": true/false}. The word "json" is present here to satisfy the API requirement.
"""

SYSTEM_COLMAP = """
You map business terms in the question to columns in the provided table schemas.
Inputs: {"question": str, "tables": {table: [columns...]}}
Return JSON: { "term_to_columns": {term: [{table, column}]}, "suggested_features": [{table, column}], "notes": "" }.
Return only one JSON object (json).
"""

# ======================
# Streamlit config
# ======================
st.set_page_config(page_title="CEO ↔ AM ↔ DS", layout="wide")
st.title("🏢 CEO ↔ AM ↔ DS — Profit Assistant")

with st.sidebar:
    st.header("⚙️ Data")
    zip_file = st.file_uploader("Upload ZIP of CSVs", type=["zip"])
    st.header("🧠 Model")
    model   = st.text_input("OpenAI model", value=DEFAULT_MODEL)
    api_key = st.text_input("OPENAI_API_KEY", value=OPENAI_API_KEY, type="password")


# ======================
# State
# ======================
if "tables_raw" not in st.session_state: st.session_state.tables_raw = None
if "tables_fe"  not in st.session_state: st.session_state.tables_fe  = {}
if "tables"     not in st.session_state: st.session_state.tables     = None
if "chat"       not in st.session_state: st.session_state.chat       = []
if "last_rendered_idx" not in st.session_state: st.session_state.last_rendered_idx = 0
if "last_am_json" not in st.session_state: st.session_state.last_am_json = {}
if "last_ds_json" not in st.session_state: st.session_state.last_ds_json = {}
if "last_user_prompt" not in st.session_state: st.session_state.last_user_prompt = ""
if "current_question" not in st.session_state: st.session_state.current_question = ""
if "threads" not in st.session_state: st.session_state.threads = []  # [{central, followups: []}]
if "central_question" not in st.session_state: st.session_state.central_question = ""
if "prior_questions" not in st.session_state: st.session_state.prior_questions = []
# Caches for latest results across actions (for explain)
if "last_results" not in st.session_state:
    st.session_state.last_results = {
        "sql": None,
        "eda": None,
        "feature_engineering": None,
        "modeling": None,      # supervised summary
        "clustering": None,    # clustering report
    }

# Lightweight business term synonyms
TERM_SYNONYMS: Dict[str, List[str]] = {
    "revenue": ["revenue", "sales", "sales_amount", "net_sales", "turnover", "gmv", "amount"],
    "profit": ["profit", "net_income", "margin", "gross_profit", "operating_income"],
    "cost": ["cost", "cogs", "cost_of_goods_sold", "expense", "expenses", "opex"],
    "price": ["price", "unit_price", "sale_price", "avg_price"],
    "quantity": ["quantity", "qty", "units", "volume"],
    "customer": ["customer", "client", "buyer", "account_id", "customer_id"],
    "date": ["date", "order_date", "invoice_date", "day", "dt"],
}

# ======================
# Helpers
# ======================
def ensure_openai():
    if not _OPENAI_AVAILABLE:
        raise RuntimeError("OpenAI SDK missing.")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY")
    return OpenAI(api_key=api_key)


def llm_json(system_prompt: str, user_payload: str) -> dict:
    """Robust JSON helper with structured-mode first, then free-form parse fallback."""
    client = ensure_openai()

    sys_msg = system_prompt.strip() + "\n\nReturn ONLY a single JSON object. This line contains the word json."
    user_msg = (user_payload or "").strip() + "\n\nPlease respond with JSON only (a single object)."

    # Preferred structured call
    try:
        resp = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
        )
        return json.loads(resp.choices[0].message.content or "{}")
    except Exception as e1:
        # Fallback: free-form then parse
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": user_msg + "\n\nIf needed, wrap JSON in ```json fences."},
                ],
                temperature=0.0,
            )
            raw = resp.choices[0].message.content or ""
        except Exception as e2:
            return {"_error": str(e1), "_fallback_error": str(e2)}

        m = re.search(r"```json\s*(\{.*?\})\s*```", raw, flags=re.DOTALL | re.IGNORECASE)
        if m:
            try: return json.loads(m.group(1).strip())
            except Exception: pass
        m2 = re.search(r"(\{.*\})", raw, flags=re.DOTALL)
        if m2:
            try: return json.loads(m2.group(1).strip())
            except Exception: pass
        return {"_raw": raw, "_parse_error": True}


def load_zip_tables(file) -> Dict[str, pd.DataFrame]:
    tables = {}
    with zipfile.ZipFile(file) as z:
        for name in z.namelist():
            if not name.lower().endswith(".csv"): continue
            with z.open(name) as f:
                df = pd.read_csv(io.BytesIO(f.read()))
            key = os.path.splitext(os.path.basename(name))[0]
            i, base = 1, key
            while key in tables:
                key = f"{base}_{i}"; i += 1
            tables[key] = df
    return tables


def get_all_tables() -> Dict[str, pd.DataFrame]:
    out = {}
    if st.session_state.tables_raw:
        out.update(st.session_state.tables_raw)
    if st.session_state.tables_fe:
        out.update(st.session_state.tables_fe)
    return out


def run_duckdb_sql(sql: str) -> pd.DataFrame:
    con = duckdb.connect(database=":memory:")
    for name, df in get_all_tables().items():
        con.register(name, df)
    return con.execute(sql).df()


def add_msg(role, content, artifacts=None):
    st.session_state.chat.append({"role": role, "content": content, "artifacts": artifacts or {}})


def render_chat(incremental: bool = True):
    msgs = st.session_state.chat
    start = st.session_state.last_rendered_idx if incremental else 0
    for m in msgs[start:]:
        with st.chat_message(m["role"]):
            st.write(m["content"])
            if m.get("artifacts"):
                with st.expander("Artifacts", expanded=False):
                    st.json(m["artifacts"])
    st.session_state.last_rendered_idx = len(msgs)


def _sql_first(maybe_sql):
    if isinstance(maybe_sql, str):
        return maybe_sql.strip()
    if isinstance(maybe_sql, list):
        for s in maybe_sql:
            if isinstance(s, str) and s.strip():
                return s.strip()
    return ""


def _explicit_new_thread(text: str) -> bool:
    t = (text or "").lower()
    return bool(re.search(r"\bnot (a|an)?\s*follow[- ]?up\b", t))


def classify_intent(previous_question: str, central_question: str, prior_questions: List[str], new_text: str) -> dict:
    try:
        payload = {
            "previous_question": previous_question or "",
            "central_question": central_question or "",
            "prior_questions": prior_questions or [],
            "new_text": new_text or "",
        }
        res = llm_json(SYSTEM_INTENT, json.dumps(payload)) or {}
        intent = (res or {}).get("intent")
        related = bool((res or {}).get("related_to_central", False))
        if intent in {"new_request", "feedback", "answers_to_clarifying"}:
            return {"intent": intent, "related": related}
    except Exception:
        pass
    low = (new_text or "").lower()
    if any(w in low for w in ["that", "this", "looks", "seems", "instead", "also", "why", "how about", "can you", "explain", "interpret"]):
        return {"intent": "feedback", "related": True}
    return {"intent": "new_request", "related": False}


def build_column_hints(question: str) -> dict:
    all_tables = get_all_tables()
    struct = {t: list(df.columns) for t, df in all_tables.items()}
    hints = {"term_to_columns": {}, "suggested_features": [], "notes": ""}
    qlow = (question or "").lower()
    for term, cands in TERM_SYNONYMS.items():
        if term in qlow:
            found = []
            for table, cols in struct.items():
                for c in cands:
                    for col in cols:
                        if c == col.lower():
                            found.append({"table": table, "column": col})
            if found:
                hints["term_to_columns"][term] = found[:5]
                hints["suggested_features"].extend(found[:3])
    try:
        payload = {"question": question, "tables": struct}
        res = llm_json(SYSTEM_COLMAP, json.dumps(paylo
