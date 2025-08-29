import os
import io
import json
import zipfile
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import duckdb
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# OpenAI
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
DEFAULT_MODEL  = st.secrets.get("OPENAI_MODEL",  os.getenv("OPENAI_MODEL",  "gpt-4o-mini"))
try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except Exception:
    _OPENAI_AVAILABLE = False

# ==============================
# System Prompts
# ==============================
SYSTEM_AM = """You are the Analytics Manager (AM)... (same as before, omitted for brevity)"""
SYSTEM_DS = """You are the Data Scientist (DS)... (same as before)"""
SYSTEM_AM_REVIEW = """You are the AM Reviewer... (same as before)"""
SYSTEM_REVIEW = """You are a Coordinator... (same as before)"""
SYSTEM_INTENT = """You classify intent... (same as before)"""

# ==============================
# Streamlit Page Config
# ==============================
st.set_page_config(page_title="CEO ↔ AM ↔ DS — Profit Assistant", layout="wide")
st.title("🏢 CEO ↔ AM ↔ DS — Profit Improvement Assistant")

with st.sidebar:
    st.header("⚙️ Data")
    zip_file = st.file_uploader("Upload ZIP of CSVs", type=["zip"])
    st.header("🧠 Model")
    model   = st.text_input("OpenAI model", value=DEFAULT_MODEL)
    api_key = st.text_input("OPENAI_API_KEY", value=OPENAI_API_KEY, type="password")

# ==============================
# State
# ==============================
if "tables" not in st.session_state: st.session_state.tables = None
if "chat" not in st.session_state: st.session_state.chat = []
if "last_rendered_idx" not in st.session_state: st.session_state.last_rendered_idx = 0
if "last_am_json" not in st.session_state: st.session_state.last_am_json = {}
if "last_ds_json" not in st.session_state: st.session_state.last_ds_json = {}
if "last_user_prompt" not in st.session_state: st.session_state.last_user_prompt = ""

# ==============================
# Helpers
# ==============================
def ensure_openai():
    if not _OPENAI_AVAILABLE:
        raise RuntimeError("OpenAI SDK missing.")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY")
    return OpenAI(api_key=api_key)

def llm_json(system_prompt: str, user_payload: str) -> dict:
    client = ensure_openai()
    resp = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_payload},
        ],
        temperature=0.0,
    )
    return json.loads(resp.choices[0].message.content or "{}")

def load_zip_tables(file) -> Dict[str, pd.DataFrame]:
    tables = {}
    with zipfile.ZipFile(file) as z:
        for name in z.namelist():
            if not name.lower().endswith(".csv"): continue
            with z.open(name) as f:
                df = pd.read_csv(io.BytesIO(f.read()))
            key = os.path.splitext(os.path.basename(name))[0]
            tables[key] = df
    return tables

def run_duckdb_sql(tables: Dict[str, pd.DataFrame], sql: str) -> pd.DataFrame:
    con = duckdb.connect(database=":memory:")
    for name, df in tables.items():
        con.register(name, df)
    return con.execute(sql).df()

def add_msg(role, content, artifacts=None):
    st.session_state.chat.append({"role": role, "content": content, "artifacts": artifacts or {}})

def render_chat(incremental=True):
    msgs = st.session_state.chat
    start = st.session_state.last_rendered_idx if incremental else 0
    for m in msgs[start:]:
        with st.chat_message(m["role"]):
            st.write(m["content"])
            if m["artifacts"]:
                with st.expander("Artifacts", expanded=False):
                    st.json(m["artifacts"])
    st.session_state.last_rendered_idx = len(msgs)

def _sql_first(maybe_sql):
    if isinstance(maybe_sql, str): return maybe_sql.strip()
    if isinstance(maybe_sql, list):
        for s in maybe_sql:
            if isinstance(s, str) and s.strip():
                return s.strip()
    return ""

# ==============================
# Model Training (unchanged)
# ==============================
def train_model(...):
    # same as your earlier implementation
    pass

# ==============================
# AM/DS/Review Steps
# ==============================
def run_am_plan(ceo_prompt: str) -> dict:
    payload = {"ceo_question": ceo_prompt, "tables": {k: list(v.columns) for k,v in (st.session_state.tables or {}).items()}}
    am_json = llm_json(SYSTEM_AM, json.dumps(payload))
    st.session_state.last_am_json = am_json
    add_msg("am", am_json.get("am_brief",""), artifacts=am_json)
    render_chat()
    return am_json

def run_ds_step(am_json: dict) -> dict:
    ds_payload = {"am_plan": am_json.get("plan_for_ds",""), "tables": {k: list(v.columns) for k,v in (st.session_state.tables or {}).items()}}
    ds_json = llm_json(SYSTEM_DS, json.dumps(ds_payload))
    st.session_state.last_ds_json = ds_json
    add_msg("ds", ds_json.get("ds_summary",""), artifacts=ds_json)
    render_chat()
    return ds_json

def am_review_before_render(ceo_prompt: str, ds_json: dict, meta: dict):
    bundle = {"ceo_question": ceo_prompt, "am_plan": st.session_state.last_am_json, "ds_json": ds_json, "meta": meta}
    return llm_json(SYSTEM_AM_REVIEW, json.dumps(bundle))

def execute_ds_action(ds_json: dict):
    action = (ds_json.get("action") or "").lower()
    if action == "sql":
        sql = _sql_first(ds_json.get("duckdb_sql"))
        df = run_duckdb_sql(st.session_state.tables, sql)
        meta = {"rows": len(df), "cols": list(df.columns), "sample": df.head(5).to_dict("records")}
        review = am_review_before_render(st.session_state.last_user_prompt, ds_json, meta)
        add_msg("am", review.get("summary_for_ceo",""), artifacts=review)
        render_chat()
        st.dataframe(df.head(25), width="stretch")

# (Similar for overview, eda, modeling ...)

# ==============================
# Turn Coordinator
# ==============================
def run_turn_ceo(text: str):
    st.session_state.last_user_prompt = text
    am_json = run_am_plan(text)
    if am_json.get("need_more_info"):
        add_msg("am","Could you clarify?",artifacts=am_json); render_chat(); return
    ds_json = run_ds_step(am_json)
    execute_ds_action(ds_json)

# ==============================
# Data Loading
# ==============================
if zip_file and st.session_state.tables is None:
    st.session_state.tables = load_zip_tables(zip_file)
    add_msg("system", f"Loaded {len(st.session_state.tables)} tables.")
    render_chat()

# ==============================
# Chat UI
# ==============================
st.subheader("Chat")
render_chat()

user_prompt = st.chat_input("You're the CEO. Ask a question (e.g., 'What data do we have?' or 'How to improve profit?')")

if user_prompt:
    add_msg("user", user_prompt)
    render_chat()
    run_turn_ceo(user_prompt)
