import os
import io
import json
import time
import zipfile
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import duckdb
import streamlit as st

# ===== Cloud-friendly config =====
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
DEFAULT_MODEL  = st.secrets.get("OPENAI_MODEL",  os.getenv("OPENAI_MODEL",  "gpt-4o-mini"))

try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except Exception:
    _OPENAI_AVAILABLE = False

# ===== ML stack =====
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# =============================
# System prompts
# =============================
SYSTEM_AM = r"""
You are the **Analytics Manager (AM)** working with a CEO (non-technical) and a Data Scientist (DS).
Responsibilities:
- Face the CEO: restate their request in business terms aiming at PROFIT improvement (revenue ↑, cost ↓, margin ↑).
- If the CEO message is ambiguous or missing info, ASK 1–3 clarifying questions (set need_more_info=true).
- Design an analytics plan for DS using ONLY available tables/columns.
- Decide the best next action for DS: overview|sql|calc|feature_engineering|modeling|what_if.
- After DS executes, evaluate results and provide concise feedback to DS.

Return ONLY valid JSON:
{
  "am_brief": "1-3 sentences for the CEO",
  "plan_for_ds": "grounded, step-by-step DS instructions using available tables/columns",
  "goal": "profit proxy to improve",
  "next_action_type": "overview|sql|calc|feature_engineering|modeling|what_if",
  "notes_to_ceo": "1-2 plain-language notes",
  "need_more_info": true|false,
  "clarifying_questions": ["q1","q2"]
}
"""

SYSTEM_DS = r"""
You are the **Data Scientist (DS)** collaborating with the AM for a CEO.

FIRST RUN behavior:
- Inspect ALL tables: rows/cols, missingness, sample rows. Propose light cleaning (type casting, trimming, NA fixes) and apply.

Per AM plan:
- Decide whether to run SQL/aggregation, feature engineering, modeling, or a simple calculation.
- When SQL, write DuckDB SQL referencing tables EXACTLY by their names.
- When modeling, define: task ("classification"|"regression"), target, features, model family ("logistic_regression"|"random_forest"|"linear_regression"|"random_forest_regressor"). (Python will actually train.)

Return ONLY valid JSON:
{
  "ds_summary": "what you did/are doing",
  "action": "overview|sql|calc|feature_engineering|modeling",
  "duckdb_sql": "SQL if action uses SQL else ''",
  "model_plan": {
    "task": "classification|regression|null",
    "target": "existing column or null",
    "features": ["col1","col2",...],
    "model_family": "logistic_regression|random_forest|linear_regression|random_forest_regressor|null"
  },
  "calc_description": "if action=calc, describe the calculation",
  "assumptions": "assumptions you made"
}
"""

SYSTEM_REVIEW = r"""
You are a **Coordinator**. Given CEO feedback and the prior AM+DS responses, produce a concise revision directive.

Return ONLY valid JSON:
{
  "revision_directive": "single concise paragraph for AM and DS with concrete changes"
}
"""

SYSTEM_INTENT = r"""
You classify the CEO's latest message.
Given the chat history and the latest CEO message, decide if it's:
- "new_request": a new question or task,
- "feedback": comments meant to revise or improve the LAST ANSWER,
- "answers_to_clarifying": replying to clarifying questions the AM asked.

Return ONLY JSON:
{"intent":"new_request|feedback|answers_to_clarifying"}
"""

# =============================
# Streamlit page
# =============================
st.set_page_config(page_title="CEO ↔ AM ↔ DS — Profit Assistant", layout="wide")
st.title("🏢 CEO ↔ AM ↔ DS — Profit Improvement Assistant")

with st.sidebar:
    st.header("⚙️ Data")
    zip_file = st.file_uploader("Upload ZIP of CSVs", type=["zip"], accept_multiple_files=False)
    st.caption("Each CSV becomes a table (name = filename without extension).")

    st.header("🧠 Model")
    model   = st.text_input("OpenAI model", value=DEFAULT_MODEL)
    api_key = st.text_input("OPENAI_API_KEY", value=OPENAI_API_KEY, type="password")
    show_schema = st.checkbox("Show table schemas", value=False)

    st.markdown("---")
    st.subheader("Preset questions")
    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = ""
    c1, c2 = st.columns(2)
    if c1.button("What data do we have?"):
        st.session_state.pending_prompt = "What data do we have? Please walk me through each table briefly with previews."
    if c2.button("Where can we improve profit quickly?"):
        st.session_state.pending_prompt = "Where can we improve profit quickly? Start with a high-level plan."

# =============================
# State
# =============================
if "tables" not in st.session_state: st.session_state.tables = None
if "ds_first_overview" not in st.session_state: st.session_state.ds_first_overview = None
if "chat" not in st.session_state: st.session_state.chat = []
if "last_am_json" not in st.session_state: st.session_state.last_am_json = {}
if "last_ds_json" not in st.session_state: st.session_state.last_ds_json = {}
if "last_user_prompt" not in st.session_state: st.session_state.last_user_prompt = ""

# =============================
# Helpers
# =============================
def ensure_openai():
    if not _OPENAI_AVAILABLE:
        raise RuntimeError("OpenAI SDK not installed. Add 'openai' to requirements.txt")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY (set in Secrets or env).")
    return OpenAI(api_key=api_key)

def llm_json(system_prompt: str, user_payload: str) -> dict:
    client = ensure_openai()
    # JSON mode first
    try:
        resp = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt + "\n\nReturn ONLY a valid JSON object."},
                {"role": "user", "content": user_payload},
            ],
            temperature=0.0,
        )
        return json.loads(resp.choices[0].message.content or "{}")
    except Exception:
        # Fallback fence
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt + "\n\nReturn a single JSON object in ```json fences."},
                    {"role": "user", "content": user_payload},
                ],
                temperature=0.0,
            )
            raw = resp.choices[0].message.content or ""
        except Exception as e2:
            st.error(f"OpenAI call failed: {e2}")
            return {"_error": str(e2)}
        import re
        m = re.search(r"```json\s*(\{.*?\})\s*```", raw, flags=re.DOTALL|re.IGNORECASE)
        if not m:
            return {"_raw": raw, "_parse_error": True}
        try:
            return json.loads(m.group(1).strip())
        except Exception as e3:
            st.error(f"JSON parsing failed: {e3}")
            return {"_raw": raw, "_parse_error": True}

def load_zip_tables(file) -> Dict[str, pd.DataFrame]:
    tables = {}
    with zipfile.ZipFile(file) as z:
        for name in z.namelist():
            if not name.lower().endswith(".csv"): continue
            with z.open(name) as f:
                b = f.read()
                try:
                    df = pd.read_csv(io.BytesIO(b))
                except Exception:
                    df = pd.read_csv(io.BytesIO(b), encoding="latin1")
            key = os.path.splitext(os.path.basename(name))[0]
            i, base = 1, key
            while key in tables:
                key = f"{base}_{i}"; i += 1
            tables[key] = df
    return tables

def duckdb_connect_with_tables(tables: Dict[str, pd.DataFrame]):
    con = duckdb.connect(database=":memory:")
    for name, df in tables.items():
        con.register(name, df)
    return con

def run_duckdb_sql(tables: Dict[str, pd.DataFrame], sql: str) -> pd.DataFrame:
    con = duckdb_connect_with_tables(tables)
    return con.execute(sql).df()

def summarize_tables(tables: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    out = {}
    for name, df in tables.items():
        miss = (df.isna().sum() / max(len(df),1)).round(3).to_dict()
        out[name] = {
            "rows": int(len(df)),
            "cols": int(len(df.columns)),
            "missing_fraction": miss,
            "sample": df.head(5).to_dict(orient="records"),  # first 5 rows
            "schema": "\n".join([f"- {c} ({str(df[c].dtype)})" for c in df.columns])
        }
    return out

def add_msg(role, content, artifacts=None):
    st.session_state.chat.append({"role": role, "content": content, "artifacts": artifacts or {}})

def render_chat():
    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.write(m["content"])
            if m["artifacts"]:
                with st.expander("Artifacts", expanded=False):
                    for k, v in m["artifacts"].items():
                        st.write(f"**{k}**")
                        if isinstance(v, str) and v.strip().upper().startswith("SELECT"):
                            st.code(v, language="sql")
                        elif isinstance(v, (dict, list)):
                            st.json(v)
                        else:
                            st.write(v)

# --- Processing helpers (progress & chat markers) ---
def processing_start(title="Agents are working..."):
    add_msg("system", f"⏳ {title}")
    st_placeholder.empty(); render_chat()
    return st.status(title, state="running")

def processing_update(status_obj, label):
    status_obj.update(label=label, state="running")
    add_msg("system", f"⏳ {label}")
    st_placeholder.empty(); render_chat()

def processing_done(status_obj, label="Done"):
    status_obj.update(label=f"✅ {label}", state="complete")
    add_msg("system", f"✅ {label}")
    st_placeholder.empty(); render_chat()

# =============================
# Load data (and DS first overview)
# =============================
if zip_file is not None and st.session_state.tables is None:
    try:
        st.session_state.tables = load_zip_tables(zip_file)
        add_msg("system", f"Loaded {len(st.session_state.tables)} tables from ZIP.")
        summaries = summarize_tables(st.session_state.tables)
        st.session_state.ds_first_overview = summaries
        add_msg("ds", "Initial data overview completed.", artifacts={"tables": summaries})
    except Exception as e:
        st.error(f"Failed to read ZIP: {e}")

# =============================
# Chat UI
# =============================
st.subheader("Chat")
st_placeholder = st.empty()
render_chat()

# consume preset prompt
preset = st.session_state.pending_prompt or ""
st.session_state.pending_prompt = ""

user_prompt = st.chat_input(
    placeholder=preset or "You're the CEO. Ask in plain language (e.g., 'What data do we have?' or 'How can we improve profit?')",
    key="ceo_chat_input"
)

# =============================
# Modeling
# =============================
def train_model(df: pd.DataFrame, task: str, target: str, features: List[str], family: str) -> Dict[str, Any]:
    report: Dict[str, Any] = {}
    if target not in df.columns:
        return {"error": f"Target '{target}' not found."}
    X = df[features] if features else df.drop(columns=[target], errors="ignore")
    y = df[target]

    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), num_cols),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                              ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
        ],
        remainder="drop",
    )

    if task == "classification":
        if not pd.api.types.is_numeric_dtype(y):
            y = pd.Categorical(y).codes
        if family in ("random_forest", "rf"):
            model = RandomForestClassifier(n_estimators=300, random_state=42)
            fam = "random_forest"
        else:
            model = LogisticRegression(max_iter=1000)
            fam = "logistic_regression"
    else:
        if family in ("random_forest_regressor", "random_forest", "rf"):
            model = RandomForestRegressor(n_estimators=300, random_state=42)
            fam = "random_forest_regressor"
        else:
            model = LinearRegression()
            fam = "linear_regression"

    pipe = Pipeline([("pre", pre), ("model", model)])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y if task=="classification" else None
    )
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    report.update({"task": task, "target": target, "features": features, "model_family": fam})

    if task == "classification":
        try:
            proba = pipe.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, proba)
        except Exception:
            proba, auc = None, None
        acc = accuracy_score(y_test, (y_pred > 0.5) if proba is not None else y_pred)
        report.update({"accuracy": float(acc), "roc_auc": (float(auc) if auc is not None else None)})
    else:
        report.update({
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "rmse": float(mean_squared_error(y_test, y_pred, squared=False)),
            "r2": float(r2_score(y_test, y_pred))
        })
    # (best-effort) top features
    try:
        mdl = pipe.named_steps["model"]
        cat_names = []
        try:
            ohe = pipe.named_steps["pre"].named_transformers_["cat"].named_steps["ohe"]
            cat_names = list(ohe.get_feature_names_out(cat_cols))
        except Exception:
            pass
        all_names = num_cols + cat_names
        if hasattr(mdl, "feature_importances_"):
            imps = mdl.feature_importances_
            idx = np.argsort(imps)[::-1][:20]
            report["top_features"] = [{"name": all_names[i], "importance": float(imps[i])} for i in idx if i < len(all_names)]
        elif hasattr(mdl, "coef_"):
            coef = np.ravel(mdl.coef_)
            idx = np.argsort(np.abs(coef))[::-1][:20]
            report["top_features"] = [{"name": all_names[i], "coef": float(coef[i])} for i in idx if i < len(all_names)]
    except Exception:
        pass

    return report

# =============================
# Core turn handlers
# =============================
def run_am_plan(ceo_prompt: str) -> dict:
    payload = {"ceo_question": ceo_prompt,
               "tables": {k: list(v.columns) for k, v in (st.session_state.tables or {}).items()}}
    am_json = llm_json(SYSTEM_AM, json.dumps(payload))
    st.session_state.last_am_json = am_json
    add_msg("am", am_json.get("am_brief",""), artifacts=am_json)
    st_placeholder.empty(); render_chat()
    return am_json

def run_ds_step(am_json: dict) -> dict:
    ds_payload = {
        "am_plan": am_json.get("plan_for_ds",""),
        "tables": {k: list(v.columns) for k, v in (st.session_state.tables or {}).items()},
        "first_run_summary": st.session_state.ds_first_overview or {}
    }
    ds_json = llm_json(SYSTEM_DS, json.dumps(ds_payload))
    st.session_state.last_ds_json = ds_json
    add_msg("ds", ds_json.get("ds_summary",""), artifacts=ds_json)
    st_placeholder.empty(); render_chat()
    return ds_json

def execute_ds_action(ds_json: dict):
    action = (ds_json.get("action") or "").lower()

    if action == "overview":
        # Show 5 rows for each table directly (inline)
        st.markdown("### 📊 Table Previews")
        for name, df in st.session_state.tables.items():
            st.markdown(f"**{name}** — first 5 rows")
            st.dataframe(df.head(5), use_container_width=True)
        add_msg("ds", "Provided previews for each table (first 5 rows).")
        st_placeholder.empty(); render_chat()

    elif action == "sql":
        sql = (ds_json.get("duckdb_sql") or "").strip()
        if not sql:
            add_msg("ds","No SQL provided."); st_placeholder.empty(); render_chat(); return
        try:
            out = run_duckdb_sql(st.session_state.tables, sql)
            add_msg("ds", "SQL results ready.", artifacts={"sql": sql})
            st_placeholder.empty(); render_chat()
            st.markdown("### 🧮 SQL Results (first 25 rows)")
            st.code(sql, language="sql")
            st.dataframe(out.head(25), use_container_width=True)
        except Exception as e:
            st.error(f"SQL failed: {e}")
            add_msg("ds", f"SQL error: {e}", artifacts={"sql": sql})
            st_placeholder.empty(); render_chat()

    elif action == "calc":
        add_msg("ds", f"Calculation: {ds_json.get('calc_description','(no description)')}")
        st_placeholder.empty(); render_chat()

    elif action in ("feature_engineering","modeling"):
        base = None
        sql = (ds_json.get("duckdb_sql") or "").strip()
        if sql:
            try:
                base = run_duckdb_sql(st.session_state.tables, sql)
            except Exception as e:
                st.error(f"Feature SQL failed: {e}")
                add_msg("ds", f"Feature SQL error: {e}", artifacts={"sql": sql})
                st_placeholder.empty(); render_chat()
        if base is None:
            # pick first table that has target, else first table
            plan = ds_json.get("model_plan") or {}
            tgt = plan.get("target")
            for name, df in st.session_state.tables.items():
                if tgt and tgt in df.columns:
                    base = df.copy(); break
            if base is None: base = next(iter(st.session_state.tables.values())).copy()

        if action == "feature_engineering":
            add_msg("ds", "Feature engineering base ready.")
            st_placeholder.empty(); render_chat()
            st.markdown("### 🧱 Feature Engineering Base (first 20 rows)")
            st.dataframe(base.head(20), use_container_width=True)
        else:
            plan = ds_json.get("model_plan") or {}
            task   = (plan.get("task") or "classification").lower()
            target = plan.get("target")
            feats  = plan.get("features") or []
            family = (plan.get("model_family") or "logistic_regression").lower()
            report = train_model(base, task, target, feats, family)
            add_msg("ds", "Model trained.", artifacts={"model_report": report})
            st_placeholder.empty(); render_chat()
            st.markdown("### 🤖 Model Report")
            st.json(report)

    else:
        add_msg("ds", f"Action '{action}' not recognized.", artifacts=ds_json)
        st_placeholder.empty(); render_chat()

def classify_intent(history: List[dict], latest_user: str) -> str:
    # Build compact history for the classifier
    h = [{"role": m["role"], "content": m["content"][:1000]} for m in history[-10:]]
    payload = {"history": h, "latest_user": latest_user}
    out = llm_json(SYSTEM_INTENT, json.dumps(payload))
    return (out or {}).get("intent","new_request")

def revise_with_feedback(feedback_text: str):
    s = processing_start("Revising based on CEO feedback…")
    bundle = {
        "ceo_prompt": st.session_state.last_user_prompt,
        "feedback_note": feedback_text,
        "last_am_json": st.session_state.last_am_json,
        "last_ds_json": st.session_state.last_ds_json
    }
    rev = llm_json(SYSTEM_REVIEW, json.dumps(bundle))
    directive = rev.get("revision_directive","")
    add_msg("system", "Coordinator directive produced.", artifacts=rev)
    st_placeholder.empty(); render_chat()

    revised_prompt = f"{st.session_state.last_user_prompt}\n\n(Director to AM/DS:) {directive}"
    processing_update(s, "AM planning with directive…")
    am_json = run_am_plan(revised_prompt)
    if am_json.get("need_more_info"):
        qs = am_json.get("clarifying_questions", [])
        if qs:
            add_msg("am", "Before proceeding, could you clarify:", artifacts={"questions": qs})
            processing_done(s, "Waiting for CEO clarification")
            return
    processing_update(s, "DS executing revised plan…")
    ds_json = run_ds_step(am_json)
    execute_ds_action(ds_json)
    processing_done(s, "Revision complete")

def run_turn_ceo(text: str):
    st.session_state.last_user_prompt = text
    if st.session_state.tables is None:
        add_msg("system", "Please upload a ZIP of CSVs first.")
        st_placeholder.empty(); render_chat()
        return

    s = processing_start("AM is designing an analytics plan…")
    am_json = run_am_plan(text)
    processing_update(s, "AM plan ready → DS executing…")

    if am_json.get("need_more_info"):
        qs = am_json.get("clarifying_questions", [])
        if qs:
            add_msg("am", "Before proceeding, could you clarify:", artifacts={"questions": qs})
            processing_done(s, "Waiting for CEO clarification")
            return
        processing_done(s, "Need more information")
        return

    ds_json = run_ds_step(am_json)
    processing_update(s, "DS decided action; executing…")
    execute_ds_action(ds_json)
    processing_done(s, "Turn complete")

# =============================
# Handle CEO message
# =============================
if user_prompt:
    add_msg("user", user_prompt)
    st_placeholder.empty(); render_chat()
    intent = classify_intent(st.session_state.chat, user_prompt)
    if intent == "feedback":
        revise_with_feedback(user_prompt)
    elif intent == "answers_to_clarifying":
        combined = f"{st.session_state.last_user_prompt}\n\n(CEO clarification:) {user_prompt}"
        run_turn_ceo(combined)
    else:
        run_turn_ceo(user_prompt)

# =============================
# Optional schemas display
# =============================
if show_schema and st.session_state.tables:
    with st.expander("📚 Tables & Schemas", expanded=False):
        for tname, df in st.session_state.tables.items():
            st.markdown(f"**{tname}** — rows: {len(df)}, cols: {len(df.columns)}")
            st.code("\n".join([f"- {c} ({str(df[c].dtype)})" for c in df.columns]), language="markdown")
