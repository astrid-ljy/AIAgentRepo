import os
import json
import time
from typing import Dict, Any, Tuple, List

import pandas as pd
import duckdb
import streamlit as st

# =============================
# Cloud-friendly config
# =============================
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
DEFAULT_MODEL  = st.secrets.get("OPENAI_MODEL",  os.getenv("OPENAI_MODEL",  "gpt-4o-mini"))

try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except Exception:
    _OPENAI_AVAILABLE = False

# ML
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import numpy as np

# =============================
# Agent system prompts
# =============================
SYSTEM_BA = r"""
You are the Business Analyst (BA) Agent focused on FINANCIAL PERFORMANCE.
Your job:
- Clarify business intent and frame the analysis in financial terms (revenue, margin, growth, retention, CAC/LTV, ARPU, etc.).
- Propose a concrete analysis/KPI plan grounded in the actual CSV schema (do NOT invent columns).
- If a concept is missing (e.g., Region), suggest the closest available column(s).

Return ONLY a valid JSON object with the specified keys (no markdown, no code fences, no extra text).

Required JSON keys:
{
  "reasoning_summary": "brief high-level reasoning",
  "ba_spec": "plain-English plan: metrics, segments, time grain, filters, any assumptions",
  "sql_feature_prep": "optional DuckDB SQL to prep data for DS (or '' if not needed)"
}
"""

SYSTEM_DS = r"""
You are the Data Scientist (DS) Agent focused on MODELING.
Your job:
- Read the BA spec + CSV schema. Decide the modeling approach (classification/regression/time-series).
- Propose: task ("classification" or "regression"), target column (existing), candidate features (existing column names),
  model_family ("logistic_regression" | "random_forest" | "linear_regression" | "random_forest_regressor"),
  validation plan, and success metrics.
- If useful, also return SQL to build features/labels from the CSV (DuckDB; table name `data` only).
- Keep it executable and schema-safe. If something is impossible, explain briefly and propose a nearest alternative.

Return ONLY a valid JSON object with the keys below (no markdown, no fences, no extra text):
{
  "reasoning_summary": "brief high-level reasoning",
  "task": "classification|regression",
  "target": "existing column name",
  "features": ["col1","col2",...],
  "model_family": "logistic_regression|random_forest|linear_regression|random_forest_regressor",
  "validation": "brief plan",
  "metrics": "what to report",
  "sql_for_model": "optional DuckDB SQL to materialize features/labels; '' if not required",
  "quick_tip": "1-liner suggestion the user can try next"
}
"""

SYSTEM_REVIEW = r"""
You are the Reviewer/Coordinator Agent.
Given the USER FEEDBACK and prior BA/DS responses, produce a concise revision directive for both agents.

Return ONLY a valid JSON object (no markdown, no fences, no extra text):
{
  "revision_directive": "single concise paragraph with concrete changes for BA and DS"
}
"""

# =============================
# Streamlit page
# =============================
st.set_page_config(page_title="BA↔DS Interactive Chat", layout="wide")
st.title("💬 Two-Agent Interactive Chat — BA (Finance) & DS (Modeling)")

with st.sidebar:
    st.header("⚙️ Setup")
    uploaded = st.file_uploader("Upload CSV", type=["csv"], accept_multiple_files=False)
    model = st.text_input("OpenAI model", value=DEFAULT_MODEL)
    api_key = st.text_input("OPENAI_API_KEY", value=OPENAI_API_KEY, type="password")
    show_schema = st.checkbox("Show inferred schema", value=True)

    st.markdown("---")
    st.subheader("🧪 Utilities")

    # LLM Health Check
    if st.button("🔎 LLM health check"):
        if not _OPENAI_AVAILABLE:
            st.error("OpenAI SDK not installed. Add `openai` to requirements.txt.")
        else:
            try:
                client = OpenAI(api_key=api_key)
                ping = client.chat.completions.create(
                    model=model,
                    messages=[{"role":"user","content":"{\"ok\": true}"}],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                )
                st.success("OpenAI reachable.")
                st.code(ping.choices[0].message.content, language="json")
            except Exception as e:
                st.error(f"OpenAI not reachable: {e}")

    # Dry Run (no LLM)
    if st.button("🔧 Dry run (no LLM)"):
        if uploaded is None:
            st.warning("Upload a CSV first.")
        else:
            try:
                df_dry = pd.read_csv(uploaded)
                st.success(f"UI OK — CSV read ({len(df_dry)} rows, {len(df_dry.columns)} cols). Showing head:")
                st.write(df_dry.head(5))
            except Exception as e:
                st.error(f"CSV read failed: {e}")

    st.markdown("---")
    st.subheader("🔁 Reproduction")
    if "last_feedback" not in st.session_state:
        st.session_state.last_feedback = ""
    if "last_user_prompt" not in st.session_state:
        st.session_state.last_user_prompt = ""
    if "last_ba_json" not in st.session_state:
        st.session_state.last_ba_json = {}
    if "last_ds_json" not in st.session_state:
        st.session_state.last_ds_json = {}

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

def llm_json(system_prompt: str, user_payload: str) -> dict:
    """
    Strict JSON mode first; fall back to fenced-block extraction if needed.
    """
    client = ensure_openai()
    # 1) Try strict JSON mode
    try:
        resp = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt + "\n\nReturn ONLY a valid JSON object. No markdown, no code fences, no extra text."},
                {"role": "user", "content": user_payload},
            ],
            temperature=0.0,
        )
        txt = (resp.choices[0].message.content or "").strip()
        return json.loads(txt)
    except Exception:
        # 2) Fallback: ask for fenced json
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt + "\n\nReturn a single JSON object in a ```json fenced block. No extra text."},
                    {"role": "user", "content": user_payload},
                ],
                temperature=0.0,
            )
            raw = (resp.choices[0].message.content or "").strip()
        except Exception as e_llm:
            st.error(f"OpenAI call failed: {type(e_llm).__name__}: {e_llm}")
            return {"_error": str(e_llm)}

        import re
        m = re.search(r"```json\s*(\{.*?\})\s*```", raw, flags=re.DOTALL | re.IGNORECASE)
        if not m:
            st.warning("Model did not return JSON. Showing raw text below.")
            with st.expander("Raw model response", expanded=False):
                st.code(raw, language="markdown")
            return {"_raw": raw, "_parse_error": True}

        jtxt = m.group(1).strip()
        try:
            return json.loads(jtxt)
        except Exception as e_parse:
            st.error(f"JSON parsing failed: {e_parse}")
            with st.expander("Raw JSON text (first 400 chars)"):
                st.code(jtxt[:400], language="json")
            return {"_raw": raw, "_parse_error": True}

def run_duckdb(df: pd.DataFrame, sql: str):
    con = duckdb.connect(database=":memory:")
    con.register("data", df)
    return con.execute(sql).df()

def add_msg(role, content, artifacts=None):
    if "chat" not in st.session_state:
        st.session_state.chat = []
    st.session_state.chat.append({"role": role, "content": content, "artifacts": artifacts or {}})

def render_chat(target_placeholder=None):
    if "chat" not in st.session_state:
        st.session_state.chat = []
    container = target_placeholder or st
    for m in st.session_state.chat:
        with container.chat_message(m["role"]):
            container.write(m["content"])
            if m["artifacts"]:
                with container.expander("Artifacts", expanded=False):
                    for k, v in m["artifacts"].items():
                        container.write(f"**{k}**")
                        if isinstance(v, str) and v.strip().upper().startswith("SELECT"):
                            container.code(v, language="sql")
                        elif isinstance(v, (dict, list)):
                            container.json(v)
                        else:
                            container.write(v)

# =============================
# Chat area (immediate re-render support)
# =============================
st.subheader("Chat")
chat_placeholder = st.empty()
render_chat(chat_placeholder)  # initial render

user_prompt = st.chat_input("Ask a business question (BA focuses on finance; DS models with Python)…")

# =============================
# Modeling functions
# =============================
def train_model_from_plan(df: pd.DataFrame, ds_plan: Dict[str, Any]) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Train a model per ds_plan: {task, target, features, model_family}
    Returns metrics/artifacts and preview of feature table.
    """
    task = (ds_plan.get("task") or "").lower()
    target = ds_plan.get("target")
    features = ds_plan.get("features") or []
    model_family = (ds_plan.get("model_family") or "").lower()

    # Validate columns
    cols = set(df.columns)
    if not target or target not in cols:
        return {"error": f"Target '{target}' not found in columns."}, pd.DataFrame()
    valid_feats = [c for c in features if c in cols and c != target]
    if not valid_feats:
        # fallback: use all except target
        valid_feats = [c for c in df.columns if c != target]

    data = df[valid_feats + [target]].copy()

    # Basic dtype handling: classify numeric vs categorical
    numeric_cols = [c for c in valid_feats if pd.api.types.is_numeric_dtype(data[c])]
    categorical_cols = [c for c in valid_feats if c not in numeric_cols]

    # If task unspecified, try to infer: binary-ish target -> classification, else regression
    if not task:
        if data[target].nunique() <= 10 and set(map(str, data[target].dropna().unique())) <= {"0","1","True","False","Yes","No","Male","Female"}:
            task = "classification"
        else:
            task = "regression"

    # For classification, try to coerce target to binary 0/1 when common labels
    y = data[target]
    if task == "classification":
        y = y.map({"Yes":1,"No":0,"True":1,"False":0,"Male":1,"Female":0}).fillna(y)
        # If still not numeric, try label-like conversion
        if not pd.api.types.is_numeric_dtype(y):
            y = pd.Categorical(y).codes

    X = data[valid_feats]

    # Preprocess: impute + one-hot for categoricals, impute for numeric
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), numeric_cols),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                              ("ohe", OneHotEncoder(handle_unknown="ignore"))]), categorical_cols),
        ],
        remainder="drop",
    )

    # Pick model
    if task == "classification":
        if model_family in ("random_forest", "rf"):
            model = RandomForestClassifier(n_estimators=200, random_state=42)
        else:
            model_family = "logistic_regression"
            model = LogisticRegression(max_iter=1000)
    else:
        if model_family in ("random_forest_regressor", "random_forest", "rf"):
            model_family = "random_forest_regressor"
            model = RandomForestRegressor(n_estimators=300, random_state=42)
        else:
            model_family = "linear_regression"
            model = LinearRegression()

    pipe = Pipeline([("pre", pre), ("model", model)])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y if task=="classification" else None)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    report: Dict[str, Any] = {
        "task": task,
        "target": target,
        "used_features": valid_feats,
        "model_family": model_family,
    }

    if task == "classification":
        # For probabilistic models, get positive class prob if available
        try:
            y_proba = pipe.predict_proba(X_test)[:, 1]
            roc = roc_auc_score(y_test, y_proba)
        except Exception:
            y_proba = None
            roc = None
        acc = accuracy_score(y_test, (y_pred > 0.5) if y_proba is not None else y_pred)
        report.update({
            "accuracy": float(acc),
            "roc_auc": (float(roc) if roc is not None else None),
            "classification_report": classification_report(y_test, (y_pred > 0.5) if y_proba is not None else y_pred, zero_division=0)
        })
    else:
        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)
        report.update({
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2)
        })

    # Feature importance / coefficients
    try:
        mdl = pipe.named_steps["model"]
        if hasattr(mdl, "feature_importances_"):
            # Need feature names after OHE
            ohe = pipe.named_steps["pre"].named_transformers_["cat"].named_steps["ohe"]
            cat_feature_names = list(ohe.get_feature_names_out(categorical_cols))
            all_feature_names = numeric_cols + cat_feature_names
            importances = mdl.feature_importances_
            idx = np.argsort(importances)[::-1][:25]
            report["top_features"] = [
                {"name": all_feature_names[i], "importance": float(importances[i])} for i in idx
            ]
        elif hasattr(mdl, "coef_"):
            # For linear/logistic: absolute coefficients
            ohe = pipe.named_steps["pre"].named_transformers_["cat"].named_steps["ohe"]
            cat_feature_names = list(ohe.get_feature_names_out(categorical_cols))
            all_feature_names = numeric_cols + cat_feature_names
            coefs = np.ravel(mdl.coef_) if hasattr(mdl, "coef_") else np.array([])
            idx = np.argsort(np.abs(coefs))[::-1][:25]
            report["top_features"] = [
                {"name": all_feature_names[i], "coef": float(coefs[i])} for i in idx
            ]
    except Exception:
        pass

    # Return also a small preview table (joined X,y)
    preview = pd.concat([X.reset_index(drop=True), pd.Series(y, name=target).reset_index(drop=True)], axis=1).head(20)
    return report, preview

# =============================
# Main turn: BA -> DS -> (optional SQLs) -> Python modeling
# =============================
def run_turn(user_prompt: str):
    # CSV/schema gating
    if uploaded is None:
        st.warning("Please upload a CSV first.")
        st.stop()

    df = pd.read_csv(uploaded)
    if df.empty:
        st.warning("Uploaded CSV has 0 rows.")
        st.stop()

    schema_txt = infer_schema(df)
    if show_schema:
        with st.expander("📜 Inferred Schema", expanded=True):
            st.code(schema_txt or "(no columns detected)", language="markdown")

    # --- BA step ---
    with st.status("🧭 BA framing financial analysis…", state="running"):
        ba_user = f"USER QUESTION:\n{user_prompt}\n\nSCHEMA for table `data`:\n{schema_txt}"
        ba_json = llm_json(SYSTEM_BA, ba_user)
        st.session_state.last_ba_json = ba_json
        add_msg("assistant", "BA: Financial framing & spec", artifacts=ba_json)

    # Re-render chat immediately so user sees BA response now
    chat_placeholder.empty()
    render_chat(chat_placeholder)

    # Optional SQL feature prep from BA
    prep_sql = (ba_json.get("sql_feature_prep") or "").strip()
    if prep_sql:
        try:
            feat_df = run_duckdb(df, prep_sql)
            add_msg("assistant", "BA: Feature prep result (preview)", artifacts={"rows": len(feat_df), "preview": feat_df.head(20).to_dict(orient="records")})
        except Exception as e:
            st.error(f"BA prep SQL failed: {e}")
            with st.expander("Executed SQL (BA prep)"):
                st.code(prep_sql, language="sql")
            add_msg("assistant", "BA prep SQL error; please adjust columns or ask BA to revise.", artifacts={"sql": prep_sql})

        chat_placeholder.empty()
        render_chat(chat_placeholder)

    # --- DS step ---
    with st.status("🧪 DS proposing modeling plan…", state="running"):
        ds_user = (
            f"BA SPEC:\n{ba_json.get('ba_spec','')}\n\n"
            f"SCHEMA for `data`:\n{schema_txt}\n\n"
            f"(Optional) BA feature-prep SQL:\n{prep_sql or '(none)'}"
        )
        ds_json = llm_json(SYSTEM_DS, ds_user)
        st.session_state.last_ds_json = ds_json
        add_msg("assistant", "DS: Modeling plan", artifacts=ds_json)

    chat_placeholder.empty()
    render_chat(chat_placeholder)

    # Optional DS SQL to materialize features/labels
    ds_sql = (ds_json.get("sql_for_model") or "").strip()
    feature_base_df = df
    if ds_sql:
        try:
            feature_base_df = run_duckdb(df, ds_sql)
            add_msg("assistant", "DS: Feature/label table (preview)", artifacts={"rows": len(feature_base_df), "preview": feature_base_df.head(20).to_dict(orient="records")})
        except Exception as e:
            st.error(f"DS feature SQL failed: {e}")
            with st.expander("Executed SQL (DS features)"):
                st.code(ds_sql, language="sql")
            add_msg("assistant", "DS feature SQL error; please adjust columns or ask DS to revise.", artifacts={"sql": ds_sql})

        chat_placeholder.empty()
        render_chat(chat_placeholder)

    # --- Python modeling (scikit-learn) ---
    with st.status("🤖 Training model in Python…", state="running"):
        report, prev = train_model_from_plan(feature_base_df, ds_json)
        add_msg("assistant", "DS: Model results", artifacts={"report": report, "feature_table_preview": prev.to_dict(orient="records")})

    chat_placeholder.empty()
    render_chat(chat_placeholder)

# =============================
# Handle user input
# =============================
if user_prompt:
    add_msg("user", user_prompt)
    st.session_state.last_user_prompt = user_prompt
    try:
        run_turn(user_prompt)
    except Exception as e:
        add_msg("assistant", f"Error: {type(e).__name__}: {e}")
        st.error(str(e))
    # Final chat render already called inside run_turn

# =============================
# Reproduce with feedback
# =============================
st.divider()
st.subheader("🔁 Reproduce answer with your feedback")

# Feedback controls placed here (independent of responses)
col1, col2 = st.columns([1,3])
with col1:
    up = st.button("👍 Helpful", key=f"up_{time.time()}")
    down = st.button("👎 Not helpful", key=f"down_{time.time()}")
with col2:
    fb_note = st.text_input("Tell the agents what to change next:", key=f"fb_{time.time()}")

if up or down or fb_note:
    st.session_state.last_feedback = (fb_note or "").strip()
    add_msg("user", f"Feedback — up={bool(up)}, down={bool(down)}, note={st.session_state.last_feedback or '(none)'}")
    chat_placeholder.empty()
    render_chat(chat_placeholder)

if st.button("Re-run BA & DS using my feedback", type="primary", use_container_width=True):
    try:
        if not st.session_state.last_user_prompt:
            st.warning("No previous user prompt to reproduce. Ask a question first.")
            st.stop()

        df = pd.read_csv(uploaded) if uploaded is not None else None
        if df is None or df.empty:
            st.warning("Please upload a non-empty CSV.")
            st.stop()

        schema_txt = infer_schema(df)
        reviewer_payload = json.dumps({
            "user_prompt": st.session_state.last_user_prompt,
            "feedback_note": st.session_state.last_feedback,
            "ba_json": st.session_state.last_ba_json,
            "ds_json": st.session_state.last_ds_json
        }, indent=2)

        rev = llm_json(SYSTEM_REVIEW, reviewer_payload)
        directive = rev.get("revision_directive", "")
        add_msg("assistant", "Coordinator: Revision directive", artifacts=rev)
        chat_placeholder.empty()
        render_chat(chat_placeholder)

        # Re-run the turn but prepend the directive to the prompt
        revised_prompt = f"{st.session_state.last_user_prompt}\n\n(Reviewer directive to BA/DS:) {directive}"
        add_msg("user", "Reproducing with feedback…")
        chat_placeholder.empty()
        render_chat(chat_placeholder)

        run_turn(revised_prompt)

    except Exception as e:
        add_msg("assistant", f"Error: {type(e).__name__}: {e}")
        st.error(str(e))
        chat_placeholder.empty()
        render_chat(chat_placeholder)

# =============================
# Footer tips
# =============================
with st.expander("🛠️ Tips", expanded=False):
    st.write("- BA focuses on financial framing. DS builds a real model in Python (scikit-learn).")
    st.write("- Each run re-reads your CSV to stay schema-accurate. The DuckDB table is always named `data`.")
    st.write("- Use feedback to request changes (e.g., “use Tenure & MonthlyCharges as features,” “try random_forest,” “target is Churn”).")
