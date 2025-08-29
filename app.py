import os
import io
import json
import time
import zipfile
from typing import Dict, Any, List, Tuple

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
# System prompts (AM & DS)
# =============================
SYSTEM_AM = r"""
You are the **Analytics Manager (AM)**. Your audience is a CEO with limited analytics background.
Your responsibilities:
- Face the CEO: restate their question in business terms focused on PROFIT improvement (revenue ↑, cost ↓, margin ↑).
- Design an analytics plan for the **Data Scientist (DS)** using the available tables/columns only (no invented fields).
- Decide whether the first step should be **data overview**, **simple SQL/aggregation**, **feature engineering**, **modeling**, or **what-if calculation**.
- Evaluate DS results and give crisp feedback for next step.

Return ONLY valid JSON with keys:
{
  "am_brief": "1-3 sentences in plain language for the CEO",
  "plan_for_ds": "step-by-step DS instructions grounded in available tables/columns",
  "goal": "what to measure or improve (profit proxy, margin, churn, etc.)",
  "next_action_type": "overview|sql|calc|feature_engineering|modeling|what_if",
  "notes_to_ceo": "1-2 helpful notes in plain language"
}
"""

SYSTEM_DS = r"""
You are the **Data Scientist (DS)** working with the AM.
Inputs: AM's plan, available tables/columns, and a multi-CSV dataset loaded from a ZIP (each CSV is a table).

Your responsibilities on FIRST RUN:
- Examine ALL tables: report rows, columns, missingness, and sample rows.
- Propose light cleaning (type casting, trimming, obvious NA fixes) and apply those steps.

For each step from the AM:
- Decide whether to run simple SQL/aggregation, feature engineering, modeling, or a simple calculation.
- When doing SQL, generate DuckDB SQL referencing tables exactly by their names.
- When doing modeling, specify task ("classification" or "regression"), target, features, and model family
  ("logistic_regression" | "random_forest" | "linear_regression" | "random_forest_regressor"). Modeling is executed in Python here.

Return ONLY valid JSON with keys:
{
  "ds_summary": "short summary of what you did/are doing",
  "action": "overview|sql|calc|feature_engineering|modeling",
  "duckdb_sql": "SQL if action uses SQL, else ''",
  "model_plan": {
    "task": "classification|regression|null",
    "target": "existing column name or null",
    "features": ["col1","col2",...],
    "model_family": "logistic_regression|random_forest|linear_regression|random_forest_regressor|null"
  },
  "calc_description": "if action=calc, describe the calculation in words",
  "assumptions": "any assumptions you made"
}
"""

SYSTEM_REVIEW = r"""
You are a **Coordinator**. Given CEO feedback and the prior AM+DS responses, produce a concise revision directive.

Return ONLY valid JSON:
{
  "revision_directive": "single concise paragraph with concrete changes for AM and DS"
}
"""

# =============================
# Streamlit page
# =============================
st.set_page_config(page_title="CEO ↔ AM ↔ DS — Profit Assistant", layout="wide")
st.title("🏢 CEO ↔ AM ↔ DS — Profit Improvement Assistant")

with st.sidebar:
    st.header("⚙️ Setup")
    zip_file = st.file_uploader("Upload ZIP of CSVs", type=["zip"], accept_multiple_files=False)
    model     = st.text_input("OpenAI model", value=DEFAULT_MODEL)
    api_key   = st.text_input("OPENAI_API_KEY", value=OPENAI_API_KEY, type="password")
    show_schema = st.checkbox("Show table schemas", value=True)

    st.markdown("---")
    st.subheader("🧪 Utilities")
    if st.button("🔎 LLM health check"):
        if not _OPENAI_AVAILABLE:
            st.error("OpenAI SDK not installed. Add `openai` to requirements.txt.")
        else:
            try:
                client = OpenAI(api_key=api_key)
                ping = client.chat.completions.create(
                    model=model,
                    response_format={"type": "json_object"},
                    messages=[{"role":"user","content":"{\"ok\": true}"}],
                    temperature=0.0,
                )
                st.success("OpenAI reachable.")
                st.code(ping.choices[0].message.content, language="json")
            except Exception as e:
                st.error(f"OpenAI not reachable: {e}")

    st.markdown("---")
    st.subheader("Preset questions")
    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = ""
    colp1, colp2 = st.columns(2)
    if colp1.button("What data do we have?"):
        st.session_state.pending_prompt = "What data do we have? Please walk me through each table briefly with previews."
    if colp2.button("Where could we improve profit quickly?"):
        st.session_state.pending_prompt = "Where could we improve profit quickly? Start with a high-level plan."

    st.markdown("---")
    st.subheader("🔁 Reproduction")
    if "last_feedback" not in st.session_state:
        st.session_state.last_feedback = ""
    if "last_user_prompt" not in st.session_state:
        st.session_state.last_user_prompt = ""
    if "last_am_json" not in st.session_state:
        st.session_state.last_am_json = {}
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

def llm_json(system_prompt: str, user_payload: str) -> dict:
    """Strict JSON mode first; fenced block fallback."""
    client = ensure_openai()
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
            st.warning("Model did not return JSON; showing raw text.")
            with st.expander("Raw model response"):
                st.code(raw, language="markdown")
            return {"_raw": raw, "_parse_error": True}
        try:
            return json.loads(m.group(1).strip())
        except Exception as e_parse:
            st.error(f"JSON parsing failed: {e_parse}")
            with st.expander("Raw JSON text (first 400 chars)"):
                st.code(m.group(1).strip()[:400], language="json")
            return {"_raw": raw, "_parse_error": True}

def load_zip_tables(file) -> Dict[str, pd.DataFrame]:
    """Read a ZIP of CSVs -> dict of {table_name: DataFrame}"""
    tables = {}
    with zipfile.ZipFile(file) as z:
        for name in z.namelist():
            if not name.lower().endswith(".csv"):
                continue
            with z.open(name) as f:
                try:
                    df = pd.read_csv(io.BytesIO(f.read()))
                except Exception:
                    df = pd.read_csv(io.BytesIO(f.read()), encoding="latin1")
            key = os.path.splitext(os.path.basename(name))[0]
            # de-duplicate table names if necessary
            i, base = 1, key
            while key in tables:
                key = f"{base}_{i}"
                i += 1
            tables[key] = df
    return tables

def infer_schema(df: pd.DataFrame, sample_rows: int = 5) -> str:
    lines = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        sample_vals = df[col].dropna().unique()[:sample_rows]
        sample_list = ", ".join([repr(v)[:60] for v in sample_vals])
        lines.append(f"- {col} ({dtype})  samples: {sample_list}")
    return "\n".join(lines)

def duckdb_connect_with_tables(tables: Dict[str, pd.DataFrame]):
    con = duckdb.connect(database=":memory:")
    for name, df in tables.items():
        con.register(name, df)
    return con

def summarize_tables(tables: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Basic summary of each table."""
    out = {}
    for name, df in tables.items():
        miss = (df.isna().sum() / max(len(df),1)).round(3).to_dict()
        out[name] = {
            "rows": int(len(df)),
            "cols": int(len(df.columns)),
            "missing_fraction": miss,
            "sample": df.head(5).to_dict(orient="records"),
            "schema": infer_schema(df)
        }
    return out

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
# Initial data loading & first-run DS overview
# =============================
if "tables" not in st.session_state:
    st.session_state.tables = None
if "ds_first_overview" not in st.session_state:
    st.session_state.ds_first_overview = None

if zip_file is not None and st.session_state.tables is None:
    try:
        st.session_state.tables = load_zip_tables(zip_file)
        add_msg("system", f"Loaded {len(st.session_state.tables)} tables from ZIP.")
        # DS first-run overview & light cleaning proposal
        summaries = summarize_tables(st.session_state.tables)
        st.session_state.ds_first_overview = summaries
        add_msg("ds", "Initial data overview completed.", artifacts={"tables": summaries})
    except Exception as e:
        st.error(f"Failed to read ZIP: {e}")

# =============================
# Chat UI
# =============================
st.subheader("Chat")
chat_placeholder = st.empty()
render_chat(chat_placeholder)

# Prompt (use preset if set)
default_prompt = st.session_state.pending_prompt or ""
if st.session_state.pending_prompt:
    st.session_state.pending_prompt = ""  # consume

user_prompt = st.chat_input(
    placeholder=default_prompt or "You're the CEO. Ask in plain language (e.g., 'What data do we have?' or 'How to improve profit?')",
    key="ceo_chat_input"  # stable key; no need for time()
)

# =============================
# Modeling utilities
# =============================
def run_duckdb_sql(tables: Dict[str, pd.DataFrame], sql: str) -> pd.DataFrame:
    con = duckdb_connect_with_tables(tables)
    return con.execute(sql).df()

def train_model(df: pd.DataFrame, task: str, target: str, features: List[str], family: str) -> Dict[str, Any]:
    report: Dict[str, Any] = {}
    if target not in df.columns:
        return {"error": f"Target '{target}' not found."}
    X = df[features] if features else df.drop(columns=[target], errors="ignore")
    y = df[target]

    # guess dtypes
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]

    # basic preprocessing
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[("imp", SimpleImputer(strategy="median"))]), num_cols),
            ("cat", Pipeline(steps=[("imp", SimpleImputer(strategy="most_frequent")),
                                    ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
        ],
        remainder="drop",
    )

    # choose model
    if task == "classification":
        if str(y.dtype) != "int64" and str(y.dtype) != "float64":
            y = pd.Categorical(y).codes
        if family in ("random_forest", "rf"):
            mdl = RandomForestClassifier(n_estimators=300, random_state=42)
            fam = "random_forest"
        else:
            mdl = LogisticRegression(max_iter=1000)
            fam = "logistic_regression"
    else:
        if family in ("random_forest_regressor", "random_forest", "rf"):
            mdl = RandomForestRegressor(n_estimators=300, random_state=42)
            fam = "random_forest_regressor"
        else:
            mdl = LinearRegression()
            fam = "linear_regression"

    pipe = Pipeline([("pre", pre), ("model", mdl)])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42,
                                                        stratify=y if task=="classification" else None)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    report.update({"task": task, "target": target, "features": features, "model_family": fam})

    if task == "classification":
        try:
            proba = pipe.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, proba)
        except Exception:
            proba = None
            auc = None
        acc = accuracy_score(y_test, (y_pred > 0.5) if proba is not None else y_pred)
        report.update({"accuracy": float(acc), "roc_auc": (float(auc) if auc is not None else None)})
    else:
        report.update({
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "rmse": float(mean_squared_error(y_test, y_pred, squared=False)),
            "r2": float(r2_score(y_test, y_pred))
        })

    # Top features (best-effort)
    try:
        mdl = pipe.named_steps["model"]
        cat_names = []
        try:
            ohe = pipe.named_steps["pre"].named_transformers_["cat"].named_steps["oh"]
            cat_names = list(ohe.get_feature_names_out(cat_cols))
        except Exception:
            pass
        all_names = num_cols + cat_names
        if hasattr(mdl, "feature_importances_"):
            imps = mdl.feature_importances_
            idx = np.argsort(imps)[::-1][:20]
            report["top_features"] = [{"name": all_names[i], "importance": float(imps[i])} for i in idx[:len(all_names)]]
        elif hasattr(mdl, "coef_"):
            coef = np.ravel(mdl.coef_)
            idx = np.argsort(np.abs(coef))[::-1][:20]
            report["top_features"] = [{"name": all_names[i], "coef": float(coef[i])} for i in idx[:len(all_names)]]
    except Exception:
        pass

    return report

# =============================
# Main turn: CEO → AM → DS → (execute)
# =============================
def run_turn(ceo_prompt: str):
    if st.session_state.tables is None:
        st.warning("Please upload a ZIP that contains CSV files first.")
        st.stop()

    # Display schema if requested
    if show_schema:
        with st.expander("📚 Tables & Schemas", expanded=False):
            for tname, df in st.session_state.tables.items():
                st.markdown(f"**{tname}**  — rows: {len(df)}, cols: {len(df.columns)}")
                st.code(infer_schema(df), language="markdown")

    # --- AM plans for DS ---
    with st.status("🧭 AM is designing an analytics plan…", state="running"):
        am_payload = {
            "ceo_question": ceo_prompt,
            "tables": {k: list(v.columns) for k, v in st.session_state.tables.items()}
        }
        am_json = llm_json(SYSTEM_AM, json.dumps(am_payload))
        st.session_state.last_am_json = am_json
        add_msg("am", am_json.get("am_brief", "AM brief."), artifacts=am_json)

    chat_placeholder.empty(); render_chat(chat_placeholder)

    # --- DS executes per plan ---
    with st.status("🧪 DS is executing the plan…", state="running"):
        ds_payload = {
            "am_plan": am_json.get("plan_for_ds", ""),
            "tables": {k: list(v.columns) for k, v in st.session_state.tables.items()},
            "first_run_summary": st.session_state.ds_first_overview if st.session_state.ds_first_overview else {}
        }
        ds_json = llm_json(SYSTEM_DS, json.dumps(ds_payload))
        st.session_state.last_ds_json = ds_json
        add_msg("ds", ds_json.get("ds_summary", "DS summary."), artifacts=ds_json)

    chat_placeholder.empty(); render_chat(chat_placeholder)

    # --- Execute action chosen by DS ---
    action = (ds_json.get("action") or "").lower()
    if action == "overview":
        # Show previews for each table
        with st.expander("📊 Table Previews", expanded=True):
            for name, df in st.session_state.tables.items():
                st.markdown(f"**{name}** (first 10 rows)")
                st.dataframe(df.head(10), use_container_width=True)
        add_msg("ds", "Provided previews for each table.")

    elif action == "sql":
        sql = (ds_json.get("duckdb_sql") or "").strip()
        if not sql:
            add_msg("ds", "No SQL provided.", artifacts={})
        else:
            try:
                out = run_duckdb_sql(st.session_state.tables, sql)
                add_msg("ds", "SQL results ready.", artifacts={"sql": sql, "preview": out.head(25).to_dict(orient="records")})
                with st.expander("SQL Results", expanded=True):
                    st.code(sql, language="sql")
                    st.dataframe(out, use_container_width=True)
            except Exception as e:
                st.error(f"SQL failed: {e}")
                add_msg("ds", f"SQL error: {e}", artifacts={"sql": sql})

    elif action == "calc":
        add_msg("ds", f"Calculation: {ds_json.get('calc_description','(no description)')}", artifacts={})

    elif action in ("feature_engineering", "modeling"):
        plan = ds_json.get("model_plan") or {}
        task   = (plan.get("task") or "").lower()
        target = plan.get("target")
        feats  = plan.get("features") or []
        family = (plan.get("model_family") or "").lower()

        # if DS provided SQL to create modeling base, run it
        base = None
        sql = (ds_json.get("duckdb_sql") or "").strip()
        if sql:
            try:
                base = run_duckdb_sql(st.session_state.tables, sql)
            except Exception as e:
                st.error(f"Feature SQL failed: {e}")
                add_msg("ds", f"Feature SQL error: {e}", artifacts={"sql": sql})
        if base is None:
            # try simple merge if only one table; else just pick any table with target
            base = None
            for name, df in st.session_state.tables.items():
                if target and target in df.columns:
                    base = df.copy()
                    break
            if base is None:
                # fallback: single table
                base = next(iter(st.session_state.tables.values())).copy()

        if action == "feature_engineering":
            # In this demo, we just show the base after SQL/selection
            add_msg("ds", "Feature engineering base ready.", artifacts={"rows": len(base), "preview": base.head(20).to_dict(orient="records")})
            with st.expander("Feature Base Preview", expanded=True):
                st.dataframe(base.head(20), use_container_width=True)

        else:  # modeling
            report = train_model(base, task or "classification", target, feats, family or "logistic_regression")
            add_msg("ds", "Model trained.", artifacts={"model_report": report})
            with st.expander("Model Report", expanded=True):
                st.json(report)

    else:
        add_msg("ds", f"Action '{action}' not recognized.", artifacts=ds_json)

    chat_placeholder.empty(); render_chat(chat_placeholder)

# =============================
# Handle CEO input
# =============================
if user_prompt:
    add_msg("user", user_prompt)
    st.session_state.last_user_prompt = user_prompt
    try:
        run_turn(user_prompt)
    except Exception as e:
        add_msg("system", f"Error: {type(e).__name__}: {e}")
        st.error(str(e))
    chat_placeholder.empty(); render_chat(chat_placeholder)

# =============================
# Feedback & Reproduce
# =============================
st.divider()
st.subheader("🔁 Reproduce with CEO feedback")

col1, col2 = st.columns([1,3])
with col1:
    up = st.button("👍 Helpful", key=f"up_{time.time()}")
    down = st.button("👎 Not helpful", key=f"down_{time.time()}")
with col2:
    fb_note = st.text_input("Tell AM & DS what to change next:", key=f"fb_{time.time()}")

if up or down or fb_note:
    st.session_state.last_feedback = (fb_note or "").strip()
    add_msg("user", f"Feedback — up={bool(up)}, down={bool(down)}, note={st.session_state.last_feedback or '(none)'}")
    chat_placeholder.empty(); render_chat(chat_placeholder)

if st.button("Re-run AM & DS using my feedback", type="primary", use_container_width=True):
    try:
        if not st.session_state.last_user_prompt:
            st.warning("No previous CEO question to reproduce. Ask first.")
            st.stop()

        payload = {
            "ceo_prompt": st.session_state.last_user_prompt,
            "feedback_note": st.session_state.last_feedback,
            "last_am_json": st.session_state.last_am_json,
            "last_ds_json": st.session_state.last_ds_json
        }
        rev = llm_json(SYSTEM_REVIEW, json.dumps(payload))
        directive = rev.get("revision_directive", "")
        add_msg("system", "Coordinator directive produced.", artifacts=rev)
        chat_placeholder.empty(); render_chat(chat_placeholder)

        revised_prompt = f"{st.session_state.last_user_prompt}\n\n(Director to AM/DS:) {directive}"
        add_msg("user", "Reproducing with feedback…")
        chat_placeholder.empty(); render_chat(chat_placeholder)
        run_turn(revised_prompt)

    except Exception as e:
        add_msg("system", f"Error: {type(e).__name__}: {e}")
        st.error(str(e))
        chat_placeholder.empty(); render_chat(chat_placeholder)

# =============================
# Footer Tips
# =============================
with st.expander("🛠️ Tips", expanded=False):
    st.write("- Upload a ZIP of CSVs. Each CSV becomes a table (registered in DuckDB by filename).")
    st.write("- Ask “What data do we have?” to see per-table previews. AM plans; DS executes.")
    st.write("- DS can choose overview/SQL/calc/feature engineering/modeling. Modeling runs in Python (scikit-learn).")
    st.write("- Use the feedback box to steer the next iteration toward profit-oriented insights.")

