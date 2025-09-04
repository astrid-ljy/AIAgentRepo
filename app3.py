import os
import io
import re
import json
import zipfile
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import duckdb
import streamlit as st

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
**Action classification:** You MUST choose exactly one of these actions for DS to perform next: `overview`, `sql`, `eda`, `calc`, `feature_engineering`, or `modeling`. No other values are allowed.
**Special rule — Data Inventory:** If the CEO asks variations of “what data do we have,” set next_action_type="overview" only and instruct DS to output **first 5 rows for each table**. No EDA/FE/modeling.
Use the provided `column_hints` to resolve business terms (e.g., revenue → sales) strictly to existing columns.
Output JSON fields:
- am_brief: 1–2 sentences paraphrasing the question for a profit-oriented plan (do not repeat verbatim).
- plan_for_ds: concrete steps referencing ONLY existing tables/columns.
- goal: profit proxy to improve (revenue ↑, cost ↓, margin ↑).
- next_action_type: overview|sql|eda|calc|feature_engineering|modeling
- action_reason: short rationale of why this action is appropriate
- notes_to_ceo: 1–2 short notes
- need_more_info: true|false
- clarifying_questions: []
Return ONLY a single JSON object. The word "json" is present here to satisfy the API requirement.
"""

SYSTEM_DS = """
You are the Data Scientist (DS). Execute the AM plan using only available columns.
**Obey action:** Your `action` MUST match `am_next_action_type` exactly. Allowed values: overview, sql, eda, calc, feature_engineering, modeling. If a different action would be better, explain briefly in ds_summary but STILL set `action` to `am_next_action_type`.
**Special rule — Data Inventory:** If AM indicates overview OR the CEO asked “what data do we have,” your action MUST be "overview" and your output should be previews of the **first 5 rows for each table**.
Leverage `column_hints` to map business terms (e.g., revenue → sales/net_sales) to real columns.
Support clustering by setting model_plan.task="clustering" (with optional n_clusters) when `action="modeling"`.
Return JSON fields:
- ds_summary: brief note of intended execution
- action: overview|sql|eda|calc|feature_engineering|modeling  # MUST equal am_next_action_type
- duckdb_sql: SQL string OR list of SQL strings (for eda or feature assembly)
- charts: optional chart specs; either a flat list (applies to first result) or list-of-lists aligned to multiple eda SQLs. Each spec: {title,type,x,y}
- model_plan: {task: classification|regression|clustering|null, target: str|null, features: [], model_family: logistic_regression|random_forest|linear_regression|random_forest_regressor|null, n_clusters: int|null}
- calc_description: string (if action=calc)
- assumptions: string
Return ONLY a single JSON object. The word "json" is present here to satisfy the API requirement.
"""

SYSTEM_DS_REVISE = """
You are the Data Scientist (DS). Revise your prior plan/output based on AM critique.
- Keep to ONLY existing tables/columns.
- Fix suitability issues AM mentioned.
- Keep it concise and executable.
Return JSON with the SAME schema you use normally:
{ "ds_summary": "...", "action": "...", "duckdb_sql": "... or [...]", "charts": [...], "model_plan": {...}, "calc_description": "...", "assumptions": "..." }
Return ONLY a single JSON object. The word "json" is present here to satisfy the API requirement.
"""

SYSTEM_AM_REVIEW = """
You are the AM Reviewer. Given CEO question, AM plan, DS action, and lightweight result meta (shapes/samples/metrics),
write a short plain-language summary for the CEO and critique suitability.
Decide explicitly if the current DS action sufficiently answers the CEO's question.
Return JSON fields:
- summary_for_ceo: 2–4 sentences in plain language
- appropriateness_check: brief assessment of method/query suitability
- gaps_or_risks: brief note on assumptions/data issues
- improvements: [1–4 concrete improvements]
- suggested_next_steps: [1–4 next actions]
- must_revise: boolean  # true if DS should revise before showing to CEO
- sufficient_to_answer: boolean  # true if this DS output answers the CEO's question adequately
- clarification_needed: boolean  # true if more input from CEO is required (e.g., definitions, scope)
- clarifying_questions: [str]  # ask the CEO directly when clarification_needed is true
- revision_notes: string # short guidance for DS on what to fix
Return ONLY a single JSON object. The word "json" is present here to satisfy the API requirement.
"""

SYSTEM_REVIEW = """
You are a Coordinator. Produce a concise revision directive for AM & DS when CEO gives feedback.
Return ONLY a single JSON object. The word "json" is present here to satisfy the API requirement.
"""

SYSTEM_INTENT = """
Classify CEO input relative to the `previous_question` as:
- new_request
- feedback
- answers_to_clarifying
Return ONLY a single JSON object with {"intent": "..."}. The word "json" is present here to satisfy the API requirement.
"""

SYSTEM_COLMAP = """
You resolve business terms in the CEO question to actual columns from the provided tables.
Rules:
- Only return column names that exist.
- Prefer measures over IDs when both match (e.g., sales_amount over sales_id).
- If multiple candidates exist, return a ranked list.
- Include which table each column comes from.
Return JSON fields:
- term_to_columns: { term: [{table: str, column: str}] }
- suggested_features: [{table: str, column: str}]
- notes: short guidance on ambiguities
Return ONLY a single JSON object. The word "json" is present here to satisfy the API requirement.
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
if "tables_raw" not in st.session_state: st.session_state.tables_raw = None  # originals
if "tables_fe"  not in st.session_state: st.session_state.tables_fe  = {}    # feature-engineered snapshots
if "tables"     not in st.session_state: st.session_state.tables     = None  # default view (merged)
if "chat"       not in st.session_state: st.session_state.chat       = []
if "last_rendered_idx" not in st.session_state: st.session_state.last_rendered_idx = 0
if "last_am_json" not in st.session_state: st.session_state.last_am_json = {}
if "last_ds_json" not in st.session_state: st.session_state.last_ds_json = {}
if "last_user_prompt" not in st.session_state: st.session_state.last_user_prompt = ""
if "current_question" not in st.session_state: st.session_state.current_question = ""
if "threads" not in st.session_state: st.session_state.threads = []  # [{question, parts:[...]}]

# Lightweight business term synonyms (heuristic fallback used before LLM col-map)
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
    """
    Robust JSON helper:
    - Ensures 'json' appears in messages (OpenAI requirement when using response_format json_object).
    - Tries structured mode first; on failure, retries without response_format and parses fenced/raw JSON.
    """
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
    """Merge raw + feature-engineered, with FE versions accessible via their own names."""
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
    """Return first non-empty SQL if input is str or list of str; else ''."""
    if isinstance(maybe_sql, str):
        return maybe_sql.strip()
    if isinstance(maybe_sql, list):
        for s in maybe_sql:
            if isinstance(s, str) and s.strip():
                return s.strip()
    return ""


def is_data_inventory_question(text: str) -> bool:
    """Heuristic: force 'overview' for data-inventory prompts."""
    q = (text or "").lower()
    triggers = [
        "what data do we have", "what data do i have", "what data do we have here",
        "what tables do we have", "what datasets", "first 5 rows", "first five rows",
        "preview tables", "show tables", "data inventory"
    ]
    return any(t in q for t in triggers)


def classify_intent(previous_question: str, new_text: str) -> str:
    """LLM-based intent classification with graceful fallback."""
    try:
        payload = {"previous_question": previous_question or "", "new_text": new_text}
        res = llm_json(SYSTEM_INTENT, json.dumps(payload))
        intent = (res or {}).get("intent")
        if intent in {"new_request", "feedback", "answers_to_clarifying"}:
            return intent
    except Exception:
        pass
    # Fallback heuristic: very short or refers to prior result → feedback
    low = (new_text or "").lower()
    if any(w in low for w in ["that", "this", "looks", "seems", "instead", "can you also", "why", "how about"]):
        return "feedback"
    return "new_request"


def build_column_hints(question: str) -> dict:
    """Use LLM + heuristic synonyms to propose relevant columns across raw + FE tables."""
    all_tables = get_all_tables()
    struct = {t: list(df.columns) for t, df in all_tables.items()}
    hints = {"term_to_columns": {}, "suggested_features": [], "notes": ""}
    # Heuristic first
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
    # LLM refinement
    try:
        payload = {"question": question, "tables": struct}
        res = llm_json(SYSTEM_COLMAP, json.dumps(payload)) or {}
        # shallow merge
        if res.get("term_to_columns"):
            hints["term_to_columns"].update(res.get("term_to_columns"))
        if res.get("suggested_features"):
            hints["suggested_features"].extend(res.get("suggested_features"))
        if res.get("notes"):
            hints["notes"] = (hints.get("notes") or "") + " " + res["notes"]
    except Exception:
        pass
    # dedupe suggested_features
    seen = set(); uniq = []
    for it in hints["suggested_features"]:
        key = (it.get("table"), it.get("column"))
        if key not in seen:
            seen.add(key); uniq.append(it)
    hints["suggested_features"] = uniq[:20]
    return hints

# ======================
# Modeling (incl. clustering)
# ======================

def train_model(df: pd.DataFrame, task: str, target: Optional[str], features: List[str], family: str, n_clusters: Optional[int]=None) -> Dict[str, Any]:
    report: Dict[str, Any] = {}

    # ---- Clustering ----
    if task == "clustering":
        use_cols = features if features else [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not use_cols:
            return {"error": "No numeric features available for clustering."}
        X = df[use_cols].copy()
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        k = int(n_clusters or 3)
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = kmeans.fit_predict(Xs)
        try:
            sil = float(silhouette_score(Xs, labels)) if k > 1 else None
        except Exception:
            sil = None
        # PCA for visualization
        try:
            p = PCA(n_components=2, random_state=42)
            coords = p.fit_transform(Xs)
            coords_df = pd.DataFrame({"pc1": coords[:,0], "pc2": coords[:,1], "cluster": labels})
        except Exception:
            coords_df = None
        sizes = pd.Series(labels).value_counts().sort_index().to_dict()
        report.update({
            "task": "clustering",
            "features": use_cols,
            "n_clusters": k,
            "cluster_sizes": {int(k_): int(v) for k_, v in sizes.items()},
            "inertia": float(getattr(kmeans, "inertia_", np.nan)),
            "silhouette": sil,
        })
        return {"report": report, "labels": labels.tolist(), "pca": coords_df}

    # ---- Supervised ----
    if target is None or target not in df.columns:
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
        X, y, test_size=0.25, random_state=42, stratify=y if task == "classification" else None
    )
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    base_report = {"task": task, "target": target, "features": features, "model_family": fam}

    if task == "classification":
        try:
            proba = pipe.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, proba)
        except Exception:
            proba, auc = None, None
        acc = accuracy_score(y_test, (y_pred > 0.5) if proba is not None else y_pred)
        base_report.update({"accuracy": float(acc), "roc_auc": (float(auc) if auc is not None else None)})
    else:
        base_report.update({
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "rmse": float(mean_squared_error(y_test, y_pred, squared=False)),
            "r2": float(r2_score(y_test, y_pred))
        })
    return base_report


# ======================
# AM/DS/Review pipeline
# ======================

def run_am_plan(prompt: str, column_hints: dict) -> dict:
    payload = {
        "ceo_question": prompt,
        "tables": {k: list(v.columns) for k, v in (get_all_tables() or {}).items()},
        "column_hints": column_hints,
    }
    am_json = llm_json(SYSTEM_AM, json.dumps(payload))
    st.session_state.last_am_json = am_json
    add_msg("am", am_json.get("am_brief", ""), artifacts=am_json)
    render_chat()
    return am_json


def run_ds_step(am_json: dict, column_hints: dict) -> dict:
    ds_payload = {
        "am_plan": am_json.get("plan_for_ds", ""),
        "am_next_action_type": am_json.get("next_action_type", "eda"),
        "tables": {k: list(v.columns) for k, v in (get_all_tables() or {}).items()},
        "column_hints": column_hints,
    }
    ds_json = llm_json(SYSTEM_DS, json.dumps(ds_payload))
    st.session_state.last_ds_json = ds_json
    add_msg("ds", ds_json.get("ds_summary", ""),
            artifacts={"action": ds_json.get("action"),
                       "duckdb_sql": ds_json.get("duckdb_sql"),
                       "model_plan": ds_json.get("model_plan")})
    render_chat()
    # Enforce valid/allowed DS action and coerce if needed
    allowed = {"overview","sql","eda","calc","feature_engineering","modeling"}
    am_action = (am_json.get("next_action_type") or "").lower()
    ds_action = (ds_json.get("action") or "").lower()
    if ds_action not in allowed:
        # common synonyms → map to closest valid action
        synonym_map = {
            "aggregate": "sql", "aggregate_sales": "sql", "aggregation": "sql",
            "summarize": "sql", "preview": "overview", "analyze": "eda",
        }
        ds_action = synonym_map.get(ds_action, am_action or "eda")
        ds_json["action"] = ds_action
    # final guard: obey AM when provided
    if am_action in allowed:
        ds_json["action"] = am_action
    return ds_json


def am_review(ceo_prompt: str, ds_json: dict, meta: dict) -> dict:
    bundle = {"ceo_question": ceo_prompt,
              "am_plan": st.session_state.last_am_json,
              "ds_json": ds_json,
              "meta": meta}
    return llm_json(SYSTEM_AM_REVIEW, json.dumps(bundle))


def revise_ds(am_json: dict, prev_ds_json: dict, review_json: dict, column_hints: dict) -> dict:
    payload = {
        "am_plan": am_json.get("plan_for_ds", ""),
        "previous_ds_json": prev_ds_json,
        "am_critique": {
            "appropriateness_check": review_json.get("appropriateness_check"),
            "revision_notes": review_json.get("revision_notes"),
            "gaps_or_risks": review_json.get("gaps_or_risks"),
            "improvements": review_json.get("improvements"),
        },
        "column_hints": column_hints,
    }
    return llm_json(SYSTEM_DS_REVISE, json.dumps(payload))


# ======================
# Build meta for AM review (no render)
# ======================

def build_meta(ds_json: dict) -> dict:
    action = (ds_json.get("action") or "").lower()

    if action == "overview":
        tables_meta = {name: {"rows": len(df), "cols": len(df.columns)} for name, df in get_all_tables().items()}
        return {"type": "overview", "tables": tables_meta}

    if action == "eda":
        raw_sql = ds_json.get("duckdb_sql")
        sql_list = raw_sql if isinstance(raw_sql, list) else [raw_sql]
        metas = []
        for sql in [_sql_first(s) for s in sql_list if s ]:
            try:
                df = run_duckdb_sql(sql)
                metas.append({"sql": sql, "rows": len(df), "cols": list(df.columns),
                              "sample": df.head(10).to_dict(orient="records")})
            except Exception as e:
                metas.append({"sql": sql, "error": str(e)})
        return {"type": "eda", "results": metas}

    if action == "sql":
        sql = _sql_first(ds_json.get("duckdb_sql"))
        if not sql: return {"type": "sql", "error": "No SQL provided"}
        try:
            out = run_duckdb_sql(sql)
            return {"type": "sql", "sql": sql, "rows": len(out), "cols": list(out.columns),
                    "sample": out.head(10).to_dict(orient="records")}
        except Exception as e:
            return {"type": "sql", "sql": sql, "error": str(e)}

    if action == "calc":
        return {"type": "calc", "desc": ds_json.get("calc_description", "")}

    if action == "feature_engineering":
        sql = _sql_first(ds_json.get("duckdb_sql"))
        try:
            base = run_duckdb_sql(sql) if sql else next(iter(get_all_tables().values())).copy()
            return {"type": "feature_engineering", "rows": len(base), "cols": list(base.columns),
                    "sample": base.head(10).to_dict(orient="records")}
        except Exception as e:
            return {"type": "feature_engineering", "error": str(e)}

    if action == "modeling":
        sql = _sql_first(ds_json.get("duckdb_sql"))
        plan = ds_json.get("model_plan") or {}
        target = plan.get("target")
        try:
            base = run_duckdb_sql(sql) if sql else None
            if base is None:
                # search both FE and RAW
                for _, df in get_all_tables().items():
                    if target and target in df.columns:
                        base = df.copy(); break
                if base is None:
                    base = next(iter(get_all_tables().values())).copy()
            return {"type": "modeling", "task": (plan.get("task") or "classification").lower(),
                    "target": target, "features": plan.get("features") or [],
                    "family": (plan.get("model_family") or "logistic_regression").lower(),
                    "n_clusters": plan.get("n_clusters"),
                    "rows": len(base), "cols": list(base.columns)}
        except Exception as e:
            return {"type": "modeling", "error": str(e)}

    return {"type": action or "unknown", "note": "no meta builder"}


# ======================
# Render final result (after review loop)
# ======================

def render_final(ds_json: dict):
    action = (ds_json.get("action") or "").lower()

    # ---- OVERVIEW ----
    if action == "overview":
        st.markdown("### 📊 Table Previews (first 5 rows)")
        for name, df in get_all_tables().items():
            st.markdown(f"**{name}** — rows: {len(df)}, cols: {len(df.columns)}")
            st.dataframe(df.head(5), width="stretch")
        add_msg("ds", "Overview rendered."); render_chat(); return

    # ---- EDA ----
    if action == "eda":
        raw_sql = ds_json.get("duckdb_sql")
        sql_list = raw_sql if isinstance(raw_sql, list) else [raw_sql]
        charts_all = ds_json.get("charts") or []
        for i, sql in enumerate([_sql_first(s) for s in sql_list][:3]):
            if not sql: continue
            try:
                df = run_duckdb_sql(sql)
                st.markdown(f"### 📈 EDA Result #{i+1} (first 50 rows)")
                st.code(sql, language="sql")
                st.dataframe(df.head(50), width="stretch")

                charts_this = []
                if charts_all and isinstance(charts_all[0], dict):
                    charts_this = charts_all if i == 0 else []
                elif charts_all and isinstance(charts_all[0], list):
                    charts_this = charts_all[i] if i < len(charts_all) else []
                for spec in (charts_this or [])[:3]:
                    title = spec.get("title") or "Chart"
                    ctype = (spec.get("type") or "bar").lower()
                    xcol = spec.get("x"); ycol = spec.get("y")
                    if isinstance(xcol,str) and isinstance(ycol,str) and xcol in df.columns and ycol in df.columns:
                        st.markdown(f"**{title}**")
                        plot_df = df[[xcol, ycol]].set_index(xcol)
                        if ctype == "line": st.line_chart(plot_df)
                        elif ctype == "area": st.area_chart(plot_df)
                        else: st.bar_chart(plot_df)
            except Exception as e:
                st.error(f"EDA SQL failed: {e}")
        add_msg("ds","EDA rendered."); render_chat(); return

    # ---- SQL ----
    if action == "sql":
        sql = _sql_first(ds_json.get("duckdb_sql"))
        if not sql:
            add_msg("ds","No SQL provided."); render_chat(); return
        try:
            out = run_duckdb_sql(sql)
            st.markdown("### 🧮 SQL Results (first 25 rows)")
            st.code(sql, language="sql")
            st.dataframe(out.head(25), width="stretch")
            add_msg("ds","SQL executed.", artifacts={"sql": sql}); render_chat()
        except Exception as e:
            st.error(f"SQL failed: {e}")
        return

    # ---- CALC ----
    if action == "calc":
        st.markdown("### 🧮 Calculation")
        st.write(ds_json.get("calc_description","(no description)"))
        add_msg("ds","Calculation displayed."); render_chat(); return

    # ---- FEATURE ENGINEERING ----
    if action == "feature_engineering":
        sql = _sql_first(ds_json.get("duckdb_sql"))
        base = run_duckdb_sql(sql) if sql else next(iter(get_all_tables().values())).copy()
        st.markdown("### 🧱 Feature Engineering Base (first 20 rows)")
        st.dataframe(base.head(20), width="stretch")
        # snapshot FE table for downstream reference
        st.session_state.tables_fe["feature_base"] = base
        add_msg("ds","Feature base ready (saved as 'feature_base')."); render_chat(); return

    # ---- MODELING ----
    if action == "modeling":
        sql = _sql_first(ds_json.get("duckdb_sql"))
        plan = ds_json.get("model_plan") or {}
        task = (plan.get("task") or "classification").lower()
        target = plan.get("target")
        base = run_duckdb_sql(sql) if sql else None
        if base is None:
            # search both FE and RAW
            for _, df in get_all_tables().items():
                if (task == "clustering") or (target and target in df.columns):
                    base = df.copy(); break
            if base is None:
                base = next(iter(get_all_tables().values())).copy()
        if task == "clustering":
            result = train_model(base, task, None, plan.get("features") or [], plan.get("model_family") or "", plan.get("n_clusters"))
            rep = result.get("report", {}) if isinstance(result, dict) else {}
            st.markdown("### 🔍 Clustering Report")
            st.json(rep)
            pca_df = result.get("pca")
            if isinstance(pca_df, pd.DataFrame):
                st.markdown("**PCA Scatter (by cluster)**")
                st.dataframe(pca_df.head(200))
            add_msg("ds","Clustering completed.", artifacts={"report": rep}); render_chat(); return
        else:
            report = train_model(base, task, target, plan.get("features") or [], (plan.get("model_family") or "logistic_regression").lower())
            st.markdown("### 🤖 Model Report")
            st.json(report)
            add_msg("ds","Model trained.", artifacts={"model_report": report}); render_chat(); return

    add_msg("ds", f"Action '{action}' not recognized.", artifacts=ds_json); render_chat()


# ======================
# Coordinator (threading + follow-ups)
# ======================

def run_turn_ceo(new_text: str):
    prev = st.session_state.current_question or ""
    intent = classify_intent(prev, new_text)

    if intent in {"feedback", "answers_to_clarifying"} and prev:
        effective_q = f"{prev.strip()}

[Follow-up]: {new_text.strip()}"
        add_msg("system", f"Interpreting your message as {intent.replace('_',' ')} on the previous question.")
    else:
        effective_q = new_text
        if prev:
            # finalize previous thread
            st.session_state.threads.append({"question": prev, "parts": []})

    st.session_state.current_question = effective_q
    st.session_state.last_user_prompt = new_text

    # 0) Column hints from RAW + FE
    col_hints = build_column_hints(effective_q)

    # 1) AM plan (with inventory short-circuit in prompts but still reviewed)
    am_json = run_am_plan(effective_q, col_hints)

    # If AM needs user info, ask and stop until the CEO replies
    if am_json.get("need_more_info"):
        qs = am_json.get("clarifying_questions") or ["Could you clarify your objective?"]
        add_msg("am", "I need a bit more context:")
        for q in qs[:3]:
            add_msg("am", f"• {q}")
        render_chat();
        return

    # 2) DS executes and enters review→revise loop
    max_loops = 3
    loop_count = 0
    ds_json = run_ds_step(am_json, col_hints)

    while loop_count < max_loops:
        loop_count += 1

        # Build meta & AM review for EVERY DS action
        meta = build_meta(ds_json)
        review = am_review(effective_q, ds_json, meta)
        add_msg("am", review.get("summary_for_ceo",""), artifacts={
            "appropriateness_check": review.get("appropriateness_check"),
            "gaps_or_risks": review.get("gaps_or_risks"),
            "improvements": review.get("improvements"),
            "suggested_next_steps": review.get("suggested_next_steps"),
            "must_revise": review.get("must_revise"),
            "sufficient_to_answer": review.get("sufficient_to_answer"),
        })
        render_chat()

        # If AM needs clarification from CEO, ask and pause the run
        if review.get("clarification_needed") and (review.get("clarifying_questions") or []):
            add_msg("am", "Before proceeding, could you clarify:")
            for q in (review.get("clarifying_questions") or [])[:3]:
                add_msg("am", f"• {q}")
            render_chat();
            return

        # If sufficient and no revision required → render final and exit
        if review.get("sufficient_to_answer") and not review.get("must_revise"):
            render_final(ds_json)
            return

        # Otherwise revise if requested
        if review.get("must_revise"):
            ds_json = revise_ds(am_json, ds_json, review, col_hints)
            add_msg("ds", ds_json.get("ds_summary","(revised)"), artifacts={"action": ds_json.get("action")}); render_chat()
            continue
        else:
            # If not sufficient but also not explicitly must_revise, nudge a targeted revision once
            review_fallback = {"appropriateness_check": "Not sufficient; attempt targeted refinement.", "revision_notes": "Tighten alignment to CEO question and use column_hints."}
            ds_json = revise_ds(am_json, ds_json, review_fallback, col_hints)
            add_msg("ds", ds_json.get("ds_summary","(auto-revised)"), artifacts={"action": ds_json.get("action")}); render_chat()
            continue

    # If we exit loop without sufficiency, render whatever we have but note limitations
    add_msg("system", "Reached review limit; presenting current best effort with noted caveats.")
    render_final(ds_json)
    return
    am_json = run_am_plan(effective_q, col_hints)
    if am_json.get("need_more_info"):
        add_msg("am", "Could you clarify?", artifacts=am_json); render_chat(); return

    # 2) DS step
    ds_json = run_ds_step(am_json, col_hints)

    # 2a) Force OVERVIEW for inventory-style questions (no EDA at this time)
    if is_data_inventory_question(effective_q):
        ds_json = {**ds_json, "action": "overview", "duckdb_sql": None}
        add_msg("system", "Routing to OVERVIEW for data inventory request."); render_chat()

    # 3) Build meta & AM review
    meta = build_meta(ds_json)
    review = am_review(effective_q, ds_json, meta)
    add_msg("am", review.get("summary_for_ceo",""), artifacts={
        "appropriateness_check": review.get("appropriateness_check"),
        "gaps_or_risks": review.get("gaps_or_risks"),
        "improvements": review.get("improvements"),
        "suggested_next_steps": review.get("suggested_next_steps"),
        "must_revise": review.get("must_revise"),
    })
    render_chat()

    # 4) If must revise, let DS revise once, then render
    if review.get("must_revise") is True:
        ds_json = revise_ds(am_json, ds_json, review, col_hints)
        add_msg("ds", ds_json.get("ds_summary","(revised)"), artifacts={"action": ds_json.get("action")}); render_chat()
        # rebuild meta (optional) and continue to final render

    # 5) Final render
    render_final(ds_json)


# ======================
# Data loading
# ======================
if zip_file and st.session_state.tables_raw is None:
    st.session_state.tables_raw = load_zip_tables(zip_file)
    st.session_state.tables = get_all_tables()
    add_msg("system", f"Loaded {len(st.session_state.tables_raw)} raw tables."); render_chat()


# ======================
# Chat UI
# ======================
st.subheader("Chat")
render_chat()

user_prompt = st.chat_input("You're the CEO. Ask a question (e.g., 'What data do we have?' or 'How to improve profit?')")
if user_prompt:
    add_msg("user", user_prompt); render_chat()
    run_turn_ceo(user_prompt)
