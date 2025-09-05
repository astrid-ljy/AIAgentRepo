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
        res = llm_json(SYSTEM_COLMAP, json.dumps(payload)) or {}
        if res.get("term_to_columns"):
            hints["term_to_columns"].update(res.get("term_to_columns"))
        if res.get("suggested_features"):
            hints["suggested_features"].extend(res.get("suggested_features"))
        if res.get("notes"):
            hints["notes"] = (hints.get("notes") or "") + " " + res["notes"]
    except Exception:
        pass
    seen = set(); uniq = []
    for it in hints["suggested_features"]:
        key = (it.get("table"), it.get("column"))
        if key not in seen:
            seen.add(key); uniq.append(it)
    hints["suggested_features"] = uniq[:20]
    return hints


# ======================
# Profiling → feature proposal → preflight
# ======================
def profile_columns(df: pd.DataFrame) -> pd.DataFrame:
    prof = []
    for c in df.columns:
        s = df[c]
        t = "numeric" if pd.api.types.is_numeric_dtype(s) else ("datetime" if np.issubdtype(s.dtype, np.datetime64) else "categorical")
        nonnull = int(s.notna().sum())
        distinct = int(s.nunique(dropna=True))
        uniq_ratio = distinct / max(nonnull, 1) if nonnull else 0.0
        flags = []
        name = str(c).lower()
        if re.search(r"(^id$|_id$|uuid|guid|hash|^code$|_code$)", name):
            flags.append("id_like")
        if re.search(r"(zip|postal|cep|postcode|latitude|longitude|lat|lon)", name):
            flags.append("geo_like")
        if t == "datetime":
            flags.append("datetime")
        if (t == "numeric" and (s.std(skipna=True) == 0)) or (distinct <= 1):
            flags.append("near_constant")
        prof.append({"col": c, "dtype": str(s.dtype), "t": t,
                     "nonnull": nonnull, "distinct": distinct,
                     "uniq_ratio": float(uniq_ratio), "flags": flags})
    return pd.DataFrame(prof)


def propose_features(task: str, prof: pd.DataFrame, allow_geo: bool=False) -> dict:
    bad = set()
    for _, r in prof.iterrows():
        rflags = set(r.get("flags", []) if isinstance(r.get("flags", []), (list, tuple, set)) else [])
        col = r.get("col")
        if col is None: 
            continue
        if "id_like" in rflags or "near_constant" in rflags or "datetime" in rflags:
            bad.add(col)
        if "geo_like" in rflags and not allow_geo:
            bad.add(col)
    if task == "clustering":
        selected = prof[(prof["t"]=="numeric") & (~prof["col"].isin(list(bad))) ]["col"].tolist()
    else:
        selected = prof[(prof["t"]=="numeric") & (~prof["col"].isin(list(bad))) ]["col"].tolist()
    excluded = []
    for _, r in prof.iterrows():
        col = r.get("col")
        if col in bad:
            rflags = r.get("flags", [])
            excluded.append({"col": col, "reason": " | ".join(rflags) if isinstance(rflags, (list, tuple)) else str(rflags)})
    return {"selected_features": selected, "excluded_features": excluded}


def preflight(task: str, proposal: dict) -> tuple[bool, str]:
    feats = proposal.get("selected_features") or []
    if task == "clustering" and len(feats) < 2:
        return False, "Need at least 2 numeric features for clustering after filtering."
    if not_feats := len(feats) == 0:
        return False, "No usable features after filtering."
    return True, ""


# ======================
# Smart base chooser (browses all tables)
# ======================
DIM_PAT = re.compile(r"product_(weight_g|length_cm|height_cm|width_cm)$", re.I)

def choose_model_base(plan: dict, question_text: str) -> Optional[pd.DataFrame]:
    """
    Browse ALL loaded tables and pick the best single base:
    - If the question mentions product/dimension: pick the table with the MOST numeric dimension-like columns.
    - Else score each table by count of usable numeric features (post filtering) and choose the best.
    (You can still join with other tables via SQL in earlier steps.)
    """
    tables = get_all_tables()
    if not tables: return None
    qlow = (question_text or "").lower()

    # If product/dimension ask → prefer most dimension-like columns
    if re.search(r"\bproduct\b", qlow) or re.search(r"\bdimension", qlow):
        best_name, best_score = None, -1
        for name, df in tables.items():
            dims = [c for c in df.columns if DIM_PAT.search(str(c)) and pd.api.types.is_numeric_dtype(df[c])]
            score = len(dims)
            if score > best_score:
                best_name, best_score = name, score
        if best_name is not None and best_score >= 2:
            return tables[best_name].copy()

    # Generic scoring by usable numeric features
    best_name, best_score = None, -1
    for name, df in tables.items():
        prof = profile_columns(df)
        prop = propose_features("clustering", prof, allow_geo=False)
        score = len(prop.get("selected_features", []))
        if score > best_score:
            best_name, best_score = name, score
    return tables[best_name].copy() if best_name else next(iter(tables.values())).copy()


# ======================
# Modeling helpers
# ======================
def infer_default_model_plan(question_text: str, plan: Optional[dict]) -> dict:
    """
    Ensure modeling defaults make sense. If task/target are missing AND the CEO likely
    wants clustering (or no target is provided), default to KMeans clustering.
    """
    plan = dict(plan or {})
    q = (question_text or "").lower()
    wants_cluster = any(k in q for k in ["cluster", "clustering", "segment", "segmentation", "kmeans", "dimension"])
    task = (plan.get("task") or "").lower()
    target = plan.get("target")

    if wants_cluster or not target or task == "clustering":
        plan.setdefault("task", "clustering")
        plan.setdefault("model_family", "kmeans")
        plan.setdefault("n_clusters", plan.get("n_clusters", 5))
        plan.setdefault("features", plan.get("features", []))
        plan.pop("target", None)  # not used in clustering
    return plan


# ======================
# Modeling (incl. clustering)
# ======================
def train_model(df: pd.DataFrame, task: str, target: Optional[str], features: List[str], family: str, n_clusters: Optional[int]=None) -> Dict[str, Any]:
    if task == "clustering":
        # Prefer explicit dimension columns if present
        dim_like = [c for c in df.columns if DIM_PAT.search(str(c))]
        dim_like = [c for c in dim_like if pd.api.types.is_numeric_dtype(df[c])]
        if not features and len(dim_like) >= 2:
            use_cols = dim_like
        else:
            if not features:
                prof = profile_columns(df)
                prop = propose_features("clustering", prof, allow_geo=False)
                ok, msg = preflight("clustering", prop)
                if not ok:
                    return {"error": msg, "feature_proposal": prop}
                use_cols = prop["selected_features"]
            else:
                use_cols = [c for c in features if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
                if len(use_cols) < 2:
                    prof = profile_columns(df)
                    prop = propose_features("clustering", prof, allow_geo=False)
                    ok, msg = preflight("clustering", prop)
                    if not ok:
                        return {"error": msg, "feature_proposal": prop}
                    use_cols = prop["selected_features"]

        X = df[use_cols].copy()
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        k = int(n_clusters or 5)
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(Xs)
        try:
            sil = float(silhouette_score(Xs, labels)) if k > 1 else None
        except Exception:
            sil = None
        try:
            p = PCA(n_components=2, random_state=42)
            coords = p.fit_transform(Xs)
            coords_df = pd.DataFrame({"pc1": coords[:,0], "pc2": coords[:,1], "cluster": labels})
        except Exception:
            coords_df = None
        try:
            centers_std = km.cluster_centers_
            centers_orig = scaler.inverse_transform(centers_std)
            centroids_df = pd.DataFrame(centers_orig, columns=use_cols)
        except Exception:
            centroids_df = None

        sizes = pd.Series(labels).value_counts().sort_index().to_dict()
        report = {
            "task": "clustering",
            "features": use_cols,
            "n_clusters": k,
            "cluster_sizes": {int(k_): int(v) for k_, v in sizes.items()},
            "inertia": float(getattr(km, "inertia_", np.nan)),
            "silhouette": sil,
        }
        return {"report": report, "labels": labels.tolist(), "pca": coords_df, "centroids": centroids_df}

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
def run_am_plan(prompt: str, column_hints: dict, context: dict) -> dict:
    payload = {
        "ceo_question": prompt,
        "tables": {k: list(v.columns) for k, v in (get_all_tables() or {}).items()},
        "column_hints": column_hints,
        "context": context,
    }
    am_json = llm_json(SYSTEM_AM, json.dumps(payload))
    st.session_state.last_am_json = am_json
    add_msg("am", am_json.get("am_brief", ""), artifacts=am_json)
    render_chat()
    return am_json


def _coerce_allowed(action: Optional[str], fallback: str) -> str:
    allowed = {"overview","sql","eda","calc","feature_engineering","modeling","explain"}
    a = (action or "").lower()
    if a in allowed: return a
    synonym_map = {
        "aggregate": "sql", "aggregate_sales": "sql", "aggregation": "sql",
        "summarize": "explain", "preview": "overview", "analyze": "eda", "interpret": "explain",
        "explanation": "explain"
    }
    return synonym_map.get(a, fallback if fallback in allowed else "eda")


def _normalize_sequence(seq, fallback_action) -> List[dict]:
    out: List[dict] = []
    for raw in (seq or [])[:5]:
        if isinstance(raw, dict):
            a = _coerce_allowed(raw.get("action"), fallback_action)
            plan = raw.get("model_plan")
            if a == "modeling":
                plan = infer_default_model_plan(st.session_state.current_question, plan)
            out.append({
                "action": a,
                "duckdb_sql": raw.get("duckdb_sql"),
                "charts": raw.get("charts"),
                "model_plan": plan,
                "calc_description": raw.get("calc_description"),
            })
        elif isinstance(raw, str):
            a = _coerce_allowed(raw, fallback_action)
            plan = infer_default_model_plan(st.session_state.current_question, {} ) if a == "modeling" else None
            out.append({"action": a, "duckdb_sql": None, "charts": None,
                        "model_plan": plan, "calc_description": None})
    if not out:
        out = [{"action": _coerce_allowed(None, fallback_action),
                "duckdb_sql": None, "charts": None,
                "model_plan": None, "calc_description": None}]
    return out


def run_ds_step(am_json: dict, column_hints: dict, thread_ctx: dict) -> dict:
    ds_payload = {
        "am_plan": am_json.get("plan_for_ds", ""),
        "am_next_action_type": am_json.get("next_action_type", "eda"),
        "am_action_sequence": am_json.get("action_sequence", []),
        "tables": {k: list(v.columns) for k, v in (get_all_tables() or {}).items()},
        "column_hints": column_hints,
        "current_question": thread_ctx.get("current_question"),
        "central_question": thread_ctx.get("central_question"),
        "prior_questions": thread_ctx.get("prior_questions", []),
    }
    ds_json = llm_json(SYSTEM_DS, json.dumps(ds_payload))
    st.session_state.last_ds_json = ds_json

    am_mode = (am_json.get("task_mode") or ("multi" if am_json.get("action_sequence") else "single")).lower()
    if am_mode == "multi":
        seq = ds_json.get("action_sequence") or am_json.get("action_sequence") or []
        norm_seq = _normalize_sequence(seq, (am_json.get("next_action_type") or "eda").lower())
        ds_json["action_sequence"] = norm_seq
        add_msg("ds", ds_json.get("ds_summary", ""), artifacts={"mode": "multi", "sequence": norm_seq})
    else:
        a = _coerce_allowed(ds_json.get("action"), (am_json.get("next_action_type") or "eda").lower())
        ds_json["action"] = (am_json.get("next_action_type") or a)
        if ds_json["action"] == "modeling":
            ds_json["model_plan"] = infer_default_model_plan(st.session_state.current_question, ds_json.get("model_plan"))
        add_msg("ds", ds_json.get("ds_summary", ""), artifacts={
            "action": ds_json.get("action"),
            "duckdb_sql": ds_json.get("duckdb_sql"),
            "model_plan": ds_json.get("model_plan")
        })

    render_chat()
    return ds_json


def am_review(ceo_prompt: str, ds_json: dict, meta: dict) -> dict:
    bundle = {"ceo_question": ceo_prompt,
              "am_plan": st.session_state.last_am_json,
              "ds_json": ds_json,
              "meta": meta,
              "central_question": st.session_state.central_question,
              "prior_questions": st.session_state.prior_questions}
    return llm_json(SYSTEM_AM_REVIEW, json.dumps(bundle))


def revise_ds(am_json: dict, prev_ds_json: dict, review_json: dict, column_hints: dict, thread_ctx: dict) -> dict:
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
        "central_question": thread_ctx.get("central_question"),
        "prior_questions": thread_ctx.get("prior_questions", []),
    }
    return llm_json(SYSTEM_DS_REVISE, json.dumps(payload))


# ======================
# Build meta for AM review (supports sequence and explain)
# ======================
def build_meta_for_action(ds_step: dict) -> dict:
    action = (ds_step.get("action") or "").lower()

    if action == "overview":
        tables_meta = {name: {"rows": len(df), "cols": len(df.columns)} for name, df in get_all_tables().items()}
        return {"type": "overview", "tables": tables_meta}

    if action == "eda":
        raw_sql = ds_step.get("duckdb_sql")
        sql_list = raw_sql if isinstance(raw_sql, list) else [raw_sql]
        metas = []
        for sql in [_sql_first(s) for s in sql_list if s]:
            try:
                df = run_duckdb_sql(sql)
                metas.append({"sql": sql, "rows": len(df), "cols": list(df.columns),
                              "sample": df.head(10).to_dict(orient="records")})
            except Exception as e:
                metas.append({"sql": sql, "error": str(e)})
        return {"type": "eda", "results": metas}

    if action == "sql":
        sql = _sql_first(ds_step.get("duckdb_sql"))
        if not sql: return {"type": "sql", "error": "No SQL provided"}
        try:
            out = run_duckdb_sql(sql)
            return {"type": "sql", "sql": sql, "rows": len(out), "cols": list(out.columns),
                    "sample": out.head(10).to_dict(orient="records")}
        except Exception as e:
            return {"type": "sql", "sql": sql, "error": str(e)}

    if action == "calc":
        return {"type": "calc", "desc": ds_step.get("calc_description", "")}

    if action == "feature_engineering":
        sql = _sql_first(ds_step.get("duckdb_sql"))
        try:
            base = run_duckdb_sql(sql) if sql else next(iter(get_all_tables().values())).copy()
            return {"type": "feature_engineering", "rows": len(base), "cols": list(base.columns),
                    "sample": base.head(10).to_dict(orient="records")}
        except Exception as e:
            return {"type": "feature_engineering", "error": str(e)}

    if action == "modeling":
        sql = _sql_first(ds_step.get("duckdb_sql"))
        plan = infer_default_model_plan(st.session_state.current_question, ds_step.get("model_plan") or {})
        target = plan.get("target")
        try:
            base = run_duckdb_sql(sql) if sql else None
            if base is None:
                base = choose_model_base(plan, st.session_state.current_question)
            if base is None:
                return {"type":"modeling","error":"No tables loaded."}
            return {"type": "modeling", "task": (plan.get("task") or "clustering").lower(),
                    "target": target, "features": plan.get("features") or [],
                    "family": (plan.get("model_family") or "kmeans").lower(),
                    "n_clusters": plan.get("n_clusters"),
                    "rows": len(base), "cols": list(base.columns)}
        except Exception as e:
            return {"type": "modeling", "error": str(e)}

    if action == "explain":
        cache = st.session_state.last_results
        meta = {"type": "explain"}
        if cache.get("clustering"):
            rep = cache["clustering"]
            meta["clustering"] = {k: rep.get(k) for k in ["features","n_clusters","silhouette","cluster_sizes"]}
        if cache.get("modeling"):
            m = cache["modeling"]
            meta["modeling"] = {k: m.get(k) for k in ["task","target","features","metrics"]}
        if cache.get("eda"):
            meta["eda"] = {"sqls": cache["eda"].get("sqls"), "sample_cols": cache["eda"].get("sample_cols")}
        if cache.get("feature_engineering"):
            meta["feature_engineering"] = {"rows": cache["feature_engineering"].get("rows"), "cols": cache["feature_engineering"].get("cols")}
        if cache.get("sql"):
            meta["sql"] = {"sql": cache["sql"].get("sql"), "rows": cache["sql"].get("rows")}
        if not any([cache.get("clustering"), cache.get("modeling"), cache.get("eda"), cache.get("feature_engineering"), cache.get("sql")]):
            meta["error"] = "no_cache"
        return meta

    return {"type": action or "unknown", "note": "no meta builder"}


# ======================
# Renderer for actions (with caching for explain)
# ======================
def render_final_for_action(ds_step: dict):
    action = (ds_step.get("action") or "").lower()

    # ---- OVERVIEW ----
    if action == "overview":
        st.markdown("### 📊 Table Previews (first 5 rows)")
        for name, df in get_all_tables().items():
            st.markdown(f"**{name}** — rows: {len(df)}, cols: {len(df.columns)}")
            st.dataframe(df.head(5), width="stretch")
        add_msg("ds", "Overview rendered.")
        return

    # ---- EDA ----
    if action == "eda":
        raw_sql = ds_step.get("duckdb_sql")
        sql_list = raw_sql if isinstance(raw_sql, list) else [raw_sql]
        charts_all = ds_step.get("charts") or []
        executed_sqls, last_cols = [], None
        for i, sql in enumerate([_sql_first(s) for s in sql_list][:3]):
            if not sql: continue
            try:
                df = run_duckdb_sql(sql)
                executed_sqls.append(sql)
                last_cols = list(df.columns)
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
        st.session_state.last_results["eda"] = {"sqls": executed_sqls, "sample_cols": last_cols}
        add_msg("ds","EDA rendered.")
        return

    # ---- SQL ----
    if action == "sql":
        sql = _sql_first(ds_step.get("duckdb_sql"))
        if not sql:
            add_msg("ds","No SQL provided.")
            return
        try:
            out = run_duckdb_sql(sql)
            st.markdown("### 🧮 SQL Results (first 25 rows)")
            st.code(sql, language="sql")
            st.dataframe(out.head(25), width="stretch")
            add_msg("ds","SQL executed.", artifacts={"sql": sql})
            st.session_state.last_results["sql"] = {"sql": sql, "rows": len(out), "cols": list(out.columns)}
        except Exception as e:
            st.error(f"SQL failed: {e}")
        return

    # ---- CALC ----
    if action == "calc":
        st.markdown("### 🧮 Calculation")
        st.write(ds_step.get("calc_description","(no description)"))
        add_msg("ds","Calculation displayed.")
        return

    # ---- FEATURE ENGINEERING ----
    if action == "feature_engineering":
        sql = _sql_first(ds_step.get("duckdb_sql"))
        base = run_duckdb_sql(sql) if sql else next(iter(get_all_tables().values())).copy()
        st.markdown("### 🧱 Feature Engineering Base (first 20 rows)")
        st.dataframe(base.head(20), width="stretch")
        st.session_state.tables_fe["feature_base"] = base
        st.session_state.last_results["feature_engineering"] = {"rows": len(base), "cols": list(base.columns)}
        add_msg("ds","Feature base ready (saved as 'feature_base').")
        return

    # ---- MODELING ----
    if action == "modeling":
        sql = _sql_first(ds_step.get("duckdb_sql"))
        plan = infer_default_model_plan(st.session_state.current_question, ds_step.get("model_plan") or {})
        task = (plan.get("task") or "clustering").lower()
        target = plan.get("target")
        base = run_duckdb_sql(sql) if sql else None
        if base is None:
            base = choose_model_base(plan, st.session_state.current_question)

        if task == "clustering":
            result = train_model(base, task, None, plan.get("features") or [], plan.get("model_family") or "", plan.get("n_clusters"))
            if result.get("error"):
                st.error(result.get("error"))
                if result.get("feature_proposal"):
                    st.markdown("**Feature Proposal (auto-derived):**")
                    st.json(result["feature_proposal"])
                return
            rep = result.get("report", {}) if isinstance(result, dict) else {}
            st.markdown("### 🔍 Clustering Report")
            st.json(rep)
            pca_df = result.get("pca")
            if isinstance(pca_df, pd.DataFrame):
                st.markdown("**PCA Scatter (by cluster)**")
                st.dataframe(pca_df.head(200))
            add_msg("ds","Clustering completed.", artifacts={"report": rep})
            # cache
            st.session_state.last_results["clustering"] = {
                "report": rep,
                "features": list(rep.get("features") or []),
                "n_clusters": int(rep.get("n_clusters") or 0),
                "cluster_sizes": rep.get("cluster_sizes") or {},
                "silhouette": rep.get("silhouette"),
                "centroids": (result.get("centroids").to_dict(orient="records") if isinstance(result.get("centroids"), pd.DataFrame) else None)
            }
            st.session_state.last_results["modeling"] = {
                "task": "clustering",
                "target": None,
                "features": list(rep.get("features") or []),
                "metrics": {"silhouette": rep.get("silhouette"), "inertia": rep.get("inertia")}
            }
            return

        report = train_model(base, task, target, plan.get("features") or [], (plan.get("model_family") or "logistic_regression").lower())
        if isinstance(report, dict) and report.get("error"):
            st.error(report.get("error"))
        st.markdown("### 🤖 Model Report")
        st.json(report)
        add_msg("ds","Model trained.", artifacts={"model_report": report})
        st.session_state.last_results["modeling"] = {
            "task": task, "target": target, "features": report.get("features"),
            "metrics": {k: report.get(k) for k in ["accuracy","roc_auc","mae","rmse","r2"] if k in report}
        }
        return

    # ---- EXPLAIN ----
    if action == "explain":
        cache = st.session_state.last_results
        if not any(cache.values()):
            add_msg("ds", "I don’t have cached results yet. Please run an analysis step first.")
            st.info("No cached results found. Run an analysis first, then ask for interpretation.")
            return

        st.markdown("### 📝 Interpretation of Latest Results")

        if cache.get("clustering"):
            ctx = cache["clustering"]
            st.markdown("#### Clustering")
            st.markdown("**Features used:** " + (", ".join(ctx.get("features") or []) or "_not recorded_"))
            st.markdown(f"**k (clusters):** {ctx.get('n_clusters')}")
            st.markdown(f"**Silhouette:** {ctx.get('silhouette')}")
            st.markdown("**Cluster sizes:**")
            st.json(ctx.get("cluster_sizes") or {})
            if ctx.get("centroids"):
                st.markdown("**Centroids (original units):**")
                st.dataframe(pd.DataFrame(ctx["centroids"]))
                try:
                    cdf = pd.DataFrame(ctx["centroids"])
                    bullets = []
                    for i, row in cdf.iterrows():
                        top = row.sort_values(ascending=False).head(min(3, len(row))).index.tolist()
                        bullets.append(f"- Cluster {i}: high on {', '.join(top)}")
                    st.markdown("**Heuristic labels:**")
                    for b in bullets: st.write(b)
                except Exception:
                    pass
            st.markdown("**Next:** Join clusters back to sales/margin/shipping for pricing & ops.")
        elif cache.get("modeling"):
            m = cache["modeling"]
            st.markdown("#### Modeling")
            st.json(m)
        elif cache.get("eda"):
            e = cache["eda"]
            st.markdown("#### EDA")
            st.write("Recent EDA queries:")
            for s in (e.get("sqls") or [])[:5]:
                st.code(s, language="sql")
            st.write("Sample columns observed:", e.get("sample_cols"))
        elif cache.get("feature_engineering"):
            fe = cache["feature_engineering"]
            st.markdown("#### Feature Engineering")
            st.json(fe)
        elif cache.get("sql"):
            sq = cache["sql"]
            st.markdown("#### SQL result context")
            st.json(sq)
        add_msg("ds", "Provided interpretation without re-running any steps.", artifacts={"explain_used": True})
        return

    add_msg("ds", f"Action '{action}' not recognized.", artifacts=ds_step)


# ======================
# Auto-progression heuristic
# ======================
def must_progress_to_modeling(thread_ctx: dict, am_json: dict, ds_json: dict) -> bool:
    """
    Decide whether to auto-advance to modeling (usually clustering).
    Conservative default: False unless the CEO clearly asked for modeling/segmentation
    and neither AM nor DS already planned to model in this turn.
    """
    if (am_json.get("next_action_type") or "").lower() == "modeling":
        return False
    if (ds_json.get("action") or "").lower() == "modeling":
        return False
    if any((isinstance(s, dict) and (s.get("action","").lower() == "modeling"))
           or (isinstance(s, str) and s.lower() == "modeling")
           for s in (ds_json.get("action_sequence") or [])):
        return False

    q = (thread_ctx.get("current_question") or thread_ctx.get("central_question") or "").lower()

    if re.search(r"\bwhat (data|datasets?) do we have\b", q) or "first 5 rows" in q:
        return False

    wants_model = any(k in q for k in [
        "cluster", "clustering", "segment", "segmentation", "kmeans",
        "unsupervised", "model", "predict", "classification", "regression"
    ])
    return bool(wants_model)


# ======================
# Coordinator (threading + arbitration + follow-ups)
# ======================
def run_turn_ceo(new_text: str):

    prev = st.session_state.current_question or ""
    central = st.session_state.central_question or ""
    prior = st.session_state.prior_questions or []

    # Explicit "not a follow up" starts a new thread
    force_new = _explicit_new_thread(new_text)

    ic = classify_intent(prev, central, prior, new_text)
    intent = "new_request" if force_new else ic.get("intent", "new_request")
    related = False if force_new else ic.get("related", False)

    # Manage threads & central question
    if intent == "new_request" and not related:
        if st.session_state.central_question:
            st.session_state.threads.append({"central": st.session_state.central_question, "followups": []})
        st.session_state.central_question = new_text
        st.session_state.current_question = new_text
    else:
        st.session_state.current_question = (central or new_text).strip() + "\n\n[Follow-up]: " + (new_text or "").strip()

    # Track prior questions
    if prev and prev not in st.session_state.prior_questions:
        st.session_state.prior_questions.append(prev)
    if new_text and new_text not in st.session_state.prior_questions:
        st.session_state.prior_questions.append(new_text)

    add_msg("system", "Context updated: central question & history considered.", artifacts={
        "intent": intent, "related": related,
        "central_question": st.session_state.central_question,
        "current_question": st.session_state.current_question,
    })
    render_chat()

    effective_q = st.session_state.current_question

    # 0) Column hints
    col_hints = build_column_hints(effective_q)

    # Context pack for AM/DS
    thread_ctx = {
        "central_question": st.session_state.central_question,
        "current_question": st.session_state.current_question,
        "prior_questions": st.session_state.prior_questions,
    }

    # 1) AM plan
    am_json = run_am_plan(effective_q, col_hints, context=thread_ctx)

    # If AM needs user info, ask and stop
    if am_json.get("need_more_info"):
        qs = am_json.get("clarifying_questions") or ["Could you clarify your objective?"]
        add_msg("am", "I need a bit more context:")
        for q in qs[:3]:
            add_msg("am", f"• {q}")
        render_chat(); return

    # 2) DS executes and enters review loop
    max_loops = 3
    loop_count = 0
    ds_json = run_ds_step(am_json, col_hints, thread_ctx)

    if must_progress_to_modeling(thread_ctx, am_json, ds_json):
        am_json = {**am_json, "task_mode":"single", "next_action_type":"modeling", "plan_for_ds": (am_json.get("plan_for_ds") or "") + " | Proceed to clustering."}
        ds_json = run_ds_step(am_json, col_hints, thread_ctx)

    # If DS asks for clarification explicitly
    if ds_json.get("need_more_info") and (ds_json.get("clarifying_questions") or []):
        add_msg("ds", "Before running further steps, I need:")
        for q in (ds_json.get("clarifying_questions") or [])[:3]:
            add_msg("ds", f"• {q}")
        render_chat(); return

    # Build metas (sequence-aware)
    def _build_metas(ds_json_local: dict) -> Union[dict, List[dict]]:
        if ds_json_local.get("action_sequence"):
            metas = []
            for step in ds_json_local.get("action_sequence")[:5]:
                metas.append(build_meta_for_action(step))
            return metas
        else:
            return build_meta_for_action({
                "action": ds_json_local.get("action"),
                "duckdb_sql": ds_json_local.get("duckdb_sql"),
                "charts": ds_json_local.get("charts"),
                "model_plan": ds_json_local.get("model_plan"),
                "calc_description": ds_json_local.get("calc_description"),
            })

    # Render (sequence-aware)
    def _render(ds_json_local: dict):
        if ds_json_local.get("action_sequence"):
            for step in ds_json_local.get("action_sequence")[:5]:
                render_final_for_action(step)
            add_msg("system", "Multi-step run rendered.")
            render_chat()
        else:
            if not ds_json_local.get("action"):
                ds_json_local["action"] = am_json.get("next_action_type") or "eda"
            render_final_for_action({
                "action": ds_json_local.get("action"),
                "duckdb_sql": ds_json_local.get("duckdb_sql"),
                "charts": ds_json_local.get("charts"),
                "model_plan": ds_json_local.get("model_plan"),
                "calc_description": ds_json_local.get("calc_description"),
            })
            render_chat()

    while loop_count < max_loops:
        loop_count += 1

        meta = _build_metas(ds_json)
        review = am_review(effective_q, ds_json, {"meta": meta, "mode": "multi" if ds_json.get("action_sequence") else "single"})
        add_msg("am", review.get("summary_for_ceo",""), artifacts={
            "appropriateness_check": review.get("appropriateness_check"),
            "gaps_or_risks": review.get("gaps_or_risks"),
            "improvements": review.get("improvements"),
            "suggested_next_steps": review.get("suggested_next_steps"),
            "must_revise": review.get("must_revise"),
            "sufficient_to_answer": review.get("sufficient_to_answer"),
        })
        render_chat()

        # If AM needs clarification from CEO
        if review.get("clarification_needed") and (review.get("clarifying_questions") or []):
            add_msg("am", "Before proceeding, could you clarify:")
            for q in (review.get("clarifying_questions") or [])[:3]:
                add_msg("am", f"• {q}")
            render_chat(); return

        # If sufficient and no revision required → render final and exit
        if review.get("sufficient_to_answer") and not review.get("must_revise"):
            _render(ds_json)
            return

        # Otherwise revise if requested
        if review.get("must_revise"):
            ds_json = revise_ds(am_json, ds_json, review, col_hints, thread_ctx)
            if ds_json.get("action_sequence"):
                ds_json["action_sequence"] = _normalize_sequence(
                    ds_json.get("action_sequence"),
                    (am_json.get("next_action_type") or "eda").lower()
                )
            add_msg("ds", ds_json.get("ds_summary","(revised)"),
                    artifacts={"mode": "multi" if ds_json.get("action_sequence") else "single"})
            render_chat()
            continue  # loop for another review

        # No revision required → render current best and exit
        _render(ds_json)
        return

    add_msg("system", "Reached review limit; presenting current best effort with noted caveats.")
    _render(ds_json)
    return


# ======================
# Data loading
# ======================
def load_if_needed():
    if zip_file and st.session_state.tables_raw is None:
        st.session_state.tables_raw = load_zip_tables(zip_file)
        st.session_state.tables = get_all_tables()
        add_msg("system", f"Loaded {len(st.session_state.tables_raw)} raw tables.")
        render_chat()

load_if_needed()

# ======================
# Chat UI
# ======================
st.subheader("Chat")
render_chat()

user_prompt = st.chat_input("You're the CEO. Ask a question (e.g., 'Cluster product dimensions', 'Explain the clusters')")
if user_prompt:
    add_msg("user", user_prompt)
    render_chat()
    run_turn_ceo(user_prompt)
