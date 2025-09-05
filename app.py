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

**Granularity & defaults**
- Decide task_mode: "single" or "multi".
- If the request implies modeling (classification, regression, clustering, forecasting) or data multi-hop work, you MUST set task_mode="multi" and provide a short action_sequence of 3–5 steps using ONLY:
  overview, sql, eda, calc, feature_engineering, modeling, explain
  Typical modeling sequence: eda → feature_engineering → modeling → explain (optionally a leading sql step if joins/filters are required).
- Use task_mode="single" ONLY for simple inventory or a single SQL/EDA/Calc ask.

**Special rule — Data Inventory:** If the CEO asks variations of “what data do we have,” set next_action_type="overview" only and instruct DS to output first 5 rows for each table. No EDA/FE/modeling.

**Follow-up rule:** If the CEO’s message is a follow-up asking to explain/interpret/what features/what changed, choose explain and DO NOT re-run work; use cached results, using the central question only as context.

Use column_hints to map business terms strictly to existing columns. You can see the CEO’s current question, central question, and prior questions.

Output JSON fields:
- am_brief
- plan_for_ds
- goal  (revenue ↑ | cost ↓ | margin ↑)
- task_mode
- next_action_type
- action_sequence  # present when task_mode=multi
- action_reason
- notes_to_ceo
- need_more_info
- clarifying_questions
Return ONLY a single JSON object. The word "json" is present here to satisfy the API requirement.
"""

SYSTEM_DS = """
You are the Data Scientist (DS). Execute the AM plan using only available columns.
You can see:
- am_next_action_type OR am_action_sequence
- current_question, central_question, prior_questions (all previous CEO inputs)
Use that history to understand whether the user is giving feedback/follow-up, and keep your work anchored to the central question. If the current message is feedback, explain how you will revise to address it.

Execution modes:
- If AM provided am_action_sequence, you may return a matching action_sequence (2–5 steps). Otherwise, return a single action that MUST match am_next_action_type.
- For each step/action, follow the allowed set exactly: overview, sql, eda, calc, feature_engineering, modeling, explain.
- If a different approach is objectively better, explain briefly in ds_summary, but still conform to the AM’s action(s).

Special rule — Data Inventory: If AM indicates overview OR the CEO asked “what data do we have,” your action MUST be "overview" and your output should be previews of the first 5 rows for each table.

Support clustering by setting model_plan.task="clustering" (with optional n_clusters) when a step uses modeling.

You may ask the CEO clarifying questions if needed before executing risky or ambiguous steps.

Return JSON fields:
- ds_summary
- need_more_info
- clarifying_questions
- action  # present if single-step
- action_sequence  # present if multi-step
- duckdb_sql
- charts
- model_plan
- calc_description
- assumptions
Return ONLY a single JSON object. The word "json" is present here to satisfy the API requirement.
"""

SYSTEM_DS_REVISE = """
You are the Data Scientist (DS). Revise your prior plan/output based on AM critique and the central question.
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
- previous_question: the immediately prior CEO input
- central_question: the central question of the active thread (if any)
- prior_questions: list of all earlier CEO inputs

Decide two things:
1) intent: one of
   - new_request (starts a new central thread)
   - feedback (comment or tweak on current central thread)
   - answers_to_clarifying (answers to AM/DS's questions)
2) related_to_central: true|false — is this new input topically related to the central question?

Return ONLY a single JSON object like {"intent": "...", "related_to_central": true/false}. The word "json" is present here to satisfy the API requirement.
"""

# Column mapping assistant used by build_column_hints
SYSTEM_COLMAP = """
You map a business question to candidate table/column names. You receive:
- question: the CEO's question
- tables: {table_name: [columns...]}

Return JSON fields:
- term_to_columns: { business_term: [ {table, column}, ... ] }
- suggested_features: [ {table, column}, ... ] limited to existing columns
- notes: short string
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
if "threads" not in st.session_state: st.session_state.threads = []  # [{central, followups: []}]
if "central_question" not in st.session_state: st.session_state.central_question = ""
if "prior_questions" not in st.session_state: st.session_state.prior_questions = []

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


def classify_intent(previous_question: str, central_question: str, prior_questions: List[str], new_text: str) -> dict:
    """LLM-based intent classification with graceful fallback and relatedness."""
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
    # Fallback heuristic
    low = (new_text or "").lower()
    if any(w in low for w in ["that", "this", "looks", "seems", "instead", "also", "why", "how about", "can you"]):
        return {"intent": "feedback", "related": True}
    return {"intent": "new_request", "related": False}


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

    pipe = Pipeline([( "pre", pre), ("model", model)])
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
# Planning/Reflection/Arbitration helpers
# ======================

def _normalize_sql_text(s: str) -> str:
    if not isinstance(s, str): return ""
    return re.sub(r"\s+", " ", s).strip().lower()

def _build_sql_top_product() -> str:
    return (
        "SELECT oi.product_id, COUNT(*) AS units, SUM(oi.price) AS sales "
        "FROM olist_order_items_dataset AS oi "
        "GROUP BY 1 "
        "ORDER BY units DESC, sales DESC "
        "LIMIT 1"
    )

def _build_sql_top_customer_for_top_product() -> str:
    return """
    WITH top_prod AS (
        SELECT oi.product_id
        FROM olist_order_items_dataset AS oi
        GROUP BY 1
        ORDER BY COUNT(*) DESC, SUM(oi.price) DESC
        LIMIT 1
    )
    SELECT c.customer_unique_id AS customer, COUNT(*) AS orders, SUM(oi.price) AS sales
    FROM olist_order_items_dataset oi
    JOIN olist_orders_dataset o  ON o.order_id = oi.order_id
    JOIN olist_customers_dataset c ON c.customer_id = o.customer_id
    JOIN top_prod t ON t.product_id = oi.product_id
    GROUP BY 1
    ORDER BY sales DESC, orders DESC
    LIMIT 1
    """.strip()

def _build_sql_top_seller_for_top_product() -> str:
    return """
    WITH top_prod AS (
        SELECT oi.product_id
        FROM olist_order_items_dataset AS oi
        GROUP BY 1
        ORDER BY COUNT(*) DESC, SUM(oi.price) DESC
        LIMIT 1
    )
    SELECT s.seller_id AS seller, COUNT(*) AS orders, SUM(oi.price) AS sales
    FROM olist_order_items_dataset oi
    JOIN olist_sellers_dataset s ON s.seller_id = oi.seller_id
    JOIN top_prod t ON t.product_id = oi.product_id
    GROUP BY 1
    ORDER BY sales DESC, orders DESC
    LIMIT 1
    """.strip()

def am_sanity_gate_and_autorewrite(am_json: dict, ds_json: dict, ceo_text: str) -> dict:
    """
    AM gets a last look BEFORE rendering:
    - De-duplicate identical SQL steps in sequences.
    - If the question is 'tell me more about this product' or similar: ensure we add top customer & top seller steps for the top product.
    """
    q = (ceo_text or "").lower()
    is_more_info_about_product = (
        "more information" in q or "more info" in q or "tell me more" in q or
        "top contributor" in q or "popular seller" in q
    )

    if not isinstance(ds_json, dict) or not ds_json.get("action_sequence"):
        return ds_json

    seq = ds_json.get("action_sequence")[:5]
    new_seq = []
    seen_sql_norm = set()

    # De-dup sql steps
    for step in seq:
        a = (step.get("action") or "").lower()
        if a != "sql":
            new_seq.append(step); continue
        sql = _sql_first(step.get("duckdb_sql"))
        norm = _normalize_sql_text(sql)
        if not norm: new_seq.append(step); continue
        if norm in seen_sql_norm:
            continue
        seen_sql_norm.add(norm)
        new_seq.append(step)

    if is_more_info_about_product:
        # ensure first step finds top product
        if not new_seq or (new_seq[0].get("action") != "sql"):
            new_seq.insert(0, {"action": "sql", "duckdb_sql": _build_sql_top_product(),
                               "charts": None, "model_plan": None, "calc_description": None})
        # ensure top customer step present
        want_sql2 = _build_sql_top_customer_for_top_product()
        if all(_normalize_sql_text(_sql_first(s.get("duckdb_sql"))) != _normalize_sql_text(want_sql2)
               for s in new_seq if s.get("action") == "sql"):
            new_seq.append({"action": "sql", "duckdb_sql": want_sql2,
                            "charts": None, "model_plan": None, "calc_description": None})
        # optional: top seller
        want_sql3 = _build_sql_top_seller_for_top_product()
        if all(_normalize_sql_text(_sql_first(s.get("duckdb_sql"))) != _normalize_sql_text(want_sql3)
               for s in new_seq if s.get("action") == "sql"):
            new_seq.append({"action": "sql", "duckdb_sql": want_sql3,
                            "charts": None, "model_plan": None, "calc_description": None})

    ds_json["action_sequence"] = new_seq
    return ds_json

def enforce_multistep_for_modeling(am_json: dict, ceo_text: str) -> dict:
    """Upgrade AM to multi-step when clustering/modeling is implied but AM returned single."""
    if not isinstance(am_json, dict):
        return am_json
    q = (ceo_text or "").lower()
    implies_model = any(k in q for k in ["cluster", "clustering", "model", "kmeans", "segment", "segmentation"])
    if (am_json.get("task_mode") or "").lower() == "multi":
        return am_json
    if is_data_inventory_question(ceo_text):
        return am_json
    if implies_model:
        am_json["task_mode"] = "multi"
        seq = [
            {"action": "eda", "note": "profile candidate features"},
            {"action": "feature_engineering", "note": "numeric features for clustering"},
            {"action": "modeling", "note": "KMeans clustering (k≈5)"},
            {"action": "explain", "note": "interpret clusters & next steps"}
        ]
        am_json["action_sequence"] = seq
        am_json["next_action_type"] = "eda"
        am_json["action_reason"] = "Clustering requires EDA → FE → modeling, followed by explanation."
    return am_json

def ds_reflect_and_enhance_sequence(ds_json: dict, ceo_text: str) -> dict:
    """DS can improve AM plan: ensure the minimal pipeline for clustering and attach a default model_plan."""
    if not isinstance(ds_json, dict): return ds_json
    q = (ceo_text or "").lower()
    wants_cluster = any(k in q for k in ["cluster", "clustering", "kmeans", "segment", "segmentation"])
    if not ds_json.get("action_sequence"):
        return ds_json
    seq = ds_json["action_sequence"][:5]
    actions = [(s.get("action") or "").lower() for s in seq]
    changed = False

    def has(a): return a in actions

    if wants_cluster:
        desired = ["eda", "feature_engineering", "modeling", "explain"]
        for a in desired:
            if not has(a):
                seq.append({"action": a}); actions.append(a); changed = True
        order = {a:i for i,a in enumerate(desired)}
        seq.sort(key=lambda s: order.get((s.get("action") or ""), 99))
        for s in seq:
            if (s.get("action") or "").lower() == "modeling":
                mp = s.get("model_plan") or {}
                if not mp.get("task"): mp["task"] = "clustering"
                if not mp.get("n_clusters"): mp["n_clusters"] = 5
                s["model_plan"] = mp
                changed = True

    if changed:
        ds_json["action_sequence"] = seq
    return ds_json


# ======================
# AM/DS/Review pipeline
# ======================

def run_am_plan(prompt: str, column_hints: dict, context: dict) -> dict:
    payload = {
        "ceo_question": prompt,
        "tables": {k: list(v.columns) for k, v in (get_all_tables() or {}).items()},
        "column_hints": column_hints,
        "context": context,  # {central_question, prior_questions, current_question}
    }
    am_json = llm_json(SYSTEM_AM, json.dumps(payload))
    st.session_state.last_am_json = am_json
    add_msg("am", am_json.get("am_brief", ""), artifacts=am_json)
    render_chat()
    return am_json


def _coerce_allowed(action: str, fallback: str) -> str:
    allowed = {"overview","sql","eda","calc","feature_engineering","modeling","explain"}
    a = (action or "").lower()
    if a in allowed: return a
    synonym_map = {
        "aggregate": "sql", "aggregate_sales": "sql", "aggregation": "sql",
        "summarize": "sql", "preview": "overview", "analyze": "eda",
        "interpret": "explain", "explanation": "explain",
    }
    return synonym_map.get(a, fallback if fallback in allowed else "eda")


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

    # Normalize single vs multi
    am_mode = (am_json.get("task_mode") or ("multi" if am_json.get("action_sequence") else "single")).lower()
    if am_mode == "multi":
        seq = ds_json.get("action_sequence") or am_json.get("action_sequence") or []
        norm_seq = []
        for step in seq[:5]:
            a = _coerce_allowed((step or {}).get("action"), (am_json.get("next_action_type") or "eda").lower())
            item = {
                "action": a,
                "duckdb_sql": (step or {}).get("duckdb_sql"),
                "charts": (step or {}).get("charts"),
                "model_plan": (step or {}).get("model_plan"),
                "calc_description": (step or {}).get("calc_description"),
            }
            norm_seq.append(item)
        ds_json["action_sequence"] = norm_seq
        add_msg("ds", ds_json.get("ds_summary", ""), artifacts={"mode": "multi", "sequence": norm_seq})
    else:
        a = _coerce_allowed(ds_json.get("action"), (am_json.get("next_action_type") or "eda").lower())
        ds_json["action"] = (am_json.get("next_action_type") or a)
        add_msg("ds", ds_json.get("ds_summary", ""), artifacts={"action": ds_json.get("action"),
                       "duckdb_sql": ds_json.get("duckdb_sql"),
                       "model_plan": ds_json.get("model_plan")})

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
# Build meta for AM review (supports sequence)
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
        for sql in [_sql_first(s) for s in sql_list if s ]:
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
        plan = ds_step.get("model_plan") or {}
        target = plan.get("target")
        try:
            base = run_duckdb_sql(sql) if sql else None
            if base is None:
                # search both FE and RAW
                for _, df in get_all_tables().items():
                    if (plan.get("task") == "clustering") or (target and target in df.columns):
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
        # snapshot FE table for downstream reference
        st.session_state.tables_fe["feature_base"] = base
        add_msg("ds","Feature base ready (saved as 'feature_base').")
        return

    # ---- MODELING ----
    if action == "modeling":
        sql = _sql_first(ds_step.get("duckdb_sql"))
        plan = ds_step.get("model_plan") or {}
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
            add_msg("ds","Clustering completed.", artifacts={"report": rep})
            return
        else:
            report = train_model(base, task, target, plan.get("features") or [], (plan.get("model_family") or "logistic_regression").lower())
            st.markdown("### 🤖 Model Report")
            st.json(report)
            add_msg("ds","Model trained.", artifacts={"model_report": report})
            return

    if action == "explain":
        st.markdown("### 📝 Interpretation / Explanation")
        st.write("Explanation step: using results of previous steps to answer the CEO’s follow-up without re-running.")
        add_msg("ds","Explanation provided.")
        return

    add_msg("ds", f"Action '{action}' not recognized.", artifacts=ds_step)


# ======================
# Coordinator (threading + follow-ups)
# ======================

def run_turn_ceo(new_text: str):
    prev = st.session_state.current_question or ""
    central = st.session_state.central_question or ""
    prior = st.session_state.prior_questions or []

    ic = classify_intent(prev, central, prior, new_text)
    intent = ic.get("intent", "new_request")
    related = ic.get("related", False)

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

    # 0) Column hints from RAW + FE
    col_hints = build_column_hints(effective_q)

    # Context pack for AM/DS
    thread_ctx = {
        "central_question": st.session_state.central_question,
        "current_question": st.session_state.current_question,
        "prior_questions": st.session_state.prior_questions,
    }

    # 1) AM plan (+ enforce multi-step for modeling asks)
    am_json = run_am_plan(effective_q, col_hints, context=thread_ctx)
    am_json = enforce_multistep_for_modeling(am_json, new_text)

    # If AM needs user info, ask and stop until the CEO replies
    if am_json.get("need_more_info"):
        qs = am_json.get("clarifying_questions") or ["Could you clarify your objective?"]
        add_msg("am", "I need a bit more context:")
        for q in qs[:3]:
            add_msg("am", f"• {q}")
        render_chat()
        return

    # 2) DS executes (single or multi)
    max_loops = 3
    loop_count = 0
    ds_json = run_ds_step(am_json, col_hints, thread_ctx)

    # DS reflection (improve sequence for clustering if needed)
    ds_json = ds_reflect_and_enhance_sequence(ds_json, new_text)
    # AM sanity gate (pre-render) – de-dup & synthesize
    ds_json = am_sanity_gate_and_autorewrite(am_json, ds_json, new_text)

    # If DS asks for clarification explicitly
    if ds_json.get("need_more_info") and (ds_json.get("clarifying_questions") or []):
        add_msg("ds", "Before running further steps, I need:")
        for q in (ds_json.get("clarifying_questions") or [])[:3]:
            add_msg("ds", f"• {q}")
        render_chat()
        return

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

        # Light duplicate SQL hint to AM reviewer
        if isinstance(meta, list):
            seen = set(); dup = False
            for m in meta:
                if m.get("type") == "sql":
                    n = _normalize_sql_text(m.get("sql",""))
                    if n in seen: dup = True; break
                    seen.add(n)
            if dup:
                meta = {"note": "dup_sql_detected", "steps": meta}

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

        # If AM needs clarification from CEO, ask and pause the run
        if review.get("clarification_needed") and (review.get("clarifying_questions") or []):
            add_msg("am", "Before proceeding, could you clarify:")
            for q in (review.get("clarifying_questions") or [])[:3]:
                add_msg("am", f"• {q}")
            render_chat()
            return

        # If sufficient and no revision required → render final and exit
        if review.get("sufficient_to_answer") and not review.get("must_revise"):
            # final pre-render sanity gate
            ds_json = am_sanity_gate_and_autorewrite(am_json, ds_json, new_text)
            _render(ds_json)
            return

        # Otherwise revise if requested
        if review.get("must_revise"):
            ds_json = revise_ds(am_json, ds_json, review, col_hints, thread_ctx)
            # ensure AM remains multi-step for modeling asks
            am_json = enforce_multistep_for_modeling(st.session_state.last_am_json, new_text)
            # DS reflection and sanity gate again
            ds_json = ds_reflect_and_enhance_sequence(ds_json, new_text)
            ds_json = am_sanity_gate_and_autorewrite(am_json, ds_json, new_text)
            add_msg("ds", ds_json.get("ds_summary","(revised)"),
                    artifacts={"mode": "multi" if ds_json.get("action_sequence") else "single"})
            render_chat()
            continue  # loop for another review

        # Fallback: if neither sufficient nor must_revise, render to avoid deadlock
        ds_json = am_sanity_gate_and_autorewrite(am_json, ds_json, new_text)
        _render(ds_json)
        return

    # If loop exhausted, present best effort after last sanity pass
    add_msg("system", "Reached review limit; presenting current best effort with noted caveats.")
    ds_json = am_sanity_gate_and_autorewrite(am_json, ds_json, new_text)
    _render(ds_json)
    return


# ======================
# UI wiring
# ======================

# Load tables on upload
if zip_file is not None and st.session_state.tables_raw is None:
    try:
        st.session_state.tables_raw = load_zip_tables(zip_file)
        st.session_state.tables = get_all_tables()
        add_msg("system", f"Loaded {len(st.session_state.tables_raw)} table(s) from ZIP.")
        render_chat()
    except Exception as e:
        st.error(f"Failed to load ZIP: {e}")

# Conversation context display
st.header("💬 Conversation Context")
st.subheader("Central Question")
st.write(st.session_state.central_question or "(none)")
if st.session_state.prior_questions:
    st.subheader("Prior Questions")
    st.write(st.session_state.prior_questions)

# Chat input
user_payload = st.chat_input("Ask the CEO ↔ AM ↔ DS assistant anything...")
if user_payload:
    add_msg("user", user_payload)
    render_chat()
    run_turn_ceo(user_payload)
