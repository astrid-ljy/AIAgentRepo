# app2.py — CEO ↔ AM ↔ DS — Profit Assistant (multi-step, threaded)
# Updated to add a reviews NLP pipeline (Spanish-aware), caching, DS schema validation,
# stratified sampling, config externalization, and an internal AM/DS action:
# "aggregate_reviews_from_text" (stays in AM/DS loop; no hidden bypass).

import os
import io
import re
import json
import zipfile
import hashlib
from typing import Dict, Any, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
import duckdb
import streamlit as st

# ---- Session-state defaults guard ----
def _ensure_state_defaults():
    ss = st.session_state
    # core chat/threading
    if "chat" not in ss: ss["chat"] = []
    if "threads" not in ss: ss["threads"] = []
    if "active_thread_idx" not in ss: ss["active_thread_idx"] = 0
    if "last_rendered_idx" not in ss: ss["last_rendered_idx"] = 0
    # user question tracking
    if "current_question" not in ss: ss["current_question"] = ""
    if "user_question_history" not in ss: ss["user_question_history"] = []
    # data tables
    if "tables_raw" not in ss: ss["tables_raw"] = {}
    if "tables" not in ss: ss["tables"] = {}
    if "tables_fe" not in ss: ss["tables_fe"] = {}
    # modeling cache
    if "last_model_result" not in ss: ss["last_model_result"] = None

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

# ---------- Optional deps for language detection ----------
try:
    import langid  # robust offline language identification
    _LANGID_AVAILABLE = True
except Exception:
    _LANGID_AVAILABLE = False

# ---------- OpenAI setup ----------
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
DEFAULT_MODEL  = st.secrets.get("OPENAI_MODEL",  os.getenv("OPENAI_MODEL",  "gpt-4o-mini"))
try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except Exception:
    _OPENAI_AVAILABLE = False


# ======================
# Configuration (externalizable)
# ======================
DEFAULT_CONFIG = {
    "allowed_actions": [
        "overview","sql","eda","calc","feature_engineering","modeling",
        # Internal named action (keeps AM/DS contract; not a hidden bypass)
        "aggregate_reviews_from_text"
    ],
    "pipeline_version": "nlp_v3.1",
    "review_keywords_es": [
        "servicio","envío","entrega","calidad","devolución",
        "precio","tiempo","atención","producto","pedido"
    ],
    "sentiment_strategies": ["openai_sentiment_v1","rule_based_v1"],
    "default_strategy": "openai_sentiment_v1",
    "batch_size": 64,
    "max_rows": 5000,
    "review_text_min_len": 20,
    "positive_threshold": 0.7,
    "dangerous_sql_patterns": ["\\bdrop\\b","\\battach\\b","\\bcopy\\s+to\\b"],
    "pii_regex": {
        "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        "url": r"https?://[^\s]+",
        "phone_intl": r"(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{2,4}\)?[\s-]?)?\d{3,4}[\s-]?\d{4}",
        "order_id": r"\b(?:ord|pedido|order)[-_]?\d{5,}\b"
    },
    "review_product_id_candidates": ["product_id","asin","sku","id_producto","id_producto_sku"],
    "translate_modes": ["Always","Auto","Never"]
}

# Sidebar config overrides (YAML or text)
with st.sidebar:
    st.header("🔧 Config")
    cfg_text = st.text_area("Config (YAML or JSON, optional)", height=160, value="")
    cfg_file = st.file_uploader("Upload config.yml/json (optional)", type=["yml","yaml","json"])

def _merge_config(base: Dict[str,Any], override: Optional[Dict[str,Any]]) -> Dict[str,Any]:
    if not override: return base
    merged = dict(base)
    for k,v in override.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = _merge_config(merged[k], v)
        else:
            merged[k] = v
    return merged

def _parse_cfg_text(s: str) -> Optional[Dict[str,Any]]:
    if not s: return None
    try:
        import yaml
        return yaml.safe_load(s)
    except Exception:
        try:
            return json.loads(s)
        except Exception:
            return None

cfg_loaded = None
if cfg_file is not None:
    try:
        txt = cfg_file.read().decode("utf-8")
        cfg_loaded = _parse_cfg_text(txt)
    except Exception:
        cfg_loaded = None

cfg_text_parsed = _parse_cfg_text(cfg_text)
CONFIG = _merge_config(DEFAULT_CONFIG, cfg_loaded or cfg_text_parsed)

PIPELINE_VERSION = CONFIG.get("pipeline_version","nlp_v3.1")
ALLOWED_ACTIONS  = set(CONFIG.get("allowed_actions", []))


# ======================
# System Prompts (updated; includes internal action)
# ======================
SYSTEM_AM = f"""
You are the Analytics Manager (AM). Plan how to answer the CEO’s business question using the available data.
You can see the user's current question, all prior questions in this conversation, and lightweight column hints.

**Task complexity decision**: Decide if this is a single-action task or a multi-action pipeline.
- Allowed atomic actions: {sorted(ALLOWED_ACTIONS)}

- For multi-action, produce a short ordered pipeline of 2–5 actions (from the same allowed list).

**Special rule — Data Inventory:** If the CEO asks variations of “what data do we have,” set actions_pipeline=["overview"] only and instruct DS to output **first 5 rows for each table**. No EDA/FE/modeling.

Use the provided `column_hints` to resolve business terms (e.g., revenue → sales) strictly to existing columns.

Output JSON fields:
- am_brief: 1–2 sentences paraphrasing the question for a profit-oriented plan (do not repeat verbatim).
- plan_for_ds: concrete steps referencing ONLY existing tables/columns.
- goal: profit proxy to improve (revenue ↑, cost ↓, margin ↑).
- task_complexity: "single" | "multi"
- actions_pipeline: [{",".join(sorted(ALLOWED_ACTIONS))}]
- action_reason: short rationale of why this pipeline is appropriate
- notes_to_ceo: 1–2 short notes
- need_more_info: true|false
- clarifying_questions: []
Return ONLY a single JSON object. This line contains the word json.
"""

SYSTEM_DS = f"""
You are the Data Scientist (DS). Execute the AM plan using only available columns.
You can see: (1) AM's action for this step, (2) the user's current question, (3) ALL prior user questions in this conversation, and (4) column_hints.

**Obey action:** Your `action` MUST match `am_next_action_type` exactly. Allowed values: {sorted(ALLOWED_ACTIONS)}.
If a different action would be better, briefly note it in ds_summary but STILL set `action` to `am_next_action_type`.

**Ask for clarification when needed:** If the step cannot proceed without missing definitions or selections, set `needs_clarification=true` and include up to 3 short questions in `clarifying_questions`.

**Special rule — Data Inventory:** If AM indicates overview OR the CEO asked “what data do we have,” your action MUST be "overview" and your output should be previews of the **first 5 rows for each table**.

Support clustering by setting model_plan.task="clustering" (with optional n_clusters) when `action="modeling"`.

Support reviews pipeline when `action="aggregate_reviews_from_text"` by summarizing per product:
- stratified sampling per product with a cap
- language detection/translation (per config + user UI)
- sentiment probabilities (pos/neu/neg)
- compute positive_share = mean(p_pos) and a table by product

Return JSON fields:
- ds_summary: brief note of intended execution
- action: {"|".join(sorted(ALLOWED_ACTIONS))}  # MUST equal am_next_action_type
- duckdb_sql: SQL string OR list of SQL strings (for eda or feature assembly); not required for reviews action
- charts: optional chart specs
- model_plan: {{task: classification|regression|clustering|null, target: str|null, features: [], model_family: logistic_regression|random_forest|linear_regression|random_forest_regressor|null, n_clusters: int|null}}
- calc_description: string
- assumptions: string
- needs_clarification: boolean
- clarifying_questions: []
Return ONLY a single JSON object. This line contains the word json.
"""

SYSTEM_DS_REVISE = """
You are the Data Scientist (DS). Revise your prior plan/output based on AM critique.
- Keep to ONLY existing tables/columns.
- Fix suitability issues AM mentioned.
- Keep it concise and executable.
Return JSON with the SAME schema you use normally:
{ "ds_summary": "...", "action": "...", "duckdb_sql": "... or [...]", "charts": [...], "model_plan": {...}, "calc_description": "...", "assumptions": "...", "needs_clarification": false, "clarifying_questions": [] }
Return ONLY a single JSON object. This line contains the word json.
"""

SYSTEM_AM_REVIEW = """
You are the AM Reviewer. Given CEO question, AM plan (including actions_pipeline), DS action, and lightweight result meta (shapes/samples/metrics),
write a short plain-language summary for the CEO and critique suitability.
Decide explicitly if the current DS action sufficiently answers the CEO's question.
Return JSON fields:
- summary_for_ceo: 2–4 sentences in plain language
- appropriateness_check: brief assessment of method/query suitability
- gaps_or_risks: brief note on assumptions/data issues
- improvements: [1–4 concrete improvements]
- suggested_next_steps: [1–4 next actions]
- must_revise: boolean
- sufficient_to_answer: boolean
- clarification_needed: boolean
- clarifying_questions: [str]
- revision_notes: string
Return ONLY a single JSON object. This line contains the word json.
"""

SYSTEM_INTENT = """
Classify CEO input relative to the `previous_question` as:
- new_request
- feedback
- answers_to_clarifying
Return ONLY a single JSON object with {"intent": "..."}. This line contains the word json.
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
Return ONLY a single JSON object. This line contains the word json.
"""

# ======================
# Streamlit page
# ======================
st.set_page_config(page_title="CEO ↔ AM ↔ DS", layout="wide")
st.title("🏢 CEO ↔ AM ↔ DS — Profit Assistant (multi-step, threaded)")

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
if "threads" not in st.session_state: st.session_state.threads = []  # [{central: str, followups: [str]}]
if "active_thread_idx" not in st.session_state: st.session_state.active_thread_idx = None
if "user_question_history" not in st.session_state: st.session_state.user_question_history = []
if "cache_stats" not in st.session_state: st.session_state.cache_stats = {"det": [0,0], "tr": [0,0], "sent": [0,0]}  # [hits,total]
if "dataset_hash" not in st.session_state: st.session_state.dataset_hash = ""
if "review_ui_state" not in st.session_state:
    st.session_state.review_ui_state = {
        "selected_table": None,
        "selected_text_col": None,
        "selected_product_col": None
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
    client = ensure_openai()
    sys_msg = system_prompt.strip() + "\n\nReturn ONLY a single JSON object. This line contains the word json."
    user_msg = (user_payload or "").strip() + "\n\nPlease respond with JSON only (a single object)."
    try:
        resp = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[{"role": "system","content": sys_msg},{"role": "user","content": user_msg}],
            temperature=0.0,
        )
        return json.loads(resp.choices[0].message.content or "{}")
    except Exception as e1:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role":"system","content": sys_msg},
                          {"role":"user","content": user_msg + "\n\nIf needed, wrap JSON in ```json fences."}],
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

def _rewrite_sql_if_missing_product_join(sql: str) -> str:
    """
    If a query selects product_id directly from olist_order_reviews_dataset,
    transparently JOIN to olist_order_items_dataset on order_id.
    Only triggers when product_id appears and items table is not already present.
    """
    try:
        import re as _re
    except Exception:
        return sql
    if not isinstance(sql, str) or not sql:
        return sql
    has_reviews = _re.search(r"\bolist_order_reviews_dataset\b", sql, flags=_re.IGNORECASE)
    has_items   = _re.search(r"\bolist_order_items_dataset\b", sql, flags=_re.IGNORECASE)
    needs_prod  = _re.search(r"\bproduct_id\b", sql, flags=_re.IGNORECASE)
    if has_reviews and needs_prod and not has_items:
        # Attach a JOIN clause to the first FROM ...reviews...
        sql = _re.sub(
            r"FROM\s+olist_order_reviews_dataset(?:\s+\w+)?",
            "FROM olist_order_reviews_dataset r JOIN olist_order_items_dataset i ON r.order_id = i.order_id",
            sql,
            flags=_re.IGNORECASE
        )
        # Qualify ambiguous columns
        sql = _re.sub(r"\bproduct_id\b", "i.product_id", sql, flags=_re.IGNORECASE)
        sql = _re.sub(r"\breview_score\b", "r.review_score", sql, flags=_re.IGNORECASE)
        sql = _re.sub(r"\breview_id\b", "r.review_id", sql, flags=_re.IGNORECASE)
        sql = _re.sub(r"\border_id\b", "r.order_id", sql, flags=_re.IGNORECASE)
    return sql


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

def is_data_inventory_question(text: str) -> bool:
    q = (text or "").lower()
    triggers = [
        "what data do we have", "what data do i have", "what data do we have here",
        "what tables do we have", "what datasets", "first 5 rows", "first five rows",
        "preview tables", "show tables", "data inventory"
    ]
    return any(t in q for t in triggers)

def classify_intent(previous_question: str, new_text: str) -> str:
    try:
        payload = {"previous_question": previous_question or "", "new_text": new_text}
        res = llm_json(SYSTEM_INTENT, json.dumps(payload))
        intent = (res or {}).get("intent")
        if intent in {"new_request", "feedback", "answers_to_clarifying"}:
            return intent
    except Exception:
        pass
    low = (new_text or "").lower()
    if any(w in low for w in ["that", "this", "looks", "seems", "instead", "can you also", "why", "how about", "follow up", "follow-up"]):
        return "feedback"
    return "new_request"

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
# Reviews NLP Utilities (Spanish-aware)
# ======================

EMOJI_REGEX = re.compile(
    r"[\U0001F300-\U0001F6FF\U0001F900-\U0001FAFF\U00002700-\U000027BF\U0001F1E6-\U0001F1FF]+"
)

def _hash_str(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()

def dataset_hash(tables: Dict[str,pd.DataFrame]) -> str:
    h = hashlib.sha256()
    for name in sorted(tables.keys()):
        h.update(name.encode("utf-8"))
        # only shapes + column names for a quick, stable hash
        df = tables[name]
        h.update(f"{len(df)}|{','.join(df.columns)}".encode("utf-8"))
    return h.hexdigest()[:16]

@st.cache_data(show_spinner=False)
def detect_lang_cached(texts: Tuple[str,...], version: str) -> List[str]:
    # Cache metric
    st.session_state.cache_stats["det"][0] += 1
    st.session_state.cache_stats["det"][1] += 1
    out = []
    for t in texts:
        if _LANGID_AVAILABLE:
            try:
                out.append(langid.classify(t or "")[0])
                continue
            except Exception:
                pass
        # fallback heuristic: presence of common Spanish function words / characters
        low = (t or "").lower()
        score = 0
        for w in [" el "," la "," de "," y "," que "," en "," un "," una "," para "," con "]:
            if w in f" {low} ":
                score += 1
        if "ñ" in low or "á" in low or "é" in low or "í" in low or "ó" in low or "ú" in low:
            score += 1
        out.append("es" if score >= 2 else "en")
    return out

def _pii_scrub(text: str, aggressive: bool) -> str:
    if not text: return text
    pat = CONFIG["pii_regex"]
    red = text
    red = re.sub(pat["email"], "[email]", red, flags=re.IGNORECASE)
    red = re.sub(pat["url"], "[url]", red, flags=re.IGNORECASE)
    red = re.sub(pat["phone_intl"], "[phone]", red, flags=re.IGNORECASE)
    red = re.sub(pat["order_id"], "[order_id]", red, flags=re.IGNORECASE)
    if aggressive:
        # strip digits bursts longer than 6
        red = re.sub(r"\d{6,}", "[num]", red)
    return red

@st.cache_data(show_spinner=False)
def translate_cached(
    items: Tuple[Tuple[str,str],...],  # (lang, text)
    model_name: str,
    pipeline_ver: str
) -> List[str]:
    # items is a tuple of (lang, text) for stable caching key
    st.session_state.cache_stats["tr"][0] += 1
    st.session_state.cache_stats["tr"][1] += 1
    client = ensure_openai()
    # Simple batching
    texts = [t for (_,t) in items]
    chunks = []
    B = 20
    for i in range(0, len(texts), B):
        batch = texts[i:i+B]
        prompt = "Translate each line to **English**. Return JSON list of strings. JSON only."
        content = "\n".join([f"{j+1}. {x}" for j,x in enumerate(batch)])
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role":"system","content":"You are a precise translator."},
                      {"role":"user","content": prompt+"\n\n"+content}],
            temperature=0.0,
            response_format={"type":"json_object"}
        )
        try:
            data = json.loads(resp.choices[0].message.content or "{}")
            out = data.get("translations") or data.get("data") or data.get("list") or None
            if isinstance(out, list):
                chunks.extend([str(x) for x in out])
            else:
                # fallback: naive split by lines (kept robust)
                chunks.extend(batch)
        except Exception:
            chunks.extend(batch)
    return chunks

def _estimate_tokens(chars: int) -> int:
    # very rough heuristic ~4 chars per token
    return max(1, chars // 4)

@st.cache_data(show_spinner=False)
def sentiment_openai_cached(
    texts: Tuple[str,...],
    model_name: str,
    pipeline_ver: str
) -> List[Dict[str, Any]]:
    st.session_state.cache_stats["sent"][0] += 1
    st.session_state.cache_stats["sent"][1] += 1
    client = ensure_openai()
    out: List[Dict[str,Any]] = []
    B = CONFIG.get("batch_size", 64)
    sys = (
        "You output JSON only. For each line, return:\n"
        "{ 'probs': {'pos':FLOAT,'neu':FLOAT,'neg':FLOAT}, 'keywords': [up to 5 key phrases in English] }.\n"
        "Probabilities must sum ~1.0."
    )
    for i in range(0, len(texts), B):
        batch = texts[i:i+B]
        user = "Classify the sentiment probabilities for each line and extract up to five key phrases.\n" + \
               "\n".join([f"{j+1}. {x}" for j,x in enumerate(batch)])
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role":"system","content": sys},
                      {"role":"user","content": user}],
            temperature=0.0,
            response_format={"type":"json_object"}
        )
        try:
            data = json.loads(resp.choices[0].message.content or "{}")
            arr = data.get("items") or data.get("results") or data.get("list") or []
            if isinstance(arr, list) and len(arr) == len(batch):
                out.extend(arr)
            else:
                # fallback: neutral baseline
                out.extend([{"probs":{"pos":0.33,"neu":0.34,"neg":0.33},"keywords":[]} for _ in batch])
        except Exception:
            out.extend([{"probs":{"pos":0.33,"neu":0.34,"neg":0.33},"keywords":[]} for _ in batch])
    return out

def sentiment_rule_based(texts: List[str]) -> List[Dict[str,Any]]:
    # Very simple lexicon for Spanish + neutral fallback
    pos_words = {"bueno","excelente","rápido","perfecto","encantado","recomendado","satisfecho","maravilloso","genial","agradable"}
    neg_words = {"malo","terrible","lento","defectuoso","dañado","horrible","fatal","pésimo","decepcionado","tarde"}
    out=[]
    for t in texts:
        low=(t or "").lower()
        p = sum(w in low for w in pos_words)
        n = sum(w in low for w in neg_words)
        total = p+n
        if total==0:
            out.append({"probs":{"pos":0.34,"neu":0.33,"neg":0.33},"keywords":[]})
        else:
            pos = p/total
            neg = n/total
            neu = max(0.0, 1.0 - (pos+neg))
            out.append({"probs":{"pos":float(pos), "neu":float(neu), "neg":float(neg)}, "keywords":[]})
    return out

def _spanish_keyword_score(s: str) -> int:
    low = (s or "").lower()
    score=0
    for kw in CONFIG.get("review_keywords_es", []):
        if kw in low:
            score += 1
    if EMOJI_REGEX.search(s or ""):
        score += 1
    return score

def _rank_text_columns(tables: Dict[str,pd.DataFrame]) -> List[Dict[str,Any]]:
    ranking=[]
    for tname, df in tables.items():
        for col in df.columns:
            s = df[col]
            if not pd.api.types.is_object_dtype(s) and not pd.api.types.is_string_dtype(s):
                continue
            nonnull = s.dropna().astype(str)
            if len(nonnull) == 0: continue
            sample = nonnull.sample(min(len(nonnull), 500), random_state=42)
            mean_len = sample.str.len().mean()
            kw_hits = sample.apply(_spanish_keyword_score).mean()
            emoji_hits = sample.apply(lambda x: 1 if EMOJI_REGEX.search(x) else 0).mean()
            score = kw_hits*2.0 + emoji_hits*1.0 + (1.0 if mean_len >= CONFIG["review_text_min_len"] else 0.0)
            ranking.append({
                "table": tname, "column": col, "score": float(score),
                "mean_len": float(mean_len), "kw_hits": float(kw_hits), "emoji_hits": float(emoji_hits)
            })
    ranking.sort(key=lambda x: x["score"], reverse=True)
    return ranking

def _find_product_col(df: pd.DataFrame) -> Optional[str]:
    cands = CONFIG.get("review_product_id_candidates", [])
    low_map = {c.lower(): c for c in df.columns}
    for c in cands:
        if c.lower() in low_map:
            return low_map[c.lower()]
    # fallback by heuristic
    for c in df.columns:
        if re.search(r"(product|sku|asin|item)", c, flags=re.IGNORECASE):
            return c
    return None

def _stratified_sample(df: pd.DataFrame, group_col: str, cap: int) -> pd.DataFrame:
    parts=[]
    for gid, g in df.groupby(group_col):
        parts.append(g.sample(min(len(g), cap), random_state=42))
    return pd.concat(parts, ignore_index=True) if parts else df.head(0)

# ======================
# Modeling (incl. clustering)
# ======================

def train_model(df: pd.DataFrame, task: str, target: Optional[str], features: List[str], family: str, n_clusters: Optional[int]=None) -> Dict[str, Any]:
    report: Dict[str, Any] = {}

    
    if task == "clustering":
        use_cols = []
        if features:
            use_cols = [c for c in features if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
        else:
            use_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not use_cols:
            return {"error": f"No usable numeric features for clustering. Requested: {features or 'None'}, "
                             f"available numeric: {[c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]}"}
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
# AM/DS/Review pipeline (with internal reviews action)
# ======================

def run_am_plan(prompt: str, column_hints: dict, user_history: List[str]) -> dict:
    payload = {
        "ceo_question": prompt,
        "tables": {k: list(v.columns) for k, v in (get_all_tables() or {}).items()},
        "column_hints": column_hints,
        "user_question_history": user_history,
        "pipeline_version": PIPELINE_VERSION
    }
    am_json = llm_json(SYSTEM_AM, json.dumps(payload))
    st.session_state.last_am_json = am_json
    add_msg("am", am_json.get("am_brief", ""), artifacts=am_json)
    render_chat()
    return am_json

# Minimal schema validation & SQL sanitize
def validate_ds_json(ds_json: dict) -> dict:
    allowed_keys = {
        "ds_summary","action","duckdb_sql","charts","model_plan","calc_description",
        "assumptions","needs_clarification","clarifying_questions","_awaiting_clarification"
    }
    cleaned = {k:v for k,v in ds_json.items() if k in allowed_keys}
    action = (cleaned.get("action") or "").lower()
    if action not in ALLOWED_ACTIONS:
        cleaned["action"] = "eda"  # safe fallback

    # sanitize SQL
    sql = cleaned.get("duckdb_sql")
    pats = CONFIG.get("dangerous_sql_patterns", [])
    def _bad(s: str) -> bool:
        return any(re.search(p, s, flags=re.IGNORECASE) for p in pats)

    if isinstance(sql, str):
        if _bad(sql): cleaned["duckdb_sql"] = ""
    elif isinstance(sql, list):
        cleaned["duckdb_sql"] = [s for s in sql if isinstance(s,str) and s and not _bad(s)]
    return cleaned

def run_ds_step_single(am_next_action: str, column_hints: dict, current_q: str, user_history: List[str], am_plan_text: str) -> dict:
    ds_payload = {
        "am_plan": am_plan_text,
        "am_next_action_type": am_next_action,
        "tables": {k: list(v.columns) for k, v in (get_all_tables() or {}).items()},
        "column_hints": column_hints,
        "current_user_question": current_q,
        "all_user_questions": user_history,
        "pipeline_version": PIPELINE_VERSION
    }
    ds_raw = llm_json(SYSTEM_DS, json.dumps(ds_payload))
    ds_json = validate_ds_json(ds_raw)
    st.session_state.last_ds_json = ds_json

    if ds_json.get("needs_clarification") and (ds_json.get("clarifying_questions") or []):
        add_msg("ds", "I need clarification before executing this step:")
        for q in (ds_json.get("clarifying_questions") or [])[:3]:
            add_msg("ds", f"• {q}")
        render_chat()
        return {"_awaiting_clarification": True, **ds_json}

    add_msg("ds", ds_json.get("ds_summary", ""),
            artifacts={"action": ds_json.get("action"),
                       "duckdb_sql": ds_json.get("duckdb_sql"),
                       "model_plan": ds_json.get("model_plan")})
    render_chat()

    render_final(ds_json)  # immediate render
    return ds_json

def run_ds_pipeline(am_json: dict, column_hints: dict, current_q: str, user_history: List[str]) -> List[dict]:
    actions = am_json.get("actions_pipeline") or []
    if not actions:
        actions = [am_json.get("next_action_type", "eda")]
    executed: List[dict] = []
    for step_idx, act in enumerate(actions[:5]):
        ds_json = run_ds_step_single(act, column_hints, current_q, user_history, am_json.get("plan_for_ds", ""))
        executed.append(ds_json)
        if ds_json.get("_awaiting_clarification"):
            break
    return executed

def am_review(ceo_prompt: str, ds_json: dict, meta: dict) -> dict:
    bundle = {"ceo_question": ceo_prompt,
              "am_plan": st.session_state.last_am_json,
              "ds_json": ds_json,
              "meta": meta}
    return llm_json(SYSTEM_AM_REVIEW, json.dumps(bundle))

def revise_ds(am_json: dict, prev_ds_json: dict, review_json: dict, column_hints: dict, current_q: str, user_history: List[str]) -> dict:
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
        "current_user_question": current_q,
        "all_user_questions": user_history,
        "pipeline_version": PIPELINE_VERSION
    }
    return llm_json(SYSTEM_DS_REVISE, json.dumps(payload))

# ======================
# Build meta for AM review
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
                df = run_duckdb_sql(_rewrite_sql_if_missing_product_join(sql))
                metas.append({"sql": sql, "rows": len(df), "cols": list(df.columns),
                              "sample": df.head(10).to_dict(orient="records")})
            except Exception as e:
                metas.append({"sql": sql, "error": str(e)})
        return {"type": "eda", "results": metas}

    if action == "sql":
        sql = _sql_first(ds_json.get("duckdb_sql"))
        if not sql: return {"type": "sql", "error": "No SQL provided"}
        try:
            out = run_duckdb_sql(_rewrite_sql_if_missing_product_join(sql))
            return {"type": "sql", "sql": sql, "rows": len(out), "cols": list(out.columns),
                    "sample": out.head(10).to_dict(orient="records")}
        except Exception as e:
            return {"type": "sql", "sql": sql, "error": str(e)}

    if action == "calc":
        return {"type": "calc", "desc": ds_json.get("calc_description", "")}

    if action == "feature_engineering":
        sql = _sql_first(ds_json.get("duckdb_sql"))
        try:
            base = run_duckdb_sql(_rewrite_sql_if_missing_product_join(sql)) if sql else next(iter(get_all_tables().values())).copy()
            return {"type": "feature_engineering", "rows": len(base), "cols": list(base.columns),
                    "sample": base.head(10).to_dict(orient="records")}
        except Exception as e:
            return {"type": "feature_engineering", "error": str(e)}

    if action == "modeling":
        sql = _sql_first(ds_json.get("duckdb_sql"))
        plan = ds_json.get("model_plan") or {}
        target = plan.get("target")
        try:
            base = run_duckdb_sql(_rewrite_sql_if_missing_product_join(sql)) if sql else None
            if base is None:
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

    if action == "aggregate_reviews_from_text":
        return {"type": "aggregate_reviews_from_text"}

    return {"type": action or "unknown", "note": "no meta builder"}

# ======================
# Reviews pipeline renderer
# ======================

def _reviews_controls_and_candidates(tables: Dict[str,pd.DataFrame]) -> Tuple[Optional[str], Optional[str], Optional[str], Dict[str,Any]]:
    with st.sidebar:
        st.header("📝 Reviews (Spanish)")
        translate_mode = st.selectbox("Translate mode", CONFIG.get("translate_modes", ["Always","Auto","Never"]), index=1)
        sentiment_strategy = st.selectbox("Sentiment strategy", CONFIG.get("sentiment_strategies", []),
                                          index=max(0, CONFIG.get("sentiment_strategies", []).index(CONFIG.get("default_strategy","openai_sentiment_v1")) if CONFIG.get("default_strategy") in CONFIG.get("sentiment_strategies", []) else 0))
        per_product_cap = st.number_input("Reviews cap per product (stratified)", min_value=10, max_value=1000, value=100, step=10)
        pos_threshold = st.slider("Positive threshold (for labeling only)", min_value=0.5, max_value=0.95, value=float(CONFIG.get("positive_threshold",0.7)), step=0.05)
        batch_size = st.number_input("Batch size (LLM)", min_value=8, max_value=256, value=int(CONFIG.get("batch_size",64)), step=8)
        aggressive_scrub = st.checkbox("Aggressive PII scrub", value=False)

    # Suggest text columns ranked by Spanish keyphrases/emoji/length
    ranking = _rank_text_columns(tables)
    if ranking:
        top = ranking[0]
    else:
        top = {"table": None, "column": None}

    st.markdown("#### Suggested review text column (Spanish-aware)")
    top_df = pd.DataFrame(ranking[:10])
    if not top_df.empty:
        st.dataframe(top_df, use_container_width=True)

    # Sidebar overrides that persist per dataset hash
    ds_hash = dataset_hash(tables)
    st.session_state.dataset_hash = ds_hash
    ui_state = st.session_state.review_ui_state

    with st.sidebar:
        # Table/column selectors
        table_names = ["(auto)"] + list(tables.keys())
        sel_table = st.selectbox("Table with reviews", table_names, index=0)
        if sel_table == "(auto)":
            sel_table_resolved = top["table"]
        else:
            sel_table_resolved = sel_table

        text_cols = ["(auto)"]
        product_cols = ["(auto)"]
        if sel_table_resolved and sel_table_resolved in tables:
            df = tables[sel_table_resolved]
            text_cols += [c for c in df.columns if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c])]
            product_cols += list(df.columns)
        sel_text = st.selectbox("Text column", text_cols, index=0)
        sel_prod = st.selectbox("Product column", product_cols, index=0)

    text_col_resolved = None
    prod_col_resolved = None
    if sel_table_resolved:
        df = tables[sel_table_resolved]
        if sel_text == "(auto)":
            text_col_resolved = top["column"] if top["table"] == sel_table_resolved else None
            if text_col_resolved is None:
                # choose longest average length object column
                obj_cols = [c for c in df.columns if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c])]
                if obj_cols:
                    lens = df[obj_cols].astype(str).apply(lambda s: s.str.len().mean()).sort_values(ascending=False)
                    text_col_resolved = lens.index[0]
        else:
            text_col_resolved = sel_text

        if sel_prod == "(auto)":
            prod_col_resolved = _find_product_col(df)
        else:
            prod_col_resolved = sel_prod

    # persist per dataset hash
    ui_state_key = f"{ds_hash}"
    st.session_state.review_ui_state = {
        "selected_table": sel_table_resolved or ui_state.get("selected_table"),
        "selected_text_col": text_col_resolved or ui_state.get("selected_text_col"),
        "selected_product_col": prod_col_resolved or ui_state.get("selected_product_col")
    }

    opts = {
        "translate_mode": translate_mode,
        "strategy": sentiment_strategy,
        "per_product_cap": int(per_product_cap),
        "pos_threshold": float(pos_threshold),
        "batch_size": int(batch_size),
        "aggressive_scrub": bool(aggressive_scrub)
    }
    return sel_table_resolved, text_col_resolved, prod_col_resolved, opts

def _run_reviews_pipeline(tables: Dict[str,pd.DataFrame], table: str, text_col: str, product_col: str, opts: Dict[str,Any]):
    df = tables[table].copy()
    base = df[[product_col, text_col]].dropna().rename(columns={product_col:"product_id", text_col:"text"})
    base["text"] = base["text"].astype(str)
    base = base[base["text"].str.len() >= CONFIG.get("review_text_min_len", 20)]

    # stratified sampling
    cap = opts.get("per_product_cap", 100)
    sampled = _stratified_sample(base, "product_id", cap)
    st.info(f"Sampling disclaimer: stratified cap = {cap} reviews per product (total {len(sampled)} rows).")

    # PII scrub (before hashing / sending to API)
    sampled["text_scrubbed"] = sampled["text"].apply(lambda t: _pii_scrub(t, aggressive=opts.get("aggressive_scrub", False)))

    # language detection
    langs = detect_lang_cached(tuple(sampled["text_scrubbed"].tolist()), PIPELINE_VERSION)
    sampled["lang"] = langs

    # translate
    mode = opts.get("translate_mode","Auto")
    to_translate_mask = sampled["lang"].ne("en") if mode == "Auto" else (sampled["lang"].notna() if mode == "Always" else pd.Series([False]*len(sampled), index=sampled.index))
    translated = sampled["text_scrubbed"].copy()
    if to_translate_mask.any():
        items = tuple((langs[i], sampled["text_scrubbed"].iloc[i]) for i in range(len(sampled)) if to_translate_mask.iloc[i])
        trans = translate_cached(items, model, PIPELINE_VERSION)
        translated.loc[to_translate_mask] = trans
    sampled["text_en"] = translated

    # sentiment + keywords
    strategy = opts.get("strategy","openai_sentiment_v1")
    if strategy == "openai_sentiment_v1":
        results = sentiment_openai_cached(tuple(sampled["text_en"].tolist()), model, PIPELINE_VERSION)
    else:
        results = sentiment_rule_based(sampled["text_en"].tolist())
    # unpack
    probs = pd.DataFrame([r.get("probs", {"pos":0.33,"neu":0.34,"neg":0.33}) for r in results])
    kws   = [r.get("keywords", []) for r in results]
    sampled = pd.concat([sampled.reset_index(drop=True), probs.reset_index(drop=True)], axis=1)
    sampled["keywords"] = kws

    # positive share as mean probability of positive
    product_summary = sampled.groupby("product_id", as_index=False).agg(
        reviews=("text", "count"),
        positive_share=("pos", "mean"),
        p_pos_mean=("pos","mean"),
        p_neu_mean=("neu","mean"),
        p_neg_mean=("neg","mean")
    ).sort_values("positive_share", ascending=False)

    # UI footer: cache/estimated token
    total_chars = sampled["text_en"].str.len().sum()
    approx_tokens = _estimate_tokens(int(total_chars))
    det_hits, det_tot = st.session_state.cache_stats["det"]
    tr_hits, tr_tot   = st.session_state.cache_stats["tr"]
    se_hits, se_tot   = st.session_state.cache_stats["sent"]
    st.caption(
        f"Cache hit rate — lang: {det_hits}/{det_tot}, translate: {tr_hits}/{tr_tot}, sentiment: {se_hits}/{se_tot}. "
        f"Estimated tokens (analysis): ~{approx_tokens:,}."
    )

    return sampled, product_summary

# ======================
# Render final result (extended)
# ======================

def render_final(ds_json: dict):
    action = (ds_json.get("action") or "").lower()

    if action == "overview":
        st.markdown("### 📊 Table Previews (first 5 rows)")
        for name, df in get_all_tables().items():
            st.markdown(f"**{name}** — rows: {len(df)}, cols: {len(df.columns)}")
            st.dataframe(df.head(5), use_container_width=True)
        add_msg("ds", "Overview rendered."); render_chat(); return

    if action == "eda":
        raw_sql = ds_json.get("duckdb_sql")
        sql_list = raw_sql if isinstance(raw_sql, list) else [raw_sql]
        charts_all = ds_json.get("charts") or []
        for i, sql in enumerate([_sql_first(s) for s in sql_list][:3]):
            if not sql: continue
            try:
                df = run_duckdb_sql(_rewrite_sql_if_missing_product_join(sql))
                st.markdown(f"### 📈 EDA Result #{i+1} (first 50 rows)")
                st.code(sql, language="sql")
                st.dataframe(df.head(50), use_container_width=True)

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
                        if ctype == "line": st.line_chart(plot_df, use_container_width=True)
                        elif ctype == "area": st.area_chart(plot_df, use_container_width=True)
                        else: st.bar_chart(plot_df, use_container_width=True)
            except Exception as e:
                st.error(f"EDA SQL failed: {e}")
        add_msg("ds","EDA rendered."); render_chat(); return

    if action == "sql":
        sql = _sql_first(ds_json.get("duckdb_sql"))
        if not sql:
            add_msg("ds","No SQL provided."); render_chat(); return
        try:
            out = run_duckdb_sql(_rewrite_sql_if_missing_product_join(sql))
            st.markdown("### 🧮 SQL Results (first 25 rows)")
            st.code(sql, language="sql")
            st.dataframe(out.head(25), use_container_width=True)
            add_msg("ds","SQL executed.", artifacts={"sql": sql}); render_chat()
        except Exception as e:
            st.error(f"SQL failed: {e}")
        return

    if action == "calc":
        st.markdown("### 🧮 Calculation")
        st.write(ds_json.get("calc_description","(no description)"))
        add_msg("ds","Calculation displayed."); render_chat(); return

    if action == "feature_engineering":
        sql = _sql_first(ds_json.get("duckdb_sql"))
        base = run_duckdb_sql(_rewrite_sql_if_missing_product_join(sql)) if sql else next(iter(get_all_tables().values())).copy()
        st.markdown("### 🧱 Feature Engineering Base (first 20 rows)")
        st.dataframe(base.head(20), use_container_width=True)
        st.session_state.tables_fe["feature_base"] = base
        add_msg("ds","Feature base ready (saved as 'feature_base')."); render_chat(); return

    
    if action == "modeling":
        sql = _sql_first(ds_json.get("duckdb_sql"))
        plan = ds_json.get("model_plan") or {}
        task = (plan.get("task") or "classification").lower()
        target = plan.get("target")
        base = run_duckdb_sql(_rewrite_sql_if_missing_product_join(sql)) if sql else None

        # 1) Prefer engineered feature base if present
        if base is None and hasattr(st.session_state, "tables_fe") and "feature_base" in st.session_state.tables_fe:
            try:
                base = st.session_state.tables_fe["feature_base"].copy()
            except Exception:
                pass

        # 2) If explicit features are requested, pick a table that actually has them
        req_feats = plan.get("features") or []
        if base is None and req_feats:
            for _, __df in get_all_tables().items():
                try:
                    if all(f in __df.columns for f in req_feats):
                        base = __df.copy()
                        break
                except Exception:
                    continue

        # 3) Otherwise fallback to any sensible table depending on task
        if base is None:
            for _, __df in get_all_tables().items():
                if (task == "clustering") or (target and target in __df.columns):
                    base = __df.copy()
                    break
        if base is None:
            base = next(iter(get_all_tables().values())).copy()

        # Feature presence guard for clustering
        if task == "clustering":
            have = []
            missing = []
            req_feats = plan.get("features") or []
            if req_feats:
                have = [f for f in req_feats if f in base.columns]
                missing = [f for f in req_feats if f not in base.columns]
            if missing and have:
                st.warning(f"Some requested features are missing and will be skipped: {missing}")
                plan["features"] = have
            elif missing and not have and req_feats:
                st.error(f"None of the requested features are present: {missing}. "
                         f"Available columns: {list(base.columns)[:25]} …")
                add_msg("ds","Clustering aborted: required features not found.")
                render_chat()
                return

            result = train_model(base, task, None, plan.get("features") or [], plan.get("model_family") or "", plan.get("n_clusters"))
            st.session_state.last_model_result = result
            _render_model_result(result)
            add_msg("ds", f"Clustering executed on features: {plan.get('features') or 'auto-numeric selection'}.")
            render_chat()
            return
    if action == "aggregate_reviews_from_text":
        tables = get_all_tables()
        if not tables:
            st.warning("Please upload data first.")
            return
        # controls + candidates
        tname, text_col, prod_col, opts = _reviews_controls_and_candidates(tables)
        if not tname or not text_col or not prod_col:
            st.warning("Select a table, text column, and product column to proceed (or use the auto suggestion).")
            return
        try:
            sampled, product_summary = _run_reviews_pipeline(tables, tname, text_col, prod_col, opts)
            st.markdown("### 🧾 Sampled Reviews (head)")
            st.dataframe(sampled.head(50), use_container_width=True)
            st.markdown("### ✅ Positive Share by Product (mean of p_pos)")
            st.dataframe(product_summary.head(200), use_container_width=True)
        except Exception as e:
            st.error(f"Reviews pipeline error: {e}")
        add_msg("ds","Reviews aggregation completed.", artifacts={"table": tname, "text_col": text_col, "product_col": prod_col}); render_chat(); return

    add_msg("ds", f"Action '{action}' not recognized.", artifacts=ds_json); render_chat()

# ======================
# Coordinator (threads)
# ======================

def _start_new_thread(central_q: str):
    st.session_state.threads.append({"central": central_q, "followups": []})
    st.session_state.active_thread_idx = len(st.session_state.threads) - 1

def _append_followup(text: str):
    idx = st.session_state.active_thread_idx
    if idx is not None and 0 <= idx < len(st.session_state.threads):
        st.session_state.threads[idx]["followups"].append(text)

def run_turn_ceo(new_text: str):
    _ensure_state_defaults()
    prev = st.session_state.current_question or ""
    intent = classify_intent(prev, new_text)
    st.session_state.user_question_history.append(new_text)

    if intent == "new_request" or st.session_state.active_thread_idx is None:
        _start_new_thread(new_text)
        effective_q = new_text
        add_msg("system", "Starting a new analysis thread for the central question.")
    else:
        _append_followup(new_text)
        central = st.session_state.threads[st.session_state.active_thread_idx]["central"]
        effective_q = central + "\n\n[Follow-up]: " + new_text
        add_msg("system", "Interpreting your message as follow-up/feedback on the active central question.")

    st.session_state.current_question = effective_q
    st.session_state.last_user_prompt = new_text

    col_hints = build_column_hints(effective_q)

    # 1) AM plan
    am_json = run_am_plan(effective_q, col_hints, st.session_state.user_question_history)

    if am_json.get("need_more_info"):
        qs = am_json.get("clarifying_questions") or ["Could you clarify your objective?"]
        add_msg("am", "I need a bit more context:")
        for q in qs[:3]:
            add_msg("am", f"• {q}")
        render_chat()
        return

    # 2) DS executes pipeline
    executed = run_ds_pipeline(am_json, col_hints, effective_q, st.session_state.user_question_history)
    if executed and executed[-1].get("_awaiting_clarification"):
        return

    if not executed:
        add_msg("system", "No DS steps executed."); render_chat(); return

    last_ds = executed[-1]

    # 3) Review
    meta = build_meta(last_ds)
    review = am_review(effective_q, last_ds, meta)
    add_msg("am", review.get("summary_for_ceo",""), artifacts={
        "appropriateness_check": review.get("appropriateness_check"),
        "gaps_or_risks": review.get("gaps_or_risks"),
        "improvements": review.get("improvements"),
        "suggested_next_steps": review.get("suggested_next_steps"),
        "must_revise": review.get("must_revise"),
        "sufficient_to_answer": review.get("sufficient_to_answer"),
    })
    render_chat()

    if review.get("clarification_needed") and (review.get("clarifying_questions") or []):
        add_msg("am", "Before proceeding, could you clarify:")
        for q in (review.get("clarifying_questions") or [])[:3]:
            add_msg("am", f"• {q}")
        render_chat()
        return

    if review.get("sufficient_to_answer") and not review.get("must_revise"):
        return

    revised = revise_ds(am_json, last_ds, review, col_hints, effective_q, st.session_state.user_question_history)
    add_msg("ds", revised.get("ds_summary","(revised)"), artifacts={"action": revised.get("action")}); render_chat()
    render_final(revised)

# ======================
# Data loading
# ======================
if zip_file and st.session_state.tables_raw is None:
    st.session_state.tables_raw = load_zip_tables(zip_file)
    st.session_state.tables = get_all_tables()
    add_msg("system", f"Loaded {len(st.session_state.tables_raw)} raw tables."); render_chat()

# ======================
# Chat UI (history preserved)
# ======================
st.subheader("Chat")
_ensure_state_defaults()
render_chat()

user_prompt = st.chat_input("You're the CEO. Ask a question (e.g., 'What data do we have?' or 'Run review analysis with aggregate_reviews_from_text')")
if user_prompt:
    add_msg("user", user_prompt); render_chat()
    run_turn_ceo(user_prompt)
