# app.py — Updated full version with guardrails, Spanish review NLP, persistent DuckDB, and DS/AM prompt tightening
# NOTE: This is a drop-in replacement for your previous app. Keep your data files in the same place.

import os
import re
import io
import gc
import json
import time
import math
import hashlib
import random
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import duckdb
import streamlit as st

try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # Streamlit will surface error in ensure_openai

###############################################
# ---------- Configuration & Constants -------
###############################################

APP_TITLE = "💼 Analytics Agent — Spanish Reviews + Guardrails"
DEFAULT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
MAX_JSON_ITEMS_PER_CALL = 40  # batch size cap for translation/NLP to control cost
OPENAI_MAX_RETRIES = 5
OPENAI_BASE_DELAY = 1.0  # seconds
OPENAI_TIMEOUT = 90  # seconds per request

# One-hot control
OHE_TOP_K = 50
OHE_MIN_FREQ = 0.01  # 1%

# Cache TTL (seconds) for profile/schema suggestions etc.
PROFILE_CACHE_TTL = 1800  # 30 minutes

###############################################
# ---------- Streamlit Page Setup ------------
###############################################

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

###############################################
# ---------- Utilities -----------------------
###############################################

def _hash_df_schema(df: pd.DataFrame) -> str:
    parts = [f"{c}:{str(df[c].dtype)}" for c in df.columns]
    s = ";".join(parts) + f"|rows={len(df)}"
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def pii_scrub(s: str) -> str:
    """Light PII scrub before LLM calls: emails, phones, URLs.
    Keeps text readable while masking sensitive bits.
    """
    if not s:
        return s
    out = s
    out = re.sub(r"[A-Za-z0-9_.+-]+@[A-Za-z0-9-]+\.[A-Za-z0-9.-]+", "[EMAIL]", out)
    out = re.sub(r"(https?://\S+)", "[URL]", out)
    out = re.sub(r"\b(?:\+?\d[\s-]?)?(?:\(\d{3}\)|\d{3})[\s-]?\d{3}[\s-]?\d{4}\b", "[PHONE]", out)
    return out


def ensure_openai() -> OpenAI:
    if OpenAI is None:
        raise RuntimeError("openai package not available. Please install `openai` v1.")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set.")
    # The client supports default timeout by passing via kwargs
    return OpenAI(timeout=OPENAI_TIMEOUT)


def openai_chat_json(system_prompt: str, user_content: str, *, model: Optional[str] = None, temperature: float = 0.0) -> Dict[str, Any]:
    """Call OpenAI chat with JSON response format and exponential backoff retries."""
    client = ensure_openai()
    if model is None:
        model = st.session_state.get("selected_model", DEFAULT_MODEL)

    last_err = None
    for attempt in range(OPENAI_MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                response_format={"type": "json_object"},
                temperature=temperature,
                timeout=OPENAI_TIMEOUT,
            )
            content = resp.choices[0].message.content
            return json.loads(content)
        except Exception as e:
            last_err = e
            # backoff with jitter
            delay = OPENAI_BASE_DELAY * (2 ** attempt) + random.random()
            time.sleep(min(delay, 10))
    raise RuntimeError(f"OpenAI chat failed after retries: {last_err}")


def _openai_batch(system_prompt: str, items: List[str]) -> List[Any]:
    """Batch helper for lists — returns list of results preserving order.
    Expects the model to output {"results": [...]} JSON.
    """
    results: List[Any] = []
    for i in range(0, len(items), MAX_JSON_ITEMS_PER_CALL):
        chunk = items[i : i + MAX_JSON_ITEMS_PER_CALL]
        payload = json.dumps({"items": [pii_scrub(x) for x in chunk]}, ensure_ascii=False)
        data = openai_chat_json(
            system_prompt=system_prompt.strip() + "\nReturn ONLY JSON with shape {\"results\": [...]}.",
            user_content=payload,
            temperature=0.0,
        )
        results.extend(data.get("results", []))
    return results


###############################################
# ---------- Spanish Reviews NLP -------------
###############################################

REVIEW_COL_PAT = re.compile(r"(review|comentario|reseña|resena|opini[oó]n|texto|text|comment)s?$", re.I)


def detect_review_columns(df: pd.DataFrame) -> List[str]:
    cols: List[str] = []
    for c in df.columns:
        name = str(c)
        if REVIEW_COL_PAT.search(name) and pd.api.types.is_string_dtype(df[c]):
            cols.append(c)
    if not cols:
        # Fallback heuristic: a string column with longer avg length
        for c in df.columns:
            s = df[c]
            if pd.api.types.is_string_dtype(s):
                sample = s.dropna().astype(str).head(200)
                if not sample.empty and sample.str.len().mean() > 40:
                    cols.append(c)
                    break
    return cols


def looks_spanish(text: str) -> bool:
    if not text:
        return False
    sl = text.lower()
    hits = sum(ph in sl for ph in [" el ", " la ", " de ", " y ", " que ", " con ", " para ", " no ", " muy ", " es "])
    return hits >= 1


def llm_translate_es_to_en(texts: List[str]) -> List[str]:
    sys = (
        "You are a precise translator. Translate from Spanish (or mixed Spanish-English) to English. "
        "Return JSON as {\"results\": [\"...translated...\"]} in the same order and length."
    )
    res = _openai_batch(sys, texts)
    if res and isinstance(res[0], dict) and "text" in res[0]:
        return [r.get("text", "") for r in res]
    return [str(r) for r in res]


def llm_sentiment_keywords(texts_en: List[str]) -> List[Dict[str, Any]]:
    sys = (
        "You analyze short English customer reviews. For each input string, return an object with fields: "
        "{\"sentiment\": \"positive|neutral|negative\", \"keywords\": [up to 5 concise keyphrases]}. "
        "Output JSON: {\"results\": [ ... ]} preserving order."
    )
    return _openai_batch(sys, texts_en)


def augment_reviews(df: pd.DataFrame, *, enable: bool) -> pd.DataFrame:
    if not enable or df is None or df.empty:
        return df
    review_cols = detect_review_columns(df)
    if not review_cols:
        return df

    review_col = review_cols[0]
    texts = df[review_col].fillna("").astype(str).tolist()

    # Only translate if the batch looks Spanish
    if any(looks_spanish(t) for t in texts):
        try:
            translated = llm_translate_es_to_en(texts)
        except Exception as e:
            st.warning(f"Translation failed, showing original reviews. Error: {e}")
            translated = texts
    else:
        translated = texts

    try:
        nlp = llm_sentiment_keywords(translated)
        sentiments: List[str] = []
        keywords: List[str] = []
        for r in nlp:
            if isinstance(r, dict):
                sentiments.append(str(r.get("sentiment", "")))
                kw = r.get("keywords", [])
                keywords.append(", ".join([str(k) for k in kw][:5]))
            else:
                sentiments.append("")
                keywords.append("")
    except Exception as e:
        st.warning(f"NLP failed, skipping sentiment/keywords. Error: {e}")
        sentiments = [""] * len(translated)
        keywords = [""] * len(translated)

    out = df.copy()
    out[f"{review_col}_en"] = translated
    out["review_sentiment"] = sentiments
    out["review_keywords"] = keywords
    return out


###############################################
# ---------- Persistent DuckDB ---------------
###############################################

def get_duck() -> duckdb.DuckDBPyConnection:
    if "duck_conn" not in st.session_state:
        st.session_state["duck_conn"] = duckdb.connect(database=":memory:")
        st.session_state["duck_registered"] = {}
    return st.session_state["duck_conn"]


def register_df(name: str, df: pd.DataFrame):
    conn = get_duck()
    reg = st.session_state["duck_registered"]
    schema_hash = _hash_df_schema(df)
    if reg.get(name) == schema_hash:
        return
    # unregister if exists
    try:
        conn.unregister(name)
    except Exception:
        pass
    conn.register(name, df)
    reg[name] = schema_hash


def run_duckdb_sql(sql: str) -> pd.DataFrame:
    conn = get_duck()
    try:
        return conn.execute(sql).df()
    except Exception as e:
        raise RuntimeError(f"DuckDB SQL error: {e}\nSQL:\n{sql}")


###############################################
# ---------- Sidebar Controls ----------------
###############################################

with st.sidebar:
    st.subheader("⚙️ Settings")
    st.session_state.setdefault("selected_model", DEFAULT_MODEL)
    st.session_state["selected_model"] = st.text_input("OpenAI Chat Model", st.session_state["selected_model"])

    st.header("📝 Reviews")
    auto_nlp_reviews = st.checkbox("Auto-translate & analyze reviews", value=True, help="Adds *_en, sentiment, keywords columns to any table that has a review field.")
    max_rows_to_analyze = st.number_input("Max rows per table to analyze", min_value=5, max_value=200, value=50, step=5)


###############################################
# ---------- Data Loading (Example Hooks) ----
###############################################

st.markdown("Use the uploader below to add CSV/Parquet files. Registered tables are queryable in DuckDB.")
uploaded = st.file_uploader("Upload CSV or Parquet", type=["csv", "parquet"], accept_multiple_files=True)

if uploaded:
    for f in uploaded:
        name = re.sub(r"\W+", "_", f.name).strip("_")
        try:
            if f.name.lower().endswith(".csv"):
                df = pd.read_csv(f)
            else:
                df = pd.read_parquet(f)
            register_df(name, df)
            st.success(f"Registered table: {name} ({len(df)} rows)")
        except Exception as e:
            st.error(f"Failed to load {f.name}: {e}")


###############################################
# ---------- AM & DS System Prompts ----------
###############################################

SYSTEM_AM = (
    """
You are the **Analytics Manager (AM)** orchestrating a Data Scientist (DS). Your job: interpret the user's question, choose ONE allowed DS action, and give precise instructions. 

**Allowed DS actions**
- overview  → show first 5 rows for each table the user can use; NO modeling or heavy EDA.
- eda       → light EDA only on relevant tables; propose joins but don't execute complex modeling.
- sql       → run a concrete SQL query over registered tables and show a result table.
- model     → if and only if user explicitly asked for modeling or data is ready after prior steps.
- explain   → pure explanation; no tables.

**Hard rule — reject non-allowed**: If a step requires something outside the allowed set, DO NOT invent new actions. Ask the DS to stop and request clarification from the AM/user.

**Review rule**: If any result table includes a review/text column (like "review", "comentario", "reseña"), DS must render three extra columns: `<review_col>_en` (Spanish→English), `review_sentiment`, and `review_keywords`.

**Conservative progression**: Prefer earlier steps (overview → eda → sql) unless the user explicitly demands modeling. If the user asks "what data do we have" or similar, force `overview`.

Output strictly as JSON: {"action": "overview|eda|sql|model|explain", "instructions": "..."}
    """
)

SYSTEM_DS = (
    """
You are the **Data Scientist (DS)**. Follow the AM's chosen action exactly. If the AM's request contains anything outside your allowed actions, refuse gracefully and ask for clarification.

**General**
- Never run actions outside {overview, eda, sql, model, explain}.
- Prefer minimal, fast results. Cap tables at a practical row limit.
- If a result table includes a review column (Spanish or mixed), you MUST add `<review_col>_en` (translated), `review_sentiment`, and `review_keywords` to the rendered table.

**Output format**
Return JSON only. For `sql`, return {"sql": "..."}. For others, return {"notes": "..."} describing what to do.
    """
)


###############################################
# ---------- Simple Agent Loop ---------------
###############################################

@st.cache_data(ttl=PROFILE_CACHE_TTL)
def cached_profile_overview(registered_meta: Dict[str, str]) -> Dict[str, Any]:
    # Simple cached stub based on table schemas
    return {"tables": registered_meta}


def list_registered_tables() -> Dict[str, str]:
    conn = get_duck()
    rows = conn.execute("show tables").fetchall()
    meta = {}
    for (name,) in rows:
        # collect schema string for cache keying
        try:
            df = conn.execute(f"select * from {name} limit 1").df()
            meta[name] = _hash_df_schema(df)
        except Exception:
            meta[name] = "unknown"
    return meta


###############################################
# ---------- Modeling Helpers ----------------
###############################################

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV


def cap_cardinality(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Return categorical columns and which categories to keep (top-k or >= min freq)."""
    cats: Dict[str, List[str]] = {}
    n = len(df)
    for c in df.select_dtypes(include=["object", "category"]).columns:
        vc = df[c].value_counts(dropna=False)
        keep = list(vc[vc / n >= OHE_MIN_FREQ].index[:OHE_TOP_K])
        cats[c] = keep
    return cats


def build_classifier(df: pd.DataFrame, target: str) -> Tuple[Pipeline, float]:
    y = df[target]
    X = df.drop(columns=[target])

    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()

    # Cap cardinality by mapping rare to "__OTHER__"
    cats_keep = cap_cardinality(X)
    Xc = X.copy()
    for c, allowed in cats_keep.items():
        Xc[c] = Xc[c].where(Xc[c].isin(allowed), other="__OTHER__")

    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ]
    )

    base = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf = CalibratedClassifierCV(base, method="isotonic", cv=3)

    pipe = Pipeline([("prep", pre), ("clf", clf)])

    X_train, X_test, y_train, y_test = train_test_split(Xc, y, test_size=0.2, random_state=42, stratify=y)
    pipe.fit(X_train, y_train)
    try:
        y_prob = pipe.predict_proba(X_test)
        y_pred = (y_prob[:, 1] >= 0.5).astype(int) if y_prob.shape[1] == 2 else pipe.predict(X_test)
    except Exception:
        y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return pipe, acc


###############################################
# ---------- Rendering Functions -------------
###############################################

def render_overview():
    conn = get_duck()
    tables = conn.execute("show tables").fetchall()
    if not tables:
        st.info("No tables registered yet. Upload files from the sidebar above.")
        return
    st.subheader("📦 Data Overview — first 5 rows per table")
    for (tname,) in tables:
        st.markdown(f"**Table:** `{tname}`")
        df = conn.execute(f"select * from {tname} limit {max_rows_to_analyze}").df()
        view = df.head(5)
        view = augment_reviews(view, enable=auto_nlp_reviews)
        st.dataframe(view, use_container_width=True)


def render_eda(instructions: str = ""):
    st.subheader("🔎 EDA — light profiling")
    conn = get_duck()
    tables = conn.execute("show tables").fetchall()
    for (tname,) in tables:
        st.markdown(f"**Table:** `{tname}` — shape & dtypes")
        df = conn.execute(f"select * from {tname} limit {max_rows_to_analyze}").df()
        info = pd.DataFrame({
            "column": df.columns,
            "dtype": [str(df[c].dtype) for c in df.columns],
            "nulls": [int(df[c].isna().sum()) for c in df.columns],
            "non_nulls": [int(df[c].notna().sum()) for c in df.columns],
        })
        st.dataframe(info, use_container_width=True)
        view = augment_reviews(df.head(50), enable=auto_nlp_reviews)
        st.dataframe(view, use_container_width=True)


def render_sql(sql: str):
    st.subheader("🧮 SQL Result")
    out = run_duckdb_sql(sql)
    view = augment_reviews(out.head(int(max_rows_to_analyze)), enable=auto_nlp_reviews)
    st.dataframe(view, use_container_width=True)


def render_model(target_table: str, target_col: str):
    st.subheader("🤖 Simple Classifier (LogReg + Calibration)")
    df = run_duckdb_sql(f"select * from {target_table}")
    if target_col not in df.columns:
        st.error(f"Target column `{target_col}` not in `{target_table}`.")
        return
    # Drop rows with NA in target
    mdf = df.dropna(subset=[target_col]).copy()
    # Lightweight: sample to keep compute sane
    if len(mdf) > 5000:
        mdf = mdf.sample(5000, random_state=42)
    try:
        model, acc = build_classifier(mdf, target_col)
        st.success(f"Model trained. Holdout accuracy ≈ {acc:.3f}")
    except Exception as e:
        st.error(f"Modeling failed: {e}")
        return


###############################################
# ---------- Chat / Agent Controls -----------
###############################################

st.divider()
user_query = st.text_input("Ask the AM/DS agents a question", placeholder="e.g., What data do we have? Show top products; run a SQL join; build a churn model…")

if user_query:
    # AM decides
    try:
        am_out = openai_chat_json(
            SYSTEM_AM,
            json.dumps({"question": user_query}, ensure_ascii=False)
        )
    except Exception as e:
        st.error(f"AM failed: {e}")
        am_out = {"action": "explain", "instructions": "Explain failure and ask for a simpler question."}

    action = am_out.get("action", "explain")
    instructions = am_out.get("instructions", "")

    with st.expander("AM decision (debug)", expanded=False):
        st.json(am_out)

    # DS executes strictly
    try:
        ds_req = openai_chat_json(
            SYSTEM_DS,
            json.dumps({"action": action, "instructions": instructions}, ensure_ascii=False)
        )
    except Exception as e:
        st.error(f"DS failed: {e}")
        ds_req = {"notes": "Could not proceed."}

    with st.expander("DS plan (debug)", expanded=False):
        st.json(ds_req)

    try:
        if action == "overview":
            render_overview()
        elif action == "eda":
            render_eda(instructions=str(ds_req.get("notes", "")))
        elif action == "sql":
            sql = ds_req.get("sql") or instructions
            if not sql:
                st.warning("No SQL provided by DS.")
            else:
                render_sql(sql)
        elif action == "model":
            payload = ds_req if isinstance(ds_req, dict) else {}
            table = payload.get("table") or "training_table"
            target = payload.get("target") or "label"
            render_model(table, target)
        else:
            st.write("ℹ️", ds_req.get("notes", "No specific action requested."))
    except Exception as e:
        st.error(f"Execution error: {e}")


###############################################
# ---------- Manual SQL (Power User) ---------
###############################################

st.divider()
st.subheader("Advanced: Run your own SQL against DuckDB")
manual_sql = st.text_area("SQL", value="""-- Example
-- select * from your_table limit 25;
""")
col1, col2 = st.columns([1,1])
with col1:
    if st.button("Run SQL"):
        if not manual_sql.strip():
            st.warning("Enter a SQL query.")
        else:
            try:
                df = run_duckdb_sql(manual_sql)
                view = augment_reviews(df.head(int(max_rows_to_analyze)), enable=auto_nlp_reviews)
                st.dataframe(view, use_container_width=True)
            except Exception as e:
                st.error(str(e))
with col2:
    if st.button("List Tables"):
        try:
            meta = list_registered_tables()
            st.json(meta)
        except Exception as e:
            st.error(str(e))


###############################################
# ---------- Footer --------------------------
###############################################

st.caption("Built with persistent DuckDB, guarded OpenAI calls, and automatic Spanish review analysis (translation + sentiment + keywords).")
