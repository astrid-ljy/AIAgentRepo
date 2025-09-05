# app.py — Updated full version with guardrails, Spanish review NLP, persistent DuckDB,
# DS/AM prompt tightening, and PRODUCT POSITIVE-SHARE AGGREGATION from review TEXT (no review_score needed)

import os
import re
import json
import time
import hashlib
import random
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
MAX_JSON_ITEMS_PER_CALL = 40  # batch cap for translation/NLP to control cost
OPENAI_MAX_RETRIES = 5
OPENAI_BASE_DELAY = 1.0  # seconds
OPENAI_TIMEOUT = 90      # seconds per request

# One-hot control (used only in optional classification of simple user asks)
ALLOW_DS_ACTIONS = {
    "overview": True,
    "eda": True,
    "sql": True,
    "model": True,
}

# Review column name patterns (kept from your code)
REVIEW_COL_PAT = re.compile(r"(review|comentario|opinion|texto|descripcion|descripción)", re.I)

DB_PATH = os.getenv("DUCKDB_PATH", "app.duckdb")

###############################################
# ---------- OpenAI helpers ------------------
###############################################

def ensure_openai():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")
    if OpenAI is None:
        raise RuntimeError("openai package not available in this environment")
    return OpenAI(api_key=key)


def with_backoff(func):
    def wrapper(*args, **kwargs):
        delay = OPENAI_BASE_DELAY
        for attempt in range(OPENAI_MAX_RETRIES):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == OPENAI_MAX_RETRIES - 1:
                    raise
                time.sleep(delay)
                delay *= 2
    return wrapper


@with_backoff
def openai_chat_json(system_prompt: str, user_json: str, *, model: Optional[str] = None) -> Dict[str, Any]:
    client = ensure_openai()
    model = model or st.session_state.get("selected_model", DEFAULT_MODEL)
    msg = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_json},
    ]
    out = client.chat.completions.create(
        model=model,
        temperature=0.1,
        response_format={"type": "json_object"},
        messages=msg,
        timeout=OPENAI_TIMEOUT,
    )
    return json.loads(out.choices[0].message.content)


@with_backoff
def openai_chat_list(system_prompt: str, user_list: List[str], *, model: Optional[str] = None) -> List[Any]:
    client = ensure_openai()
    model = model or st.session_state.get("selected_model", DEFAULT_MODEL)
    parts = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps({"items": user_list}, ensure_ascii=False)},
    ]
    out = client.chat.completions.create(
        model=model,
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=parts,
        timeout=OPENAI_TIMEOUT,
    )
    js = json.loads(out.choices[0].message.content)
    return js.get("results", [])


def _openai_batch(system_prompt: str, items: List[str]) -> List[Any]:
    results: List[Any] = []
    for i in range(0, len(items), MAX_JSON_ITEMS_PER_CALL):
        chunk = items[i:i + MAX_JSON_ITEMS_PER_CALL]
        try:
            results.extend(openai_chat_list(system_prompt, chunk))
        except Exception as e:
            # Best-effort: return placeholders for failed items
            results.extend([{} for _ in chunk])
    return results

###############################################
# ---------- Spanish-aware helpers -----------
###############################################

# --- Spanish-aware review ranking helpers ---
SPANISH_KEY_PHRASES = [
    "servicio","envío","envio","entrega","calidad","devolución","devolucion","precio",
    "tiempo","atención","atencion","cliente","producto","tamaño","tamano","color",
    "funciona","mal","bien","rápido","rapido","lento","paquete","dañado","danado",
    "reembolso","garantía","garantia","recomendado","defectuoso","vendedor",
    "descripcion","descripción","original","falso",
]
EMOJI_PATTERN = re.compile(r"[🌀-🛿🤀-🧿☀-⛿✀-➿]")


def _looks_spanish_score(text: str) -> float:
    if not isinstance(text, str) or not text:
        return 0.0
    t = text.lower()
    hits = 0
    if any(c in t for c in ["á","é","í","ó","ú","ñ"]):
        hits += 1
    for w in [" el "," la "," los "," las "," de "," para "," con "," sin "," muy "," no "," es "]:
            if w in f" {t} ":
                hits += 1
    for kw in SPANISH_KEY_PHRASES[:10]:
        if kw in t:
            hits += 1
    tokens = max(1, len(t.split()))
    return min(1.0, hits / min(tokens, 20))


def _score_text_column(series: pd.Series) -> float:
    try:
        s = series.dropna().astype(str).sample(min(500, len(series)), random_state=42)
    except Exception:
        s = series.dropna().astype(str)
    if len(s) == 0:
        return 0.0
    s_lower = s.str.lower()
    def key_hits(x: str) -> int:
        return sum(1 for kw in SPANISH_KEY_PHRASES if kw in x)
    key_score   = s_lower.apply(key_hits).mean()
    emoji_score = s_lower.apply(lambda x: len(EMOJI_PATTERN.findall(x))).mean()
    len_score   = np.clip(s_lower.str.len().mean() / 200.0, 0, 1)
    es_score    = s_lower.apply(_looks_spanish_score).mean()
    return float(0.45*key_score + 0.25*emoji_score + 0.20*len_score + 0.10*es_score)


###############################################
# ---------- System prompts ------------------
###############################################

SYSTEM_AM = """
You are an Analytics Manager (AM). Your job is to pick the best next action for a Data Scientist (DS) to answer the user's question using the available database.
Allowed actions: overview, eda, sql, model. If the question requires sentiment from REVIEW TEXT (no numeric review score), tell DS to aggregate reviews by product using translation→sentiment pipeline.
Return JSON: {"action": one of ["overview","eda","sql","model"], "instructions": "..."}
"""

SYSTEM_DS = """
You are a Data Scientist (DS). Follow AM instructions. If AM asks for review-based positive share, do NOT assume a numeric score. Instead, you must:
1) locate the review TEXT column (Spanish possible),
2) translate (Spanish→English) only if needed,
3) run sentiment, and
4) aggregate by product via order join.
Return JSON: {"notes": "what you will do", "sql": "SQL if needed or empty string"}
"""

###############################################
# ---------- DuckDB Utils --------------------
###############################################

class DB:
    def __init__(self, path: str):
        self.con = duckdb.connect(path)

    def tables(self) -> List[str]:
        return [r[0] for r in self.con.execute("SHOW TABLES").fetchall()]

    def head(self, table: str, n: int = 5) -> pd.DataFrame:
        return self.con.execute(f"SELECT * FROM {table} LIMIT {n}").fetchdf()

    def sql(self, q: str) -> pd.DataFrame:
        return self.con.execute(q).fetchdf()


db = DB(DB_PATH)

###############################################
# ---------- Review/Text Pipeline ------------
###############################################

def detect_review_columns(df: pd.DataFrame) -> List[str]:
    """Rank text-like columns by Spanish review-likelihood; return best-first list."""
    textlike = [c for c in df.columns if pd.api.types.is_string_dtype(df[c])]
    if not textlike:
        return []
    scores = {c: _score_text_column(df[c]) for c in textlike}
    ranked = sorted(textlike, key=lambda c: scores[c], reverse=True)
    preferred = [c for c in ranked if REVIEW_COL_PAT.search(str(c))]
    if preferred:
        ranked = preferred + [c for c in ranked if c not in preferred]
    ranked = [c for c in ranked if scores[c] >= 0.40]
    return ranked


def looks_spanish(text: str) -> bool:
    return _looks_spanish_score(text) >= 0.25


@st.cache_data(show_spinner=False)
def llm_translate_es_to_en(texts: List[str]) -> List[str]:
    sys = (
        "You are a precise translator. Translate from Spanish (or mixed Spanish-English) to English. "
        "Return JSON as {\"results\": [\"...translated...\"]} in the same order and length."
    )
    res = _openai_batch(sys, texts)
    if res and isinstance(res[0], dict) and "text" in res[0]:
        return [r.get("text", "") for r in res]
    return [str(r) for r in res]


@st.cache_data(show_spinner=False)
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

    # Only translate if batch looks Spanish
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
    # Standardize column name for downstream components
    out[f"{review_col}_en"] = translated
    out["review_sentiment"] = sentiments
    out["review_keywords"] = keywords
    return out


###############################################
# ---------- Positive Share Aggregation ------
###############################################

def compute_product_positive_share(sample_cap: int = 3000) -> pd.DataFrame:
    # 1) Choose source tables (your existing logic)
    # Expect: orders, order_items, reviews (TEXT)
    tables = db.tables()
    if not {"orders", "order_items", "reviews"}.issubset(set(tables)):
        st.warning("orders/order_items/reviews tables not found in DuckDB")
        return pd.DataFrame()

    # 2) Sample reviews to control cost
    reviews = db.sql("SELECT * FROM reviews")
    if sample_cap:
        reviews = reviews.sample(min(len(reviews), sample_cap), random_state=42)

    # 3) Spanish-aware review augmentation (translate/sentiment)
    reviews_aug = augment_reviews(reviews, enable=True)

    # 4) Join reviews → orders → products (depends on your schema; keep your original SQL)
    # NOTE: we assume review has order_id; order_items maps order_id→product_id
    tmp = db.con.register("__tmp_reviews", reviews_aug)
    j = db.con.execute(
        """
        WITH r AS (
          SELECT * FROM __tmp_reviews
        ),
        oi AS (
          SELECT order_id, product_id FROM order_items
        )
        SELECT oi.product_id,
               r.review_sentiment AS sentiment
        FROM r
        JOIN oi USING(order_id)
        """
    ).fetchdf()

    if j.empty:
        return pd.DataFrame()

    # 5) Convert label→prob (soften)
    label_to_prob = {"positive": 1.0, "neutral": 0.5, "negative": 0.0}
    j["p_pos"] = j["sentiment"].map(label_to_prob).fillna(0.5)

    # 6) Aggregate
    agg = (
        j.groupby("product_id").agg(
            reviews=("p_pos", "size"),
            pos_est=("p_pos", "mean"),
        ).reset_index()
    )
    agg["positive_share"] = (agg["p_pos"] if "p_pos" in agg.columns else agg["pos_est"]).clip(0, 1)
    agg = agg.sort_values(["positive_share", "pos_est", "reviews"], ascending=[False, False, False])
    return agg


###############################################
# ---------- Sidebar Controls ----------------
###############################################

with st.sidebar:
    st.subheader("⚙️ Settings")
    st.session_state.setdefault("selected_model", DEFAULT_MODEL)
    st.session_state["selected_model"] = st.text_input("OpenAI Chat Model", st.session_state["selected_model"])  # keep your existing UI
    st.session_state.setdefault("max_reviews_for_positive_share", 3000)
    st.session_state["max_reviews_for_positive_share"] = st.number_input(
        "Max reviews analyzed (cap)", min_value=500, max_value=100000, value=int(st.session_state["max_reviews_for_positive_share"]), step=500
    )

###############################################
# ---------- App Header & Inputs -------------
###############################################

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

st.write("This version keeps your architecture and adds Spanish-aware review detection, caching for translation & sentiment, and removes the hidden bypass.")

user_query = st.text_input(
    "Ask a question",
    placeholder="e.g., What data do we have? Which product has the highest proportion of positive reviews?"
)

# quick intent hook: if user asks for highest proportion of positive reviews, seed AM text
intent_positive_share = bool(re.search(r"(highest|top).*(proportion|share).*(positive).*(review)", user_query or "", re.I))

# Store sidebar control for compute pipeline
st.session_state["max_reviews_for_positive_share"] = int(st.session_state.get("max_reviews_for_positive_share", 3000))

if user_query:
    if intent_positive_share:
        user_query = ("Which product has the highest proportion of positive reviews? "
                      "Important: There is no numeric review_score. Use review TEXT (translate Spanish→English → sentiment) "
                      "and aggregate by product via order_id join.")
    if True:
        # AM decides
        try:
            am_out = openai_chat_json(
                SYSTEM_AM,
                json.dumps({"question": user_query}, ensure_ascii=False)
            )
        except Exception as e:
            st.error(f"AM failed: {e}")
            am_out = {"action": "explain", "instructions": "Provide a plain-language explanation."}

        st.write("**AM decision**:", am_out)

        # DS executes/clarifies
        try:
            ds_out = openai_chat_json(
                SYSTEM_DS,
                json.dumps({"am": am_out, "question": user_query}, ensure_ascii=False)
            )
        except Exception as e:
            st.error(f"DS failed: {e}")
            ds_out = {"notes": "", "sql": ""}

        st.write("**DS plan**:", ds_out)

        # Execute if SQL; or run special pipeline if AM implied review aggregation
        ran_pipeline = False
        if "review" in user_query.lower() and ("positive" in user_query.lower() or "sentiment" in user_query.lower()):
            ran_pipeline = True
            agg = compute_product_positive_share(sample_cap=st.session_state["max_reviews_for_positive_share"])
            if agg is not None and not agg.empty:
                st.subheader("Top products by positive review share (from TEXT)")
                st.dataframe(agg, use_container_width=True)
            else:
                st.info("No results from review-based aggregation.")

        sql_q = (ds_out.get("sql") or "").strip()
        if sql_q:
            try:
                df = db.sql(sql_q)
                st.subheader("SQL result")
                st.dataframe(df, use_container_width=True)
            except Exception as e:
                st.error(f"SQL execution failed: {e}")
        elif not ran_pipeline:
            st.info("No SQL to run and no review aggregation requested.")

else:
    st.caption("Tip: Ask ‘Which product has the highest proportion of positive reviews?’ to trigger the review pipeline via AM/DS.")


###############################################
# ---------- Footer --------------------------
###############################################

st.caption("Spanish-aware column ranking uses keyphrases, emoji, length, and a lightweight language heuristic. Translation & sentiment are cached to reduce cost/latency.")
