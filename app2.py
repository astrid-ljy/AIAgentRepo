# app4.py — fully updated
# Key upgrades:
# - Spanish-aware review column ranking (keywords + emoji)
# - Sidebar overrides for table/column, thresholds, sampling
# - Language handling modes: auto / always / never (translate)
# - Pluggable NLP providers (OpenAI / rule-based fallback)
# - Probability-based sentiment + configurable positive threshold
# - Stratified sampling per product
# - Caching with st.cache_data + optional DuckDB persistence
# - No hidden regex bypass; actions go through a single dispatcher
# - Basic JSON schema validation for DS outputs
#
# Note: This file is self-contained and avoids new hard dependencies.
# If you want robust language detection/translation, plug in your preferred
# libraries/providers in the provider hooks below.

from __future__ import annotations
import os
import re
import json
import time
import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import duckdb
import pandas as pd
import numpy as np
import streamlit as st

# -----------------------------
# Config (can be moved to YAML)
# -----------------------------
DEFAULT_CONFIG = {
    "nlp": {
        "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "lang_mode": "auto",  # auto | always | never (translate to English)
        "translate_provider": "openai",  # openai | none
        "sentiment_provider": "openai",  # openai | rule_based
        "output": "probs",  # label | probs
        "positive_threshold": 0.7,
        "pipeline_version": "nlp_v3.1_es",
    },
    "sampling": {
        "mode": "per_product_cap",  # global | per_product_cap
        "cap_per_product": 50,
        "global_max_reviews": 5000,
        "random_state": 42,
    },
    "columns": {
        "product_id": "product_id",
        "review_text": "auto",  # auto | explicit column name
    },
    "cache": {
        "use_duckdb": True,
        "db_path": "review_cache.duckdb",
        "table": "review_nlp_cache",
    },
}

# Spanish key phrases likely in consumer reviews (lowercase)
SPANISH_KEY_PHRASES = [
    "servicio", "envío", "envio", "entrega", "calidad", "devolución", "devolucion",
    "precio", "tiempo", "atención", "atencion", "cliente", "producto", "tamaño",
    "tamano", "color", "funciona", "mal", "bien", "rápido", "rapido", "lento",
    "paquete", "dañado", "danado", "reembolso", "garantía", "garantia", "recomendado",
    "defectuoso", "vendedor", "descripcion", "descripción", "original", "falso",
]

EMOJI_PATTERN = re.compile(r"[\U0001F300-\U0001F6FF\U0001F900-\U0001F9FF\u2600-\u26FF\u2700-\u27BF]")

POS_WORDS_ES = set([
    "excelente", "bueno", "buen", "genial", "perfecto", "recomendado", "encantado",
    "fantástico", "fantastico", "maravilloso", "satisfecho", "rápido", "rapido",
    "funciona", "cumple", "calidad", "vale", "barato", "confiable",
])
NEG_WORDS_ES = set([
    "malo", "peor", "horrible", "terrible", "defectuoso", "roto", "dañado", "danado",
    "lento", "tarde", "caro", "engañado", "enganado", "decepcionado", "decepción",
    "decepcion", "no funciona", "mal", "incompleto", "falso",
])

EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
URL_RE = re.compile(r"https?://\S+")
PHONE_RE = re.compile(r"(?:(?:\+\d{1,3}[\s-]?)?\(?\d{3}\)?[\s-]?\d{3}[\s-]?\d{4})")

# -----------------------------
# Utilities
# -----------------------------

def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def scrub_pii(text: str) -> str:
    if not isinstance(text, str):
        return text
    text = EMAIL_RE.sub("<email>", text)
    text = URL_RE.sub("<url>", text)
    text = PHONE_RE.sub("<phone>", text)
    return text


def looks_spanish(text: str) -> float:
    """Lightweight Spanish likelihood score (0-1) without extra deps.
    Heuristic: count Spanish keywords/diacritics vs tokens.
    """
    if not isinstance(text, str) or not text:
        return 0.0
    t = text.lower()
    hits = 0
    # diacritics and common words
    if any(c in t for c in ["á", "é", "í", "ó", "ú", "ñ"]):
        hits += 1
    for w in ["el", "la", "los", "las", "de", "para", "con", "sin", "muy", "no", "sí", "si"]:
        if f" {w} " in f" {t} ":
            hits += 1
    # key phrases presence
    for kw in SPANISH_KEY_PHRASES[:10]:
        if kw in t:
            hits += 1
    # normalize
    tokens = max(1, len(t.split()))
    return min(1.0, hits / min(tokens, 20))


# -----------------------------
# Column ranking (Spanish-aware)
# -----------------------------

def score_text_column(series: pd.Series) -> float:
    """Score how likely a column holds Spanish review text.
    Combines: key phrase hits, emoji frequency, avg length, and Spanish-likelihood.
    """
    try:
        s = series.dropna().astype(str).sample(min(500, len(series)), random_state=42)
    except Exception:
        s = series.dropna().astype(str)
    if len(s) == 0:
        return 0.0

    s_lower = s.str.lower()
    # key phrase hits per row
    def key_hits(text: str) -> int:
        return sum(1 for kw in SPANISH_KEY_PHRASES if kw in text)

    key_score = s_lower.apply(key_hits).mean()
    emoji_score = s_lower.apply(lambda x: len(EMOJI_PATTERN.findall(x))).mean()
    len_score = np.clip(s_lower.str.len().mean() / 200.0, 0, 1)  # prefer medium-long text
    es_score = s_lower.apply(looks_spanish).mean()

    # Weighted combo
    score = 0.45 * key_score + 0.25 * emoji_score + 0.20 * len_score + 0.10 * es_score
    return float(score)


def suggest_review_text_column(df: pd.DataFrame) -> Optional[str]:
    textlike = [c for c in df.columns if df[c].dtype == object]
    if not textlike:
        return None
    scores = {c: score_text_column(df[c]) for c in textlike}
    best = max(scores, key=scores.get)
    # Avoid obvious non-review columns by name
    if re.search(r"(descripcion|description|review|comentario|opinion|texto)", best, re.I):
        return best
    # sanity threshold; if too low, return None so user selects manually
    return best if scores[best] >= 0.4 else None


# -----------------------------
# Caching layer (Streamlit + DuckDB persistence)
# -----------------------------

def get_duck():
    cfg = DEFAULT_CONFIG["cache"]
    con = duckdb.connect(cfg["db_path"]) if cfg["use_duckdb"] else None
    if con:
        con.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {cfg['table']} (
              review_sha TEXT,
              lang TEXT,
              text_en TEXT,
              p_pos DOUBLE,
              p_neu DOUBLE,
              p_neg DOUBLE,
              label TEXT,
              model TEXT,
              pipeline_version TEXT,
              ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
              PRIMARY KEY (review_sha, model, pipeline_version)
            )
            """
        )
    return con


DUCK = get_duck()


def cache_lookup(review_sha: str, model: str, pipeline_version: str):
    if DUCK is None:
        return None
    cfg = DEFAULT_CONFIG["cache"]
    try:
        res = DUCK.execute(
            f"SELECT lang, text_en, p_pos, p_neu, p_neg, label FROM {cfg['table']} WHERE review_sha=? AND model=? AND pipeline_version=?",
            [review_sha, model, pipeline_version],
        ).fetchone()
        if res:
            lang, text_en, p_pos, p_neu, p_neg, label = res
            return {"lang": lang, "text_en": text_en, "p_pos": p_pos, "p_neu": p_neu, "p_neg": p_neg, "label": label}
    except Exception:
        pass
    return None


def cache_write(review_sha: str, model: str, pipeline_version: str, row: Dict[str, Any]):
    if DUCK is None:
        return
    cfg = DEFAULT_CONFIG["cache"]
    try:
        DUCK.execute(
            f"""
            INSERT OR REPLACE INTO {cfg['table']}
            (review_sha, lang, text_en, p_pos, p_neu, p_neg, label, model, pipeline_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [review_sha, row.get("lang"), row.get("text_en"), row.get("p_pos"), row.get("p_neu"), row.get("p_neg"), row.get("label"), model, pipeline_version],
        )
    except Exception:
        pass


# -----------------------------
# NLP providers (pluggable)
# -----------------------------

def openai_translate(texts: List[str], target_lang: str = "en") -> List[str]:
    """Stub for OpenAI translation. Replace with your provider call.
    If no API key, returns original text.
    """
    if not os.getenv("OPENAI_API_KEY"):
        return texts
    # Implement your actual translation here.
    return texts


def openai_sentiment_probs(texts: List[str]) -> List[Dict[str, float]]:
    """Stub for OpenAI sentiment returning probabilities.
    If no API key, falls back to rule-based.
    """
    if not os.getenv("OPENAI_API_KEY"):
        return rule_based_sentiment_probs(texts)
    # Implement your actual LLM call.
    # Expected return per item: {"p_pos": float, "p_neu": float, "p_neg": float}
    # For demo, fallback to rule-based even when key exists.
    return rule_based_sentiment_probs(texts)


def rule_based_sentiment_probs(texts: List[str]) -> List[Dict[str, float]]:
    out = []
    for t in texts:
        tl = str(t).lower()
        pos = sum(1 for w in POS_WORDS_ES if w in tl)
        neg = sum(1 for w in NEG_WORDS_ES if w in tl)
        # crude neutral bias
        if pos == 0 and neg == 0:
            out.append({"p_pos": 0.15, "p_neu": 0.7, "p_neg": 0.15})
        else:
            total = max(1, pos + neg)
            p_pos = 0.2 + 0.8 * (pos / total)
            p_neg = 0.2 + 0.8 * (neg / total)
            # normalize to sum to <= 1; keep some neutral mass
            s = p_pos + p_neg
            if s >= 0.9:
                p_neu = 0.1
            else:
                p_neu = 1.0 - s
            out.append({"p_pos": float(p_pos), "p_neu": float(p_neu), "p_neg": float(p_neg)})
    return out


# -----------------------------
# Review pipeline (detect → translate → sentiment → aggregate)
# -----------------------------

@st.cache_data(show_spinner=False)
def detect_language_series(texts: List[str], mode: str) -> List[str]:
    langs = []
    for t in texts:
        if mode == "never":
            langs.append("es")  # assume Spanish to keep pipeline simple
        elif mode == "always":
            langs.append("es")
        else:
            # auto: heuristic
            langs.append("es" if looks_spanish(str(t)) >= 0.25 else "en")
    return langs


@st.cache_data(show_spinner=False)
def translate_if_needed(texts: List[str], langs: List[str], mode: str) -> List[str]:
    if mode == "never":
        return texts
    needs = [i for i, lg in enumerate(langs) if lg != "en"]
    if not needs:
        return texts
    src = [texts[i] for i in needs]
    # Provider call
    tr = openai_translate(src, target_lang="en") if mode in {"auto", "always"} else src
    out = list(texts)
    for idx, s in zip(needs, tr):
        out[idx] = s
    return out


@st.cache_data(show_spinner=False)
def sentiment_scores(texts_en: List[str], provider: str) -> List[Dict[str, float]]:
    if provider == "openai":
        return openai_sentiment_probs(texts_en)
    return rule_based_sentiment_probs(texts_en)


def process_reviews(df: pd.DataFrame, product_col: str, text_col: str, cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    model = cfg["nlp"]["model"]
    pver = cfg["nlp"]["pipeline_version"]
    lang_mode = cfg["nlp"]["lang_mode"]
    s_provider = cfg["nlp"]["sentiment_provider"]

    # Prepare rows
    texts = df[text_col].astype(str).apply(scrub_pii).tolist()
    shas = [sha256_text(t) for t in texts]

    # Fetch from cache where possible
    cached_rows = [cache_lookup(h, model, pver) for h in shas]
    hit = [i for i, r in enumerate(cached_rows) if r is not None]
    miss = [i for i, r in enumerate(cached_rows) if r is None]

    lang_list = [cached_rows[i]["lang"] if i in hit else None for i in range(len(texts))]
    text_en = [cached_rows[i]["text_en"] if i in hit else None for i in range(len(texts))]
    p_pos = [cached_rows[i]["p_pos"] if i in hit else None for i in range(len(texts))]
    p_neu = [cached_rows[i]["p_neu"] if i in hit else None for i in range(len(texts))]
    p_neg = [cached_rows[i]["p_neg"] if i in hit else None for i in range(len(texts))]

    # Compute for misses
    if miss:
        miss_texts = [texts[i] for i in miss]
        langs = detect_language_series(miss_texts, lang_mode)
        tr_texts = translate_if_needed(miss_texts, langs, lang_mode)
        probs = sentiment_scores(tr_texts, s_provider)
        for j, i in enumerate(miss):
            lang_list[i] = langs[j]
            text_en[i] = tr_texts[j]
            p_pos[i] = probs[j]["p_pos"]
            p_neu[i] = probs[j]["p_neu"]
            p_neg[i] = probs[j]["p_neg"]
            cache_write(shas[i], model, pver, {
                "lang": lang_list[i],
                "text_en": text_en[i],
                "p_pos": p_pos[i],
                "p_neu": p_neu[i],
                "p_neg": p_neg[i],
                "label": None,
            })

    out = df[[product_col, text_col]].copy()
    out["lang"] = lang_list
    out["text_en"] = text_en
    out["p_pos"] = p_pos
    out["p_neu"] = p_neu
    out["p_neg"] = p_neg

    stats = {
        "cache_hits": len(hit),
        "cache_misses": len(miss),
        "model": model,
        "pipeline_version": pver,
    }
    return out, stats


# -----------------------------
# Stratified sampling and aggregation
# -----------------------------

def stratified_sample(df: pd.DataFrame, by: str, cap: int, random_state: int = 42) -> pd.DataFrame:
    if by not in df.columns:
        return df.sample(min(len(df), cap), random_state=random_state)
    return (df.groupby(by, group_keys=False)
              .apply(lambda g: g.sample(min(len(g), cap), random_state=random_state)))


def aggregate_positive_share(scored: pd.DataFrame, product_col: str, threshold: float) -> pd.DataFrame:
    agg = scored.groupby(product_col).agg(
        reviews=("p_pos", "size"),
        pos_est=("p_pos", "mean"),
        pos_count=("p_pos", lambda s: int((s >= threshold).sum())),
    ).reset_index()
    agg["positive_share"] = agg["pos_count"] / agg["reviews"].clip(lower=1)
    agg = agg.sort_values(["positive_share", "pos_est", "reviews"], ascending=[False, False, False])
    return agg


# -----------------------------
# Minimal AM/DS action dispatcher (no hidden bypass)
# -----------------------------
ALLOWED_ACTIONS = {
    "aggregate_reviews_from_text": "Aggregate review sentiment by product",
}

@dataclass
class DSPlan:
    action: str
    params: Dict[str, Any]

    @staticmethod
    def from_json(js: Dict[str, Any]) -> "DSPlan":
        if not isinstance(js, dict):
            raise ValueError("DS output must be a JSON object")
        action = js.get("action")
        params = js.get("params", {})
        if action not in ALLOWED_ACTIONS:
            raise ValueError(f"Action '{action}' not allowed")
        if not isinstance(params, dict):
            raise ValueError("params must be an object")
        return DSPlan(action=action, params=params)


def dispatch(plan: DSPlan, state: Dict[str, Any]):
    if plan.action == "aggregate_reviews_from_text":
        return run_review_aggregation(state)
    raise ValueError("Unhandled action")


# -----------------------------
# Streamlit app
# -----------------------------

def load_data() -> pd.DataFrame:
    # Replace with your actual data loader. For demo, create a small frame.
    data = {
        "product_id": ["A", "A", "B", "B", "B"],
        "comentario": [
            "El servicio fue excelente y la entrega rápida 😄",
            "La calidad del producto es buena, recomendado",
            "El envío llegó tarde y el paquete dañado",
            "Precio caro y atención al cliente lenta",
            "Funciona perfecto, muy satisfecho",
        ],
    }
    return pd.DataFrame(data)


def run_review_aggregation(state: Dict[str, Any]):
    cfg = state["config"]
    df = state["df"]

    product_col = state["product_col"]
    text_col = state["text_col"]

    # Sampling
    sm = cfg["sampling"]
    if sm["mode"] == "per_product_cap":
        df_work = stratified_sample(df, product_col, sm["cap_per_product"], sm["random_state"])
    else:
        df_work = df.sample(min(len(df), sm["global_max_reviews"]), random_state=sm["random_state"])

    # Process
    scored, stats = process_reviews(df_work, product_col, text_col, cfg)

    # Aggregate
    threshold = cfg["nlp"]["positive_threshold"]
    agg = aggregate_positive_share(scored, product_col, threshold)

    # UI render
    st.subheader("Top products by positive share")
    st.dataframe(agg, use_container_width=True)

    with st.expander("Sampled scored reviews"):
        st.dataframe(scored.head(200), use_container_width=True)

    st.caption(
        f"Processed {len(scored)} reviews | cache hits {stats['cache_hits']} / misses {stats['cache_misses']} | model {stats['model']} | pipeline {stats['pipeline_version']}"
    )

    return {"agg": agg, "scored": scored}


def main():
    st.set_page_config(page_title="Review NLP (Spanish-aware)", layout="wide")
    st.title("Review NLP (Spanish-aware, pluggable, cached)")

    # Load config (could be extended to read from YAML)
    cfg = DEFAULT_CONFIG.copy()

    # Sidebar controls
    st.sidebar.header("Settings")

    # Language behavior
    cfg["nlp"]["lang_mode"] = st.sidebar.selectbox(
        "Language handling", ["auto", "always", "never"], index=["auto", "always", "never"].index(cfg["nlp"]["lang_mode"]),
        help="Auto-detect Spanish and translate to English as needed; Always = assume non-English → translate; Never = no translation",
    )
    cfg["nlp"]["sentiment_provider"] = st.sidebar.selectbox(
        "Sentiment provider", ["openai", "rule_based"], index=["openai", "rule_based"].index(cfg["nlp"]["sentiment_provider"]),
    )
    cfg["nlp"]["positive_threshold"] = st.sidebar.slider(
        "Positive threshold (p_pos)", 0.5, 0.95, float(cfg["nlp"]["positive_threshold"]), 0.01,
    )

    # Sampling
    cfg["sampling"]["mode"] = st.sidebar.selectbox("Sampling mode", ["per_product_cap", "global"], index=0)
    if cfg["sampling"]["mode"] == "per_product_cap":
        cfg["sampling"]["cap_per_product"] = st.sidebar.number_input("Cap per product", 10, 1000, cfg["sampling"]["cap_per_product"], 10)
    else:
        cfg["sampling"]["global_max_reviews"] = st.sidebar.number_input("Global max reviews", 100, 100000, cfg["sampling"]["global_max_reviews"], 100)

    # Data
    df = load_data()

    # Column selection
    product_col_guess = DEFAULT_CONFIG["columns"]["product_id"] if DEFAULT_CONFIG["columns"]["product_id"] in df.columns else st.sidebar.selectbox("Product column", list(df.columns))

    text_col_guess = suggest_review_text_column(df) if DEFAULT_CONFIG["columns"]["review_text"] == "auto" else DEFAULT_CONFIG["columns"]["review_text"]
    text_col = st.sidebar.selectbox("Review text column", [text_col_guess] + [c for c in df.columns if c != text_col_guess]) if text_col_guess else st.sidebar.selectbox("Review text column", list(df.columns))

    st.write(
        f"**Using product column:** `{product_col_guess}`  |  **Review text column suggestion:** `{text_col}`"
    )

    # Plan (AM/DS): here we simulate DS plan creation; in your app, this would come from DS agent
    ds_plan_json = {
        "action": "aggregate_reviews_from_text",
        "params": {"note": "Aggregate Spanish reviews by product with probability-based sentiment."},
    }

    # Validate DS output
    try:
        plan = DSPlan.from_json(ds_plan_json)
    except Exception as e:
        st.error(f"Invalid DS plan: {e}")
        return

    # Dispatch
    state = {
        "config": cfg,
        "df": df,
        "product_col": product_col_guess,
        "text_col": text_col,
    }

    result = dispatch(plan, state)

    st.success("Pipeline complete.")


if __name__ == "__main__":
    main()
