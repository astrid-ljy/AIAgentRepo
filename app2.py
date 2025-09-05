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

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error, mean_squared_error, r2_score

# ============== OpenAI ==============
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
DEFAULT_MODEL  = st.secrets.get("OPENAI_MODEL",  os.getenv("OPENAI_MODEL",  "gpt-4o-mini"))
try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except Exception:
    _OPENAI_AVAILABLE = False

def ensure_openai():
    if not _OPENAI_AVAILABLE:
        raise RuntimeError("OpenAI SDK missing.")
    api = st.session_state.get("_api_key") or OPENAI_API_KEY
    if not api:
        raise RuntimeError("Missing OPENAI_API_KEY")
    return OpenAI(api_key=api)

# ============== UI ==============
st.set_page_config(page_title="CEO ↔ AM ↔ DS — Profit Assistant", layout="wide")
st.title("🏢 CEO ↔ AM ↔ DS — Profit Assistant")

with st.sidebar:
    st.header("⚙️ Data")
    zip_file = st.file_uploader("Upload ZIP of CSVs", type=["zip"])
    st.header("🧠 Model")
    model   = st.text_input("OpenAI model", value=DEFAULT_MODEL, key="_model")
    api_key = st.text_input("OPENAI_API_KEY", value=OPENAI_API_KEY, type="password", key="_api_key")

# ============== State ==============
if "tables_raw" not in st.session_state: st.session_state.tables_raw = {}
if "chat" not in st.session_state: st.session_state.chat = []

def add_msg(role, content, artifacts=None):
    st.session_state.chat.append({"role": role, "content": content, "artifacts": artifacts or {}})

def render_chat():
    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.write(m["content"])
            if m.get("artifacts"):
                with st.expander("Artifacts"):
                    st.json(m["artifacts"])

# ============== Data helpers ==============
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
    return dict(st.session_state.tables_raw)

def run_sql(sql: str) -> pd.DataFrame:
    con = duckdb.connect(database=":memory:")
    for name, df in get_all_tables().items():
        con.register(name, df)
    return con.execute(sql).df()

# ============== Profiling & feature proposal ==============
DIM_PAT = re.compile(r"product_(weight_g|length_cm|height_cm|width_cm)$", re.I)
PRODUCT_NAME_PAT = re.compile(r"\b(product|sku|item)\b", re.I)

def profile_columns(df: pd.DataFrame) -> pd.DataFrame:
    prof = []
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s): t = "numeric"
        elif np.issubdtype(s.dtype, np.datetime64): t = "datetime"
        else: t = "categorical"
        flags = []
        name = str(c).lower()
        if re.search(r"(^id$|_id$|uuid|guid|hash|^code$|_code$)", name): flags.append("id_like")
        if re.search(r"(zip|postal|cep|postcode|latitude|longitude|lat|lon)", name): flags.append("geo_like")
        if t == "datetime": flags.append("datetime")
        if (t == "numeric" and (s.std(skipna=True) == 0)) or (s.nunique(dropna=True) <= 1): flags.append("near_constant")
        if not pd.api.types.is_numeric_dtype(s) and s.astype(str).str.len().mean() > 40: flags.append("long_text")
        prof.append({"col": c, "t": t, "flags": flags})
    return pd.DataFrame(prof)

def propose_features_for_clustering(df: pd.DataFrame, allow_geo=False) -> List[str]:
    prof = profile_columns(df)
    bad = set()
    for _, r in prof.iterrows():
        col = r["col"]; flags = set(r["flags"])
        if "id_like" in flags or "near_constant" in flags or "datetime" in flags: bad.add(col)
        if "geo_like" in flags and not allow_geo: bad.add(col)
    feats = [c for c in df.columns if c not in bad and pd.api.types.is_numeric_dtype(df[c])]
    return feats

def choose_product_dimension_table(question: str) -> Tuple[str, List[str]]:
    """
    If the user asks for clustering and mentions product/dimension,
    pick the table with the most product_* dimension-like numeric columns.
    Otherwise pick the table with the most usable numeric features.
    """
    q = (question or "").lower()
    tables = get_all_tables()
    if not tables: return None, []

    # Product-dimension specific path
    if "cluster" in q or "clustering" in q or "segment" in q:
        if "product" in q or "dimension" in q:
            best = (None, -1, [])
            for name, df in tables.items():
                dim_cols = [c for c in df.columns if DIM_PAT.search(str(c)) and pd.api.types.is_numeric_dtype(df[c])]
                if len(dim_cols) > best[1]:
                    best = (name, len(dim_cols), dim_cols)
            if best[0] and best[1] >= 2:
                return best[0], best[2]

    # Generic: most usable numeric features
    best = (None, -1, [])
    for name, df in tables.items():
        feats = propose_features_for_clustering(df, allow_geo=False)
        if len(feats) > best[1]:
            best = (name, len(feats), feats)
    return (best[0], best[2]) if best[0] else (list(tables.keys())[0], propose_features_for_clustering(list(tables.values())[0]))

# ============== Clustering ==============
def run_kmeans_clustering(question: str, n_clusters: int = 5) -> Dict[str, Any]:
    tbl, feats = choose_product_dimension_table(question)
    if not tbl:
        return {"error": "No tables loaded."}
    df = get_all_tables()[tbl].copy()

    # Prefer explicit dimension columns when available
    dim_like = [c for c in df.columns if DIM_PAT.search(str(c)) and pd.api.types.is_numeric_dtype(df[c])]
    use_cols = dim_like if len(dim_like) >= 2 else [c for c in feats if pd.api.types.is_numeric_dtype(df[c])]
    if len(use_cols) < 2:
        return {"error": "Need at least 2 numeric features for clustering.", "table": tbl, "features": use_cols}

    X = df[use_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    X = X.dropna(how="all")
    if len(X) < 2:
        return {"error": "Not enough valid rows after cleaning to run clustering.", "table": tbl, "features": use_cols}

    imp = SimpleImputer(strategy="median")
    scl = StandardScaler()
    Xs = scl.fit_transform(imp.fit_transform(X))

    k = max(1, min(int(n_clusters), Xs.shape[0]))
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(Xs)

    try: sil = float(silhouette_score(Xs, labels)) if k > 1 else None
    except Exception: sil = None

    # PCA for plotting
    try:
        p = PCA(n_components=2, random_state=42)
        coords = p.fit_transform(Xs)
        coords_df = pd.DataFrame({"pc1": coords[:,0], "pc2": coords[:,1], "cluster": labels})
    except Exception:
        coords_df = None

    # Centroids back to original scale
    try:
        centers_std = km.cluster_centers_
        centers_orig = scl.inverse_transform(centers_std)
        centroids_df = pd.DataFrame(centers_orig, columns=use_cols)
    except Exception:
        centroids_df = None

    sizes = pd.Series(labels).value_counts().sort_index().to_dict()
    return {
        "table": tbl,
        "features": use_cols,
        "n_clusters": int(k),
        "cluster_sizes": {int(k_): int(v) for k_, v in sizes.items()},
        "inertia": float(getattr(km, "inertia_", np.nan)),
        "silhouette": sil,
        "labels": labels.tolist(),
        "pca": coords_df,
        "centroids": centroids_df,
    }

# ============== NLP: Spanish reviews (sentiment, topics, translation) ==============
SYSTEM_REVIEW_PICKER = """
You identify which table/column most likely contains review/comment text, given candidates.
Return JSON: {"table": str, "column": str, "language": "es"|"en"|..., "confidence": 0-1}
Only JSON, one object.
"""
SYSTEM_SPANISH_SENTIMENT = """
Eres un analista de reseñas en español. Recibes una lista de comentarios (reseñas) en español.
Para cada comentario devuelve un objeto JSON con:
- sentiment_label: "positivo" | "neutral" | "negativo"
- sentiment_score: número entre -1 y 1 (negativo a positivo)
- temas: lista de etiquetas de alto nivel entre ["producto","envio","entrega","calidad","precio","devolucion","atencion","vendedor","paquete","otros"]
- resumen_es: resumen corto en español (<= 20 palabras)
- traduccion_en: traducción concisa al inglés
Devuelve *solo* una lista JSON de objetos (sin texto adicional).
"""

def _sample_text_values(df: pd.DataFrame, col: str, k: int = 3) -> List[str]:
    try:
        vals = df[col].dropna().astype(str).tolist()
        return [v[:300] for v in vals[:k]]
    except Exception:
        return []

def propose_review_target() -> Optional[Tuple[str, str, str]]:
    tables = get_all_tables()
    if not tables: return None
    candidates = []
    for t, df in tables.items():
        prof = profile_columns(df)
        for _, r in prof.iterrows():
            if "long_text" in r["flags"]:
                samples = _sample_text_values(df, r["col"], 3)
                if samples:
                    candidates.append({"table": t, "column": r["col"], "samples": samples})
    if not candidates:
        return None

    try:
        client = ensure_openai()
        resp = client.chat.completions.create(
            model=st.session_state.get("_model") or DEFAULT_MODEL,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_REVIEW_PICKER},
                {"role": "user", "content": json.dumps({"candidates": candidates[:6]})},
            ],
            temperature=0,
        )
        data = json.loads(resp.choices[0].message.content)
        t = data.get("table"); c = data.get("column"); lang = data.get("language") or "es"
        if t in tables and c in tables[t].columns:
            return (t, c, lang)
    except Exception:
        pass
    first = candidates[0]
    return (first["table"], first["column"], "es")

def batch_spanish_sentiment_and_topics(rows: List[str]) -> List[dict]:
    try:
        client = ensure_openai()
        resp = client.chat.completions.create(
            model=st.session_state.get("_model") or DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_SPANISH_SENTIMENT},
                {"role": "user", "content": json.dumps({"comentarios": rows[:50]})},
            ],
            temperature=0,
        )
        data = json.loads(resp.choices[0].message.content)
        if isinstance(data, list):
            return data
    except Exception as e:
        return [{"error": str(e)}]
    return []

def run_reviews_nlp(limit: int = 400) -> Optional[pd.DataFrame]:
    pick = propose_review_target()
    if not pick:
        st.warning("No suitable review/comment column found.")
        return None
    table, column, lang = pick
    df = get_all_tables()[table]
    series = df[column].dropna().astype(str).head(limit)
    if series.empty:
        st.warning("Review column is empty.")
        return None
    results = batch_spanish_sentiment_and_topics(series.tolist())
    if not results or (isinstance(results, dict) and results.get("error")):
        st.error("NLP call failed.")
        return None
    out = pd.DataFrame(results)
    out.insert(0, "_text", series.values)
    out.insert(1, "_table", table)
    out.insert(2, "_column", column)
    out.insert(3, "_language", lang)
    return out

# ============== Simple AM/DS loop (lightweight) ==============
def handle_user_request(prompt: str):
    pl = prompt.lower().strip()

    # Load data first
    if not get_all_tables():
        add_msg("assistant", "Please upload a ZIP of CSVs first.")
        return

    # NLP path
    if any(w in pl for w in ["review", "reseña", "opinion", "comentario", "sentiment", "sentimiento", "keyword", "tema", "spanish", "español"]):
        add_msg("am", "Routing to NLP: Spanish review sentiment, topics, and translation.")
        df = run_reviews_nlp()
        if df is not None:
            add_msg("ds", "Spanish review analysis complete. Rendering first 10 rows:", artifacts={"preview": df.head(10).to_dict(orient="records")})
            st.dataframe(df.head(200))
        return

    # Clustering path
    if "cluster" in pl or "clustering" in pl or "segment" in pl:
        add_msg("am", "Clustering requested. Selecting table/columns. If product dimension, I will use product_* size/weight features.")
        res = run_kmeans_clustering(prompt, n_clusters=5)
        if "error" in res:
            add_msg("ds", f"Clustering failed: {res['error']}", artifacts=res)
        else:
            info = {k: v for k, v in res.items() if k in ("table","features","n_clusters","cluster_sizes","inertia","silhouette")}
            add_msg("ds", "Clustering done. See artifacts for details.", artifacts=info)
            if isinstance(res.get("pca"), pd.DataFrame):
                st.subheader("Cluster projection (PCA)")
                st.dataframe(res["pca"].head(200))
            if isinstance(res.get("centroids"), pd.DataFrame):
                st.subheader("Cluster centroids (original scale)")
                st.dataframe(res["centroids"])
        return

    # Otherwise: small SQL preview helper
    add_msg("am", "No clustering/NLP keywords detected. You can run SQL with `SELECT ...` or ask for clustering or review NLP.")
    if pl.strip().startswith("select"):
        try:
            out = run_sql(prompt)
            add_msg("ds", f"SQL executed. {len(out)} rows.", artifacts={"columns": list(out.columns)})
            st.dataframe(out.head(500))
        except Exception as e:
            add_msg("ds", f"SQL error: {e}")

# ============== Load data UI ==============
if zip_file:
    try:
        st.session_state.tables_raw = load_zip_tables(zip_file)
        st.success(f"Loaded {len(st.session_state.tables_raw)} tables: {', '.join(st.session_state.tables_raw.keys())}")
    except Exception as e:
        st.error(f"Failed to load ZIP: {e}")

# ============== Chat UI ==============
user_in = st.chat_input("Ask a question (e.g., 'Do a clustering analysis for product dimension', 'Analyze Spanish review sentiment')")
if user_in:
    add_msg("user", user_in)
    handle_user_request(user_in)

render_chat()
