import json
import re
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from openai import OpenAI

ALLOWED_SQL_KEYWORDS = {
    "SELECT", "FROM", "WHERE", "GROUP", "BY", "HAVING", "ORDER", "LIMIT", "WITH", "AS",
}

# ------------------------------
# Data cleaning
# ------------------------------
def clean_telco_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Trim string columns
    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = df[c].astype(str).str.strip()
    # Cast TotalCharges to numeric with coercion
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    # SeniorCitizen 0/1 to Yes/No
    if "SeniorCitizen" in df.columns and pd.api.types.is_integer_dtype(df["SeniorCitizen"]) or set(df["SeniorCitizen"].unique()) <= {0,1}:
        df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"}).fillna(df["SeniorCitizen"])
    # Standardize category names (lowercase -> title)
    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = df[c].astype(str).str.replace("_", " ").str.replace("-", "-")
    return df

# ------------------------------
# DuckDB init and schema
# ------------------------------

def init_duckdb_with_df(df: pd.DataFrame):
    con = duckdb.connect(database=':memory:')
    con.register('telco_df', df)
    # Create a persistent view name 'telco'
    con.execute("CREATE OR REPLACE VIEW telco AS SELECT * FROM telco_df")
    return con


def schema_tool(con) -> Tuple[Dict[str, str], List[str], Dict[str, List[str]]]:
    """Return schema mapping {column: type}, whitelist list, and limited unique category values."""
    info = con.execute("PRAGMA table_info('telco')").fetchdf()
    schema = {row["name"]: row["type"] for _, row in info.iterrows()}
    whitelist = list(schema.keys())
    # Unique values for categoricals (cap 30)
    category_values = {}
    for col, typ in schema.items():
        if typ.upper().startswith("VARCHAR") or typ.upper() in {"TEXT", "STRING"}:
            try:
                vals = con.execute(f"SELECT DISTINCT {col} FROM telco LIMIT 1000").fetchdf()[col].dropna().astype(str)
                vals = sorted(vals.unique().tolist())[:30]
                category_values[col] = vals
            except Exception:
                pass
    return schema, whitelist, category_values

# ------------------------------
# SQL validation & execution
# ------------------------------

def _is_select_only(sql: str) -> bool:
    s = re.sub(r"--.*?$|/\*.*?\*/", " ", sql, flags=re.S | re.M).strip()
    # Disallow semicolon chaining
    if ";" in s:
        return False
    # Only allow keywords set
    tokens = re.findall(r"[A-Za-z_]+", s)
    upper = [t.upper() for t in tokens]
    # Ensure SELECT appears first
    if "SELECT" not in upper:
        return False
    try:
        first_kw = next(t for t in upper if t in ALLOWED_SQL_KEYWORDS)
        if first_kw != "SELECT":
            return False
    except StopIteration:
        return False
    # No forbidden words
    forbidden = {"INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "ATTACH", "COPY", "EXPORT", "IMPORT"}
    if any(w in upper for w in forbidden):
        return False
    return True


def _columns_in_sql(sql: str) -> List[str]:
    # Extremely conservative: collect words that match identifier pattern and compare to whitelist
    # This is a heuristic; we also require plan provides columns_used.
    words = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", sql))
    # Remove keywords
    keywords = {k.lower() for k in list(ALLOWED_SQL_KEYWORDS) + [
        "and", "or", "not", "as", "on", "case", "when", "then", "end", "over", "partition", "count", "sum",
        "avg", "min", "max", "distinct", "between", "like", "in", "is", "null", "true", "false",
        "select", "from", "where", "group", "by", "having", "order", "limit", "with"
    ]}
    return [w for w in words if w.lower() not in keywords]


def sql_tool(con, sql: str, whitelist: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    if not sql:
        raise ValueError("Empty SQL plan.")
    if not _is_select_only(sql):
        raise ValueError("Only SELECT queries are allowed.")
    # Validate identifiers are whitelisted columns or known tables/aliases
    cols = _columns_in_sql(sql)
    for c in cols:
        if c in {"telco", "t"}:  # allowed table/alias
            continue
        if c not in whitelist and c.lower() not in [w.lower() for w in whitelist]:
            # might be a function; allow if looks like function name (has '(' in SQL around it). skip hard fail for common functions
            if re.search(rf"\b{re.escape(c)}\s*\(", sql):
                continue
            # numeric literal
            if re.match(r"^\d+$", c):
                continue
            raise ValueError(f"Unknown or disallowed identifier: {c}")

    df_full = con.execute(sql).fetchdf()
    rowcount = len(df_full)
    df_preview = df_full.head(15)
    return df_full, df_preview, rowcount

# ------------------------------
# Plotting
# ------------------------------

def plot_tool(kind: str, df: pd.DataFrame, x: str, y: Optional[str] = None, agg: Optional[str] = None,
              bins: Optional[List[int]] = None, title: Optional[str] = None) -> str:
    if kind not in {"bar", "line", "hist", "box", "pie"}:
        raise ValueError("Unsupported plot kind.")
    if x is None and kind not in {"hist", "box", "pie"}:
        raise ValueError("x is required for this plot kind.")

    fig = plt.figure()

    if kind == "bar":
        data = df
        if y and agg:
            grouped = data.groupby(x)[y].agg(agg)
            grouped.plot(kind="bar")
        elif y:
            grouped = data.groupby(x)[y].mean()
            grouped.plot(kind="bar")
        else:
            data[x].value_counts().plot(kind="bar")
    elif kind == "line":
        if y:
            df.plot(x=x, y=y)
        else:
            df[x].plot()
    elif kind == "hist":
        target = y or x
        df[target].plot(kind="hist", bins=bins)
    elif kind == "box":
        target = y or x
        df[[target]].plot(kind="box")
    elif kind == "pie":
        s = df[x].value_counts()
        s.plot(kind="pie", autopct="%.1f%%")

    if title:
        plt.title(title)

    tmpdir = tempfile.gettempdir()
    out_path = str(Path(tmpdir) / f"plot_{abs(hash((kind, x, y, title)))%1_000_000}.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path

# ------------------------------
# NL -> SQL planning via OpenAI (JSON-only tool output)
# ------------------------------

def plan_sql_from_nl(question: str, schema: dict, whitelist: list, model: str = "gpt-4o-mini") -> dict:
    system_path = Path("agent_system_prompt.txt")
    if system_path.exists():
        system_text = system_path.read_text(encoding="utf-8")
    else:
        system_text = "You are Telco Churn Data Scientist Agent."

    client = OpenAI()

    # Compose planning content: include schema & whitelist
    content = [
        {"type": "text", "text": system_text},
        {"type": "text", "text": "Schema columns and types:"},
        {"type": "text", "text": json.dumps(schema, indent=2)},
        {"type": "text", "text": "Column whitelist:"},
        {"type": "text", "text": json.dumps(whitelist)},
        {"type": "text", "text": (
            "Return ONLY a strict JSON object with keys: sql, columns_used (list), "
            "needs_plot (bool), plot (object: kind, x, y, agg, bins, title). "
            "SQL must be read-only over table telco, SELECT-only. Use safe columns only."
        )},
    ]

    # Force JSON output using response_format
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_text},
            {"role": "user", "content": [
                {"type": "text", "text": (
                    "Given the telco schema above, plan ONE SQL query (SELECT-only) that best answers: "
                    f"{question}\n"
                    "If SQL is unnecessary (e.g., a pure EDA helper), you may still return a simple SELECT that produces the needed frame for plotting."
                )}
            ]},
        ],
    )

    raw = resp.choices[0].message.content
    try:
        plan = json.loads(raw)
    except Exception as e:
        raise ValueError(f"Planner did not return JSON: {raw[:200]}...")

    # Basic validation of plan fields
    for k in ["sql", "columns_used", "needs_plot", "plot"]:
        if k not in plan:
            raise ValueError(f"Planner missing key: {k}")
    if not isinstance(plan.get("columns_used"), list):
        raise ValueError("columns_used must be a list")
    # Ensure only whitelisted columns
    bad = [c for c in plan["columns_used"] if c not in whitelist]
    if bad:
        raise ValueError(f"Planner referenced unknown columns: {bad}")

    return plan