import os
import json
import tempfile
from pathlib import Path

import streamlit as st
import pandas as pd
import duckdb

from agent_tools import (
    init_duckdb_with_df,
    schema_tool,
    plan_sql_from_nl,
    sql_tool,
    plot_tool,
    clean_telco_df,
)
from eda_utils import churn_rate, univariate_lift, num_compare, bin_tenure

st.set_page_config(page_title="Telco Churn Data Scientist Agent", layout="wide")

# ------------------------------
# Sidebar: File loader & Model
# ------------------------------
st.sidebar.header("Data & Settings")

# Model selection (default gpt-4o-mini)
model_default = "gpt-4o-mini"
model = st.sidebar.selectbox("OpenAI model", ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"], index=0)

# API key
api_key_env = os.getenv("OPENAI_API_KEY", "")
if not api_key_env:
    st.sidebar.warning("OPENAI_API_KEY not found in env. Set it before asking questions.")

uploaded = st.sidebar.file_uploader("Load IBM Telco CSV (WA_Fn-UseC_-Telco-Customer-Churn.csv)", type=["csv"])
path_text = st.sidebar.text_input("...or path to CSV", value="")

@st.cache_data(show_spinner=False)
def load_data(file_bytes: bytes | None, path: str | None):
    if file_bytes is None and not path:
        return None
    try:
        if file_bytes is not None:
            df = pd.read_csv(file_bytes)
        else:
            df = pd.read_csv(path)
        df = clean_telco_df(df)
        return df
    except Exception as e:
        st.sidebar.error(f"Failed to load CSV: {e}")
        return None

# Choose source
source = None
if uploaded is not None:
    source = uploaded
elif path_text:
    source = path_text

# Load
if source is None:
    st.info("Upload or specify a path to the IBM Telco churn CSV to begin.")
    st.stop()

if uploaded is not None:
    df = load_data(uploaded, None)
else:
    df = load_data(None, path_text)

if df is None:
    st.stop()

# ------------------------------
# DuckDB init & schema
# ------------------------------
con = init_duckdb_with_df(df)
schema, whitelist, category_values = schema_tool(con)

# ------------------------------
# Sidebar: Data Summary & Quick EDA
# ------------------------------
with st.sidebar.expander("Data Summary", expanded=True):
    rows, cols = df.shape
    st.write(f"**Rows:** {rows}  •  **Columns:** {cols}")
    # Target balance
    target_counts = df["Churn"].value_counts(dropna=False)
    pos = int(target_counts.get("Yes", 0))
    neg = int(target_counts.get("No", 0))
    total = pos + neg
    bal = (pos / total * 100) if total else 0
    st.write(f"**Churn Yes:** {pos} ({bal:.1f}%)  •  **No:** {neg} ({100-bal:.1f}%)")

    # Null counts
    nulls = df.isna().sum()
    nz = nulls[nulls > 0]
    if not nz.empty:
        st.write("**Null counts (non-zero):**")
        st.dataframe(nz.to_frame("nulls"))
    else:
        st.write("No nulls detected after cleaning.")

    # Top categories
    for col in ["Contract", "InternetService", "PaymentMethod"]:
        if col in df.columns:
            st.write(f"**Top {col}:**")
            st.dataframe(df[col].value_counts().head(5).to_frame("count"))

st.sidebar.markdown("---")
st.sidebar.subheader("Quick EDA")
if st.sidebar.button("Overall churn rate"):
    rate = churn_rate(df)
    st.success(f"Overall churn rate: **{rate*100:.2f}%**")

if st.sidebar.button("Churn by Contract & InternetService"):
    grp = churn_rate(df, by=["Contract", "InternetService"]).reset_index()
    st.dataframe(grp)

if st.sidebar.button("MonthlyCharges vs Churn (means)"):
    res = num_compare(df, "MonthlyCharges")
    st.dataframe(res)

if st.sidebar.button("Tenure distribution + churn by bins"):
    binned = bin_tenure(df)
    grp = churn_rate(binned, by=["tenure_bin"]).reset_index()
    st.dataframe(grp)

# ------------------------------
# Preset question chips
# ------------------------------
preset_qs = [
    "What’s the overall churn rate?",
    "Churn by contract and internet service?",
    "Which features have the strongest association with churn (univariate lift)?",
    "Do monthly charges differ between churned and non-churned?",
    "Plot churn rate by tenure bins.",
]
st.caption("Try a preset:")
cols = st.columns(len(preset_qs))
for i, q in enumerate(preset_qs):
    if cols[i].button(q, key=f"preset_{i}"):
        st.session_state.setdefault("preset", q)

# ------------------------------
# Chat state
# ------------------------------
if "history" not in st.session_state:
    st.session_state.history = []
if "last_df" not in st.session_state:
    st.session_state.last_df = None
if "last_full_df" not in st.session_state:
    st.session_state.last_full_df = None
if "preset" in st.session_state:
    default_q = st.session_state.preset
else:
    default_q = "Ask a question about churn…"

st.title("📉 Telco Churn Data Scientist Agent")

user_q = st.text_input("Your question", value=default_q)
ask = st.button("Ask")

# Download last result
if st.session_state.last_full_df is not None and not st.session_state.last_full_df.empty:
    csv_bytes = st.session_state.last_full_df.to_csv(index=False).encode()
    st.download_button("Download last result (CSV)", data=csv_bytes, file_name="result.csv")

# ------------------------------
# Handle question
# ------------------------------
if ask and user_q and user_q != default_q:
    # Plan SQL/plot via LLM, with strong constraints
    try:
        plan = plan_sql_from_nl(
            question=user_q,
            schema=schema,
            whitelist=whitelist,
            model=model or model_default,
        )
    except Exception as e:
        st.error(f"Could not plan query: {e}")
        st.stop()

    # Execute SQL if present and valid
    df_preview = None
    sql_str = plan.get("sql") or ""
    columns_used = plan.get("columns_used") or []
    needs_plot = bool(plan.get("needs_plot"))
    plot_spec = plan.get("plot") or {}

    # Guardrails: ensure columns are whitelisted and SQL is SELECT-only
    try:
        df_result, df_preview, rowcount = sql_tool(con, sql_str, whitelist)
        st.session_state.last_full_df = df_result
        st.session_state.last_df = df_preview
    except Exception as e:
        # If the question is clearly unrelated, tell the user gently
        unrelated = "CEO" in user_q or "weather" in user_q or "stock" in user_q or "who is" in user_q.lower()
        if unrelated:
            st.warning("This agent only answers questions using the loaded Telco churn dataset. Try asking about churn rates, contracts, payments, tenure, or charges.")
        else:
            st.error(f"SQL validation/execution error: {e}")
        st.stop()

    # Optional plot
    plot_path = None
    if needs_plot and df_preview is not None and not df_preview.empty:
        try:
            plot_path = plot_tool(
                kind=plot_spec.get("kind"),
                df=df_preview,
                x=plot_spec.get("x"),
                y=plot_spec.get("y"),
                agg=plot_spec.get("agg"),
                bins=plot_spec.get("bins"),
                title=plot_spec.get("title"),
            )
        except Exception as e:
            st.info(f"Plot skipped: {e}")

    # Build response blocks per system rules
    with st.container():
        st.subheader("TL;DR")
        # Create a compact automatic TL;DR from the result
        tldr = ""
        if "churn" in sql_str.lower() and rowcount:
            # naive extraction if there is a rate column
            rate_cols = [c for c in df_preview.columns if "rate" in c.lower()]
            if rate_cols:
                try:
                    r = float(df_preview[rate_cols[0]].iloc[0])
                    tldr = f"Estimated churn rate around {r*100:.1f}% for the selected slice."
                except Exception:
                    tldr = f"Returned {rowcount} rows. See details below."
            else:
                tldr = f"Returned {rowcount} rows. See details below."
        else:
            tldr = f"Returned {rowcount} rows. See details below."
        st.write(tldr)

        st.subheader("Details")
        # Minimal bullet summary (static for now; could add small templates)
        bullets = [
            f"Rows returned: **{rowcount}**",
            f"Columns referenced: {', '.join(columns_used) if columns_used else 'N/A'}",
            "Preview shown below (first ≤15 rows).",
        ]
        st.write("\n".join([f"- {b}" for b in bullets]))

        st.subheader("Evidence")
        st.markdown("**SQL used**")
        st.code(sql_str or "<no sql>", language="sql")
        if df_preview is not None:
            st.dataframe(df_preview)
        if plot_path and Path(plot_path).exists():
            st.image(plot_path, use_column_width=True)

        st.caption("Columns referenced: " + (", ".join(columns_used) if columns_used else "N/A"))

# ------------------------------
# Footer note
# ------------------------------
st.caption("This agent answers strictly from the currently loaded Telco CSV. Read-only. No external data or browsing.")