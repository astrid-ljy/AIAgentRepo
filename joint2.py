import os
import json
import time
import pandas as pd
import duckdb
import streamlit as st

# ====== Cloud-friendly config ======
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
DEFAULT_MODEL  = st.secrets.get("OPENAI_MODEL",  os.getenv("OPENAI_MODEL",  "gpt-4o-mini"))

try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except Exception:
    _OPENAI_AVAILABLE = False

# ====== Agent system prompts ======
SYSTEM_BA = r"""
You are the Business Analyst (BA) Agent focused on FINANCIAL PERFORMANCE.
Your job:
- Clarify business intent and frame the analysis in financial terms (revenue, margin, growth, retention, CAC/LTV, ARPU, etc.).
- Propose a concrete analysis or KPI plan grounded in the actual CSV schema (no invented columns).
- If a concept is missing (e.g., Region), suggest the closest available column(s).

Output ONE ```json block with keys:
{
  "reasoning_summary": "brief high-level reasoning (no secrets)",
  "ba_spec": "plain-English plan: metrics, segments, time grain, filters, and any assumptions",
  "sql_feature_prep": "optional DuckDB SQL to prep data for DS (or '' if not needed)"
}
"""

SYSTEM_DS = r"""
You are the Data Scientist (DS) Agent focused on MODELING.
Your job:
- Read the BA spec + CSV schema. Decide the modeling approach (classification/regression/time-series/cohorts).
- Propose: target, candidate features, model family, validation plan, and measurable success criteria.
- If useful, also return a SQL snippet to build features/labels from the CSV (DuckDB; table name `data` only).
- Keep it executable and schema-safe. If something is impossible, say so and propose a nearest alternative.

Output ONE ```json block with keys:
{
  "reasoning_summary": "brief high-level reasoning (no secrets)",
  "model_plan": "succinct steps: target, features, model choice, validation, metrics",
  "sql_for_model": "DuckDB SQL to materialize features/labels; '' if not required",
  "quick_tip": "1-liner practical suggestion the user can try next"
}
"""

SYSTEM_REVIEW = r"""
You are the Reviewer/Coordinator Agent.
Task: Given the USER FEEDBACK and the prior BA+DS responses, produce a single revised instruction set that both agents should follow to reproduce a better answer.

Return ONE ```json block with:
{
  "revision_directive": "single concise paragraph with concrete changes for BA and DS"
}
"""

# ====== Streamlit page ======
st.set_page_config(page_title="BA↔DS Interactive Chat", layout="wide")
st.title("💬 Two-Agent Interactive Chat — BA (Finance) & DS (Modeling)")

with st.sidebar:
    st.header("⚙️ Setup")
    uploaded = st.file_uploader("Upload CSV", type=["csv"], accept_multiple_files=False)
    model = st.text_input("OpenAI model", value=DEFAULT_MODEL)
    api_key = st.text_input("OpenAI API Key", value=OPENAI_API_KEY, type="password")
    show_schema = st.checkbox("Show inferred schema", value=True)

    st.markdown("---")
    st.subheader("🔁 Reproduction")
    if "last_feedback" not in st.session_state:
        st.session_state.last_feedback = ""
    if "last_user_prompt" not in st.session_state:
        st.session_state.last_user_prompt = ""
    if "last_ba_json" not in st.session_state:
        st.session_state.last_ba_json = {}
    if "last_ds_json" not in st.session_state:
        st.session_state.last_ds_json = {}

# ====== Helpers ======
def ensure_openai():
    if not _OPENAI_AVAILABLE:
        raise RuntimeError("OpenAI SDK not installed. Add 'openai' to requirements.txt")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY (set in Streamlit Secrets or env).")
    return OpenAI(api_key=api_key)

def infer_schema(df: pd.DataFrame, sample_rows: int = 5) -> str:
    lines = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        sample_vals = df[col].dropna().unique()[:sample_rows]
        sample_list = ", ".join([repr(v)[:60] for v in sample_vals])
        lines.append(f"- {col} ({dtype})  samples: {sample_list}")
    return "\n".join(lines)

def llm_json(system_prompt: str, user_payload: str) -> dict:
    client = ensure_openai()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_payload + "\n\nReturn a single JSON object in a ```json fenced block."},
        ],
        temperature=0.1,
    )
    txt = resp.choices[0].message.content or ""
    import re
    m = re.search(r"```json\s*(\{.*?\})\s*```", txt, flags=re.DOTALL | re.IGNORECASE)
    jtxt = m.group(1).strip() if m else "{}"
    try:
        return json.loads(jtxt)
    except Exception:
        return {"_raw": txt, "_parse_error": True}

def run_duckdb(df: pd.DataFrame, sql: str):
    con = duckdb.connect(database=":memory:")
    con.register("data", df)
    return con.execute(sql).df()

def add_msg(role, content, artifacts=None):
    if "chat" not in st.session_state:
        st.session_state.chat = []
    st.session_state.chat.append({"role": role, "content": content, "artifacts": artifacts or {}})

def render_chat():
    if "chat" not in st.session_state:
        st.session_state.chat = []
    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.write(m["content"])
            if m["artifacts"]:
                with st.expander("Artifacts", expanded=False):
                    for k, v in m["artifacts"].items():
                        st.write(f"**{k}**")
                        if isinstance(v, str) and v.strip().upper().startswith("SELECT"):
                            st.code(v, language="sql")
                        else:
                            try:
                                st.json(v)
                            except Exception:
                                st.write(v)

# ====== Chat area ======
st.subheader("Chat")
render_chat()
user_prompt = st.chat_input("Ask a business question (BA will focus on finance; DS will propose modeling)…")

# ====== Main turn ======
if user_prompt:
    add_msg("user", user_prompt)
    st.session_state.last_user_prompt = user_prompt
    try:
        if uploaded is None:
            raise RuntimeError("Please upload a CSV in the sidebar.")
        # Re-read CSV each turn (schema freshness)
        df = pd.read_csv(uploaded)
        schema_txt = infer_schema(df)

        if show_schema:
            with st.expander("📜 Inferred Schema", expanded=False):
                st.code(schema_txt, language="markdown")

        # --- BA step ---
        with st.status("🧭 BA framing financial analysis…", state="running"):
            ba_user = f"USER QUESTION:\n{user_prompt}\n\nSCHEMA for table `data`:\n{schema_txt}"
            ba_json = llm_json(SYSTEM_BA, ba_user)
            st.session_state.last_ba_json = ba_json
            add_msg("assistant", "BA: Financial framing & spec", artifacts=ba_json)

        # Optional SQL feature prep from BA
        prep_sql = (ba_json.get("sql_feature_prep") or "").strip()
        if prep_sql:
            try:
                feat_df = run_duckdb(df, prep_sql)
                add_msg("assistant", "BA: Feature prep result (preview)", artifacts={"rows": len(feat_df), "preview": feat_df.head(20).to_dict(orient="records")})
            except Exception as e:
                add_msg("assistant", f"BA prep SQL failed: {e}", artifacts={"sql": prep_sql})

        # --- DS step ---
        with st.status("🧪 DS proposing modeling plan…", state="running"):
            ds_user = (
                f"BA SPEC:\n{ba_json.get('ba_spec','')}\n\n"
                f"SCHEMA for `data`:\n{schema_txt}\n\n"
                f"(Optional) BA feature-prep SQL:\n{prep_sql or '(none)'}"
            )
            ds_json = llm_json(SYSTEM_DS, ds_user)
            st.session_state.last_ds_json = ds_json
            add_msg("assistant", "DS: Modeling plan", artifacts=ds_json)

        # Optional DS SQL to materialize features/labels
        ds_sql = (ds_json.get("sql_for_model") or "").strip()
        if ds_sql:
            try:
                mdf = run_duckdb(df, ds_sql)
                add_msg("assistant", "DS: Feature/label table (preview)", artifacts={"rows": len(mdf), "preview": mdf.head(20).to_dict(orient="records")})
            except Exception as e:
                add_msg("assistant", f"DS feature SQL failed: {e}", artifacts={"sql": ds_sql})

        # Feedback UI
        with st.container():
            st.markdown("#### Feedback controls")
            col1, col2 = st.columns([1,3])
            with col1:
                up = st.button("👍 Helpful", key=f"up_{time.time()}")
                down = st.button("👎 Not helpful", key=f"down_{time.time()}")
            with col2:
                fb_note = st.text_input("Tell the agents what to change next:", key=f"fb_{time.time()}")

        if up or down or fb_note:
            st.session_state.last_feedback = (fb_note or "").strip()
            add_msg("user", f"Feedback — up={bool(up)}, down={bool(down)}, note={st.session_state.last_feedback or '(none)'}")

    except Exception as e:
        add_msg("assistant", f"Error: {type(e).__name__}: {e}")
        st.error(str(e))

# ====== Reproduce with feedback ======
st.divider()
st.subheader("🔁 Reproduce answer with your feedback")
if st.button("Re-run BA & DS using my feedback to improve the answer", type="primary", use_container_width=True):
    try:
        if uploaded is None:
            raise RuntimeError("Please upload a CSV in the sidebar.")
        if not st.session_state.last_user_prompt:
            raise RuntimeError("No previous user prompt to reproduce. Ask a question first.")
        df = pd.read_csv(uploaded)
        schema_txt = infer_schema(df)

        # Build a revision directive from feedback + prior BA/DS outputs
        reviewer_payload = json.dumps({
            "user_prompt": st.session_state.last_user_prompt,
            "feedback_note": st.session_state.last_feedback,
            "ba_json": st.session_state.last_ba_json,
            "ds_json": st.session_state.last_ds_json
        }, indent=2)
        rev = llm_json(SYSTEM_REVIEW, reviewer_payload)
        directive = rev.get("revision_directive", "")
        add_msg("assistant", "Coordinator: Revision directive", artifacts=rev)

        # Re-run BA with directive
        ba_user = (
            f"USER QUESTION (original):\n{st.session_state.last_user_prompt}\n\n"
            f"REVISION DIRECTIVE:\n{directive}\n\n"
            f"SCHEMA for table `data`:\n{schema_txt}"
        )
        ba_json = llm_json(SYSTEM_BA, ba_user)
        st.session_state.last_ba_json = ba_json
        add_msg("assistant", "BA (revised): Financial framing & spec", artifacts=ba_json)

        prep_sql = (ba_json.get("sql_feature_prep") or "").strip()
        if prep_sql:
            try:
                feat_df = run_duckdb(df, prep_sql)
                add_msg("assistant", "BA (revised): Feature prep result (preview)", artifacts={"rows": len(feat_df), "preview": feat_df.head(20).to_dict(orient="records")})
            except Exception as e:
                add_msg("assistant", f"BA (revised) prep SQL failed: {e}", artifacts={"sql": prep_sql})

        # Re-run DS with directive
        ds_user = (
            f"(REVISED) BA SPEC:\n{ba_json.get('ba_spec','')}\n\n"
            f"REVISION DIRECTIVE:\n{directive}\n\n"
            f"SCHEMA for `data`:\n{schema_txt}\n\n"
            f"(Optional) BA feature-prep SQL:\n{prep_sql or '(none)'}"
        )
        ds_json = llm_json(SYSTEM_DS, ds_user)
        st.session_state.last_ds_json = ds_json
        add_msg("assistant", "DS (revised): Modeling plan", artifacts=ds_json)

        ds_sql = (ds_json.get("sql_for_model") or "").strip()
        if ds_sql:
            try:
                mdf = run_duckdb(df, ds_sql)
                add_msg("assistant", "DS (revised): Feature/label table (preview)", artifacts={"rows": len(mdf), "preview": mdf.head(20).to_dict(orient="records")})
            except Exception as e:
                add_msg("assistant", f"DS (revised) feature SQL failed: {e}", artifacts={"sql": ds_sql})

    except Exception as e:
        add_msg("assistant", f"Error: {type(e).__name__}: {e}")
        st.error(str(e))

# ====== Footer tips ======
with st.expander("🛠️ Tips", expanded=False):
    st.write("- BA focuses on financial framing. DS focuses on modeling (target, features, model, validation).")
    st.write("- Each run **re-reads your CSV** to stay schema-accurate. Table is always named `data`.")
    st.write("- Use the feedback box to ask for changes (e.g., “use margin %, not revenue,” “predict churn in next 1 month”).")
