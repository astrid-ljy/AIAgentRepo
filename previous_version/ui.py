"""
User interface and rendering functionality.
"""
import streamlit as st
from typing import Any, Dict
from database import run_duckdb_sql, get_all_tables

def add_msg(role: str, content: str, artifacts: Any = None):
    """Add a message to the chat."""
    st.session_state.chat.append({
        "role": role,
        "content": content,
        "artifacts": artifacts
    })

def generate_artifact_summary(artifacts):
    """Generate a summary of artifacts for display."""
    if not artifacts:
        return ""

    # Handle different types of artifacts
    if "action" in artifacts:
        action = artifacts.get("action", "")
        if action == "sql":
            return "Analyst is running SQL queries to analyze data"
        elif action == "clustering":
            return "Data scientist is grouping similar data points together"
        elif action == "eda":
            return "Analyst is exploring the data to understand patterns"

    if "action_sequence" in artifacts:
        return "Data team is executing a multi-step analysis plan"

    if "sql" in artifacts:
        return f"Executed SQL query"

    return "Processing request"

def render_chat(incremental: bool = True):
    """Render the chat interface."""
    start_idx = st.session_state.last_rendered_idx if incremental else 0

    for i in range(start_idx, len(st.session_state.chat)):
        msg = st.session_state.chat[i]
        role = msg["role"]
        content = msg["content"]
        artifacts = msg.get("artifacts")

        with st.chat_message(role):
            if content:
                st.write(content)

            # Show artifact summary if available
            if artifacts:
                summary = generate_artifact_summary(artifacts)
                if summary:
                    st.caption(f"💡 {summary}")

    st.session_state.last_rendered_idx = len(st.session_state.chat)

def _sql_first(maybe_sql):
    """Get first SQL query from various formats."""
    if isinstance(maybe_sql, list):
        return maybe_sql[0] if maybe_sql else None
    return maybe_sql

def render_final_for_action(ds_step: dict):
    """Render the final results for a DS action."""
    action = (ds_step.get("action") or "").lower()

    # ---- OVERVIEW ----
    if action == "overview":
        st.markdown("### 📊 Table Previews (first 5 rows)")
        for name, df in get_all_tables().items():
            st.markdown(f"**{name}** — rows: {len(df)}, cols: {len(df.columns)}")
            st.dataframe(df.head(5), width="stretch")
        add_msg("ds", "Overview rendered.")
        return

    # ---- SQL ----
    if action == "sql":
        sql = _sql_first(ds_step.get("duckdb_sql"))
        if not sql:
            add_msg("ds", "No SQL provided.")
            return
        try:
            out = run_duckdb_sql(sql)
            st.markdown("### 🧮 SQL Results (first 25 rows)")
            st.code(sql, language="sql")
            st.dataframe(out.head(25), width="stretch")
            add_msg("ds", "SQL executed.", artifacts={"sql": sql})
            if "last_results" not in st.session_state:
                st.session_state.last_results = {}
            st.session_state.last_results["sql"] = {"sql": sql, "rows": len(out), "cols": list(out.columns)}
        except Exception as e:
            st.error(f"SQL failed: {e}")
        return

    # ---- EDA ----
    if action == "eda":
        raw_sql = ds_step.get("duckdb_sql")
        sql_list = raw_sql if isinstance(raw_sql, list) else [raw_sql]
        executed_sqls = []
        for i, sql in enumerate([_sql_first(s) for s in sql_list][:3]):
            if not sql:
                continue
            try:
                df = run_duckdb_sql(sql)
                executed_sqls.append(sql)
                st.markdown(f"### 📈 EDA Result #{i+1} (first 50 rows)")
                st.code(sql, language="sql")
                st.dataframe(df.head(50), width="stretch")
            except Exception as e:
                st.error(f"EDA SQL #{i+1} failed: {e}")
        add_msg("ds", f"EDA completed with {len(executed_sqls)} queries.")
        return

    # ---- CALC ----
    if action == "calc":
        st.markdown("### 🧮 Calculation")
        st.write(ds_step.get("calc_description", "(no description)"))
        add_msg("ds", "Calculation displayed.")
        return

    # ---- MODELING ----
    if action == "modeling":
        st.markdown("### 🤖 Machine Learning Analysis")
        plan = ds_step.get("model_plan", {})
        task = plan.get("task", "clustering")
        st.write(f"**Task**: {task}")
        if plan.get("target"):
            st.write(f"**Target**: {plan.get('target')}")
        if plan.get("features"):
            st.write(f"**Features**: {', '.join(plan.get('features', []))}")
        add_msg("ds", f"Machine learning analysis completed: {task}")
        return

    # ---- EXPLAIN ----
    if action == "explain":
        st.markdown("### 💬 Analysis Explanation")
        st.write(ds_step.get("explanation", "Analysis explanation not provided."))
        add_msg("ds", "Explanation provided.")
        return

    # ---- DEFAULT ----
    st.markdown(f"### ⚙️ Action: {action}")
    st.write(f"Action '{action}' completed.")
    add_msg("ds", f"Action '{action}' completed.")