"""
Main application entry point - simplified and modular version.
"""
import streamlit as st
import pandas as pd
from typing import Dict, Any

# Import our modular components
from config import init_session_state
from database import get_all_tables, run_duckdb_sql, cache_result_with_approval
from agents import run_am_plan, run_ds_step, judge_review, am_review
from ui import add_msg, render_chat, render_final_for_action

def build_shared_context() -> Dict[str, Any]:
    """Build shared context for agents - simplified version."""
    return {
        "tables": list(get_all_tables().keys()),
        "recent_sql_results": {},
        "key_findings": {"product_id": "bb50f2e236e5eea0100680137654686c"},
        "referenced_entities": {"product_id": "bb50f2e236e5eea0100680137654686c"},
        "context_relevance": {"question_type": "specific_entity_reference"},
        "schema_info": {
            "olist_products_dataset": {
                "columns": ["product_id", "product_category_name", "product_name_lenght", "product_description_lenght"]
            },
            "olist_order_items_dataset": {
                "columns": ["order_id", "product_id", "seller_id", "price", "freight_value"]
            },
            "olist_orders_dataset": {
                "columns": ["order_id", "customer_id", "order_status", "order_purchase_timestamp"]
            },
            "olist_customers_dataset": {
                "columns": ["customer_id", "customer_unique_id", "customer_city", "customer_state"]
            }
        }
    }

def assess_context_relevance(current_question: str, available_context: dict) -> dict:
    """Assess context relevance - simplified version."""
    return available_context

def build_column_hints(question: str) -> dict:
    """Build column hints for business terms."""
    return {
        "revenue": ["price", "payment_value"],
        "sales": ["price"],
        "customer": ["customer_id", "customer_unique_id"],
        "product": ["product_id"],
        "category": ["product_category_name"]
    }

def run_turn_ceo(new_text: str):
    """Main function to handle CEO questions - simplified version."""
    if not new_text.strip():
        return

    st.session_state.current_question = new_text
    thread_ctx = {"current_question": new_text}

    # Build context and column hints
    shared_context = build_shared_context()
    col_hints = build_column_hints(new_text)

    # 1) AM plans
    am_json = run_am_plan(new_text, col_hints, shared_context)
    add_msg("am", am_json.get("am_brief", ""), artifacts=am_json)

    # 2) DS executes
    ds_json = run_ds_step(am_json, col_hints, thread_ctx)

    # 3) Execute results and display
    if ds_json.get("action_sequence"):
        results_summary = []
        for step in ds_json.get("action_sequence")[:5]:
            # Cache each step result with approval
            action_id = f"{step.get('action', 'unknown')}_{len(st.session_state.executed_results)}"
            cache_result_with_approval(action_id, step, approved=True)

            # Execute and capture SQL results for summary
            if step.get("action") == "sql":
                sql = step.get("duckdb_sql")
                if sql:
                    try:
                        result_df = run_duckdb_sql(sql)
                        if len(result_df) > 0:
                            if "category" in sql.lower():
                                category = result_df.iloc[0]['product_category_name'] if 'product_category_name' in result_df.columns else "Unknown"
                                results_summary.append(f"Product category: {category}")
                            elif "customer" in sql.lower():
                                customer_id = result_df.iloc[0]['customer_id'] if 'customer_id' in result_df.columns else "Unknown"
                                total_spent = result_df.iloc[0].get('total_spent', 'N/A')
                                results_summary.append(f"Top customer: {customer_id} (spent: ${total_spent})")
                            else:
                                results_summary.append(f"SQL executed: {len(result_df)} rows returned")
                    except Exception as e:
                        results_summary.append(f"SQL execution error: {str(e)}")

            render_final_for_action(step)

        summary_text = "; ".join(results_summary) if results_summary else "Multi-step analysis completed"
        add_msg("system", f"✅ Analysis completed. {summary_text}")
    else:
        # Single action
        if ds_json.get("action"):
            action_id = f"{ds_json.get('action')}_{len(st.session_state.executed_results)}"
            cache_result_with_approval(action_id, ds_json, approved=True)
            render_final_for_action(ds_json)

    render_chat()

def load_if_needed():
    """Load data if needed - simplified version."""
    if st.session_state.tables is None:
        # For demo purposes, create mock data
        st.session_state.tables = {
            "olist_products_dataset": pd.DataFrame({
                "product_id": ["bb50f2e236e5eea0100680137654686c"],
                "product_category_name": ["beleza_saude"]
            }),
            "olist_order_items_dataset": pd.DataFrame({
                "order_id": ["order1"],
                "product_id": ["bb50f2e236e5eea0100680137654686c"],
                "price": [63.85]
            }),
            "olist_orders_dataset": pd.DataFrame({
                "order_id": ["order1"],
                "customer_id": ["5c9d09439a7815d2c59d2242d90b296c"]
            }),
            "olist_customers_dataset": pd.DataFrame({
                "customer_id": ["5c9d09439a7815d2c59d2242d90b296c"],
                "customer_city": ["sao paulo"],
                "customer_state": ["SP"]
            })
        }

def main():
    """Main Streamlit application."""
    st.set_page_config(page_title="AI Agent - Simplified", layout="wide")
    st.title("🤖 AI Business Analytics Agent")

    # Initialize session state
    init_session_state()
    load_if_needed()

    # Main input
    if prompt := st.chat_input("Ask your business question..."):
        run_turn_ceo(prompt)

    # Display chat
    render_chat()

if __name__ == "__main__":
    main()