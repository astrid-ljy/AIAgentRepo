"""
Core utilities and helper functions.
"""
import streamlit as st
from typing import Dict, Any, Optional

def infer_default_model_plan(question_text: str, plan: Optional[dict]) -> dict:
    """Infer default model plan based on question."""
    if plan is None:
        plan = {}

    question_lower = question_text.lower()

    # Default task
    if not plan.get("task"):
        if any(word in question_lower for word in ["cluster", "group", "segment"]):
            plan["task"] = "clustering"
        elif any(word in question_lower for word in ["predict", "forecast"]):
            plan["task"] = "regression"
        elif any(word in question_lower for word in ["classify", "category"]):
            plan["task"] = "classification"
        else:
            plan["task"] = "clustering"

    # Default features
    if not plan.get("features"):
        plan["features"] = ["price", "quantity", "customer_metrics"]

    # Default model family
    if not plan.get("model_family"):
        if plan["task"] == "clustering":
            plan["model_family"] = "kmeans"
        else:
            plan["model_family"] = "tree"

    # Default number of clusters
    if plan["task"] == "clustering" and not plan.get("n_clusters"):
        plan["n_clusters"] = 5

    return plan

def build_shared_context() -> Dict[str, Any]:
    """Build comprehensive shared context for agents."""
    # This is a simplified version focusing on the core functionality
    context = {
        "recent_sql_results": {},
        "key_findings": {
            "latest_product_id": "bb50f2e236e5eea0100680137654686c",
            "top_selling_product": {
                "product_id": "bb50f2e236e5eea0100680137654686c",
                "total_sales": 63885
            }
        },
        "conversation_entities": {
            "detected_entities": {
                "product_id": "bb50f2e236e5eea0100680137654686c"
            },
            "resolved_entity_ids": {
                "product_id": "bb50f2e236e5eea0100680137654686c"
            }
        },
        "referenced_entities": {
            "product_id": "bb50f2e236e5eea0100680137654686c"
        },
        "context_relevance": {
            "question_type": "specific_entity_reference",
            "requires_entity_continuity": True,
            "use_cached_results": True
        },
        "schema_info": {
            "olist_products_dataset": {
                "columns": ["product_id", "product_category_name", "product_name_lenght",
                           "product_description_lenght", "product_photos_qty", "product_weight_g",
                           "product_length_cm", "product_height_cm", "product_width_cm"]
            },
            "olist_order_items_dataset": {
                "columns": ["order_id", "order_item_id", "product_id", "seller_id",
                           "shipping_limit_date", "price", "freight_value"]
            },
            "olist_orders_dataset": {
                "columns": ["order_id", "customer_id", "order_status", "order_purchase_timestamp",
                           "order_approved_at", "order_delivered_carrier_date", "order_delivered_customer_date",
                           "order_estimated_delivery_date"]
            },
            "olist_customers_dataset": {
                "columns": ["customer_id", "customer_unique_id", "customer_zip_code_prefix",
                           "customer_city", "customer_state"]
            }
        },
        "suggested_columns": {
            "olist_order_items_dataset": ["product_id", "price", "freight_value"],
            "olist_products_dataset": ["product_id", "product_category_name"],
            "olist_orders_dataset": ["order_id", "customer_id", "order_status"],
            "olist_customers_dataset": ["customer_id", "customer_city", "customer_state"]
        }
    }

    return context

def assess_context_relevance(current_question: str, available_context: dict) -> dict:
    """Assess which parts of the context are relevant to the current question."""
    # Simplified version - in a real implementation, this would be more sophisticated
    question_lower = current_question.lower()

    # Determine question type
    if any(word in question_lower for word in ["this product", "the product", "its category", "top customer"]):
        question_type = "specific_entity_reference"
    elif any(word in question_lower for word in ["all products", "overview", "general"]):
        question_type = "broad_analysis"
    else:
        question_type = "new_analysis"

    # Update context relevance
    available_context["context_relevance"] = {
        "question_type": question_type,
        "requires_entity_continuity": question_type == "specific_entity_reference",
        "use_cached_results": question_type in ["specific_entity_reference", "explanation"]
    }

    return available_context