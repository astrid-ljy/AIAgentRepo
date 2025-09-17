"""
Advanced SQL generation and fallback functionality - Complete version from original app.py
"""
import streamlit as st
from typing import Dict, Any, List

def generate_fallback_sql(user_question: str, shared_context: dict) -> str:
    """Generate flexible fallback SQL for any entity type when LLM fails."""
    question_lower = user_question.lower()

    # Get entity resolution data
    conversation_entities = shared_context.get("conversation_entities", {})
    detected_entities = conversation_entities.get("detected_entities", {})
    resolved_ids = conversation_entities.get("resolved_entity_ids", {})
    key_findings = shared_context.get("key_findings", {})

    # SQL Templates for different entity types and query patterns
    sql_templates = {
        # CUSTOMER queries
        "customer_top_products_frequency": """
            SELECT oi.product_id, COUNT(*) as purchase_frequency
            FROM olist_order_items_dataset oi
            JOIN olist_orders_dataset o ON oi.order_id = o.order_id
            WHERE o.customer_id = '{customer_id}'
            GROUP BY oi.product_id
            ORDER BY purchase_frequency DESC
            LIMIT {limit}
        """,
        "customer_top_products_sales": """
            SELECT oi.product_id, SUM(oi.price) as total_sales
            FROM olist_order_items_dataset oi
            JOIN olist_orders_dataset o ON oi.order_id = o.order_id
            WHERE o.customer_id = '{customer_id}'
            GROUP BY oi.product_id
            ORDER BY total_sales DESC
            LIMIT {limit}
        """,

        # PRODUCT queries
        "product_details": """
            SELECT p.*, pc.product_category_name_english
            FROM olist_products_dataset p
            LEFT JOIN product_category_name_translation pc ON p.product_category_name = pc.product_category_name
            WHERE p.product_id = '{product_id}'
        """,
        "product_top_customers": """
            SELECT o.customer_id, SUM(oi.price) as total_spent, COUNT(*) as purchase_count
            FROM olist_order_items_dataset oi
            JOIN olist_orders_dataset o ON oi.order_id = o.order_id
            WHERE oi.product_id = '{product_id}'
            GROUP BY o.customer_id
            ORDER BY total_spent DESC
            LIMIT {limit}
        """,

        # SELLER queries
        "seller_performance": """
            SELECT s.seller_id, s.seller_city, s.seller_state,
                   COUNT(oi.order_id) as total_orders,
                   SUM(oi.price) as total_revenue
            FROM olist_sellers_dataset s
            JOIN olist_order_items_dataset oi ON s.seller_id = oi.seller_id
            WHERE s.seller_id = '{seller_id}'
            GROUP BY s.seller_id, s.seller_city, s.seller_state
        """,
        "top_sellers": """
            SELECT oi.seller_id, SUM(oi.price) as total_revenue, COUNT(*) as total_orders
            FROM olist_order_items_dataset oi
            GROUP BY oi.seller_id
            ORDER BY total_revenue DESC
            LIMIT {limit}
        """,

        # ORDER queries
        "order_details": """
            SELECT o.*, oi.product_id, oi.price, oi.freight_value
            FROM olist_orders_dataset o
            JOIN olist_order_items_dataset oi ON o.order_id = oi.order_id
            WHERE o.order_id = '{order_id}'
        """,

        # REVIEW queries
        "product_reviews": """
            SELECT r.review_score, r.review_comment_title, r.review_comment_message
            FROM olist_order_reviews_dataset r
            JOIN olist_orders_dataset o ON r.order_id = o.order_id
            JOIN olist_order_items_dataset oi ON o.order_id = oi.order_id
            WHERE oi.product_id = '{product_id}'
            ORDER BY r.review_creation_date DESC
            LIMIT {limit}
        """,

        # CATEGORY queries
        "category_performance": """
            SELECT p.product_category_name,
                   COUNT(*) as total_products,
                   AVG(oi.price) as avg_price,
                   SUM(oi.price) as total_sales
            FROM olist_products_dataset p
            JOIN olist_order_items_dataset oi ON p.product_id = oi.product_id
            WHERE p.product_category_name = '{category}'
            GROUP BY p.product_category_name
        """
    }

    # Determine query type and entity
    limit = 3 if "three" in question_lower or "top 3" in question_lower else 5

    # Match question patterns to appropriate SQL templates
    for entity_type, entity_data in detected_entities.items():
        entity_id_key = f"{entity_type}_id"
        entity_id = resolved_ids.get(entity_id_key) or key_findings.get(f"latest_{entity_type}_id")

        if not entity_id:
            continue

        # Customer-specific queries
        if entity_type == "customer" and entity_data.get("referenced"):
            if "frequency" in question_lower and "product" in question_lower:
                return sql_templates["customer_top_products_frequency"].format(customer_id=entity_id, limit=limit)
            elif "sales" in question_lower and "product" in question_lower:
                return sql_templates["customer_top_products_sales"].format(customer_id=entity_id, limit=limit)

        # Product-specific queries
        elif entity_type == "product" and entity_data.get("referenced"):
            if "category" in question_lower or "details" in question_lower or "information" in question_lower:
                return sql_templates["product_details"].format(product_id=entity_id)
            elif "customer" in question_lower and ("top" in question_lower or "contributor" in question_lower):
                return sql_templates["product_top_customers"].format(product_id=entity_id, limit=limit)
            elif "review" in question_lower:
                return sql_templates["product_reviews"].format(product_id=entity_id, limit=limit)

        # Seller-specific queries
        elif entity_type == "seller" and entity_data.get("referenced"):
            return sql_templates["seller_performance"].format(seller_id=entity_id)

        # Order-specific queries
        elif entity_type == "order" and entity_data.get("referenced"):
            return sql_templates["order_details"].format(order_id=entity_id)

    # Fallback to common aggregation queries
    if "top selling product" in question_lower:
        return sql_templates["customer_top_products_sales"].format(customer_id="ANY", limit=1).replace("WHERE o.customer_id = 'ANY'", "")
    elif "top seller" in question_lower:
        return sql_templates["top_sellers"].format(limit=limit)

    # Ultimate fallback
    return "SELECT 'Unable to generate appropriate query for this request' as message"

def generate_contextual_fallback_sql(user_question: str, shared_context: dict, step_description: str, step_index: int) -> str:
    """Generate context-aware fallback SQL for specific action sequence steps."""
    question_lower = user_question.lower()
    step_desc_lower = step_description.lower()

    # Get entity resolution data
    conversation_entities = shared_context.get("conversation_entities", {})
    detected_entities = conversation_entities.get("detected_entities", {})
    resolved_ids = conversation_entities.get("resolved_entity_ids", {})
    key_findings = shared_context.get("key_findings", {})

    # Try to get product_id from context (multiple sources)
    product_id = (resolved_ids.get("product_id") or
                 key_findings.get("latest_product_id") or
                 shared_context.get("referenced_entities", {}).get("product_id"))

    # If we have a product context, generate product-specific queries
    if product_id:
        # Step 1 or category-related: Product details/category
        if step_index == 0 or "category" in step_desc_lower or "information" in step_desc_lower or "details" in step_desc_lower:
            return f"""
                SELECT p.product_category_name, p.product_name_lenght, p.product_description_lenght,
                       p.product_photos_qty, p.product_weight_g, p.product_length_cm,
                       p.product_height_cm, p.product_width_cm,
                       pc.product_category_name_english
                FROM olist_products_dataset p
                LEFT JOIN product_category_name_translation pc ON p.product_category_name = pc.product_category_name
                WHERE p.product_id = '{product_id}'
            """.strip()

        # Step 2 or customer-related: Top customers for this product
        elif step_index == 1 or "customer" in step_desc_lower or "top" in step_desc_lower or "contributor" in step_desc_lower:
            return f"""
                SELECT o.customer_id, SUM(oi.price) as total_spent, COUNT(*) as purchase_count,
                       c.customer_city, c.customer_state
                FROM olist_order_items_dataset oi
                JOIN olist_orders_dataset o ON oi.order_id = o.order_id
                JOIN olist_customers_dataset c ON o.customer_id = c.customer_id
                WHERE oi.product_id = '{product_id}'
                GROUP BY o.customer_id, c.customer_city, c.customer_state
                ORDER BY total_spent DESC
                LIMIT 5
            """.strip()

    # Fallback to original method if no specific context
    return generate_fallback_sql(user_question, shared_context)

def validate_ds_response(ds_response: dict) -> Dict[str, Any]:
    """Validate DS response for critical errors before judge review."""
    issues = []

    # Check for NULL SQL queries in single action
    if ds_response.get("action") == "sql":
        sql = ds_response.get("duckdb_sql")
        if sql is None or sql == "NULL" or (isinstance(sql, str) and sql.strip() == ""):
            issues.append("CRITICAL: SQL action has NULL/empty duckdb_sql field")

    # Check for NULL SQL queries in action sequence
    if ds_response.get("action_sequence"):
        for i, step in enumerate(ds_response.get("action_sequence", [])):
            if step.get("action") == "sql":
                sql = step.get("duckdb_sql")
                if sql is None or sql == "NULL" or (isinstance(sql, str) and sql.strip() == ""):
                    issues.append(f"CRITICAL: Action sequence step {i+1} has NULL/empty duckdb_sql field")

    # Check for missing required fields (but allow empty ds_summary)
    if ds_response.get("ds_summary") is None:
        issues.append("Missing ds_summary field")

    if not ds_response.get("action") and not ds_response.get("action_sequence"):
        issues.append("Missing both action and action_sequence fields")

    return {
        "valid": len(issues) == 0,
        "issues": issues
    }

def fix_ds_response_with_fallback(ds_response: dict, user_question: str, shared_context: dict) -> dict:
    """Fix DS response by generating fallback SQL queries for NULL values."""
    fixed_response = ds_response.copy()

    # Fix single SQL action
    if ds_response.get("action") == "sql":
        sql = ds_response.get("duckdb_sql")
        if sql is None or sql == "NULL" or (isinstance(sql, str) and sql.strip() == ""):
            fallback_sql = generate_fallback_sql(user_question, shared_context)
            fixed_response["duckdb_sql"] = fallback_sql
            fixed_response["ds_summary"] = fixed_response.get("ds_summary", "") + " [FALLBACK SQL GENERATED]"

    # Fix action sequence SQL
    if ds_response.get("action_sequence"):
        fixed_sequence = []
        for i, step in enumerate(ds_response.get("action_sequence", [])):
            fixed_step = step.copy()
            if step.get("action") == "sql":
                sql = step.get("duckdb_sql")
                if sql is None or sql == "NULL" or (isinstance(sql, str) and sql.strip() == ""):
                    # Generate context-aware fallback SQL based on step description
                    step_description = step.get("description", "").lower()
                    fallback_sql = generate_contextual_fallback_sql(user_question, shared_context, step_description, i)
                    fixed_step["duckdb_sql"] = fallback_sql
            fixed_sequence.append(fixed_step)
        fixed_response["action_sequence"] = fixed_sequence
        fixed_response["ds_summary"] = fixed_response.get("ds_summary", "") + " [FALLBACK SQL GENERATED]"

    return fixed_response
