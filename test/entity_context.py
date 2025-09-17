"""
Entity detection and context management - Complete version from original app.py
"""
import streamlit as st
from typing import Dict, Any, List

# Brazilian E-commerce Dataset Entity Schema
ENTITY_SCHEMA = {
    "customer": {
        "primary_key": "customer_id",
        "tables": ["olist_customers_dataset", "olist_orders_dataset"],
        "reference_patterns": ["this customer", "the customer", "customer", "buyer"],
        "typical_queries": ["top customer", "customer analysis", "customer behavior"],
        "related_entities": ["order", "review", "geolocation"]
    },
    "product": {
        "primary_key": "product_id",
        "tables": ["olist_products_dataset", "olist_order_items_dataset"],
        "reference_patterns": ["this product", "the product", "product", "item"],
        "typical_queries": ["top selling product", "product analysis", "product category"],
        "related_entities": ["order", "seller", "review", "category"]
    },
    "order": {
        "primary_key": "order_id",
        "tables": ["olist_orders_dataset", "olist_order_items_dataset"],
        "reference_patterns": ["this order", "the order", "order", "purchase"],
        "typical_queries": ["order analysis", "order status", "order value"],
        "related_entities": ["customer", "product", "seller", "payment", "review"]
    },
    "seller": {
        "primary_key": "seller_id",
        "tables": ["olist_sellers_dataset", "olist_order_items_dataset"],
        "reference_patterns": ["this seller", "the seller", "seller", "vendor"],
        "typical_queries": ["top seller", "seller performance", "seller analysis"],
        "related_entities": ["product", "order", "geolocation"]
    },
    "payment": {
        "primary_key": "order_id",  # payments are linked to orders
        "tables": ["olist_order_payments_dataset"],
        "reference_patterns": ["this payment", "the payment", "payment", "transaction"],
        "typical_queries": ["payment method", "payment analysis", "payment value"],
        "related_entities": ["order", "customer"]
    },
    "review": {
        "primary_key": "review_id",
        "tables": ["olist_order_reviews_dataset"],
        "reference_patterns": ["this review", "the review", "review", "rating"],
        "typical_queries": ["review analysis", "review sentiment", "review score"],
        "related_entities": ["order", "customer", "product"]
    },
    "category": {
        "primary_key": "product_category_name",
        "tables": ["olist_products_dataset"],
        "reference_patterns": ["this category", "the category", "category", "product type"],
        "typical_queries": ["category analysis", "top category", "category performance"],
        "related_entities": ["product"]
    },
    "geolocation": {
        "primary_key": "geolocation_zip_code_prefix",
        "tables": ["olist_geolocation_dataset", "olist_customers_dataset", "olist_sellers_dataset"],
        "reference_patterns": ["this location", "the location", "city", "state"],
        "typical_queries": ["location analysis", "geographic distribution", "regional sales"],
        "related_entities": ["customer", "seller"]
    }
}

def detect_entity_references(text: str) -> Dict[str, Any]:
    """Dynamically detect entity references in user questions."""
    text_lower = text.lower()
    detected_entities = {}

    for entity_type, schema in ENTITY_SCHEMA.items():
        # Check for reference patterns
        for pattern in schema["reference_patterns"]:
            if pattern in text_lower:
                detected_entities[entity_type] = {
                    "referenced": True,
                    "pattern_matched": pattern,
                    "primary_key": schema["primary_key"],
                    "tables": schema["tables"]
                }
                break

        # Check for typical query patterns
        for query_pattern in schema["typical_queries"]:
            if query_pattern in text_lower:
                if entity_type not in detected_entities:
                    detected_entities[entity_type] = {
                        "referenced": False,
                        "query_type": query_pattern,
                        "primary_key": schema["primary_key"],
                        "tables": schema["tables"]
                    }

    return detected_entities

def extract_entity_ids_from_results(query_results: dict, entity_types: List[str]) -> Dict[str, str]:
    """Extract actual entity IDs from SQL query results."""
    extracted_ids = {}

    if not query_results or not isinstance(query_results, dict):
        return extracted_ids

    for result_key, result_data in query_results.items():
        # Handle case where result_data is None or not a dict
        if not result_data or not isinstance(result_data, dict):
            continue

        sample_data = result_data.get("sample_data", [])
        if sample_data and isinstance(sample_data, list) and len(sample_data) > 0:
            first_row = sample_data[0]

            # Ensure first_row is a dict
            if not isinstance(first_row, dict):
                continue

            # Look for entity IDs in the result data
            for entity_type in entity_types:
                if entity_type in ENTITY_SCHEMA:
                    primary_key = ENTITY_SCHEMA[entity_type]["primary_key"]
                    if primary_key in first_row:
                        extracted_ids[f"{entity_type}_id"] = str(first_row[primary_key])

    return extracted_ids

def resolve_contextual_entities(current_question: str, conversation_history: List[str], cached_results: dict) -> Dict[str, Any]:
    """Resolve 'this X' references to actual entity IDs from conversation context."""

    # Detect what entities are being referenced in current question
    current_entities = detect_entity_references(current_question)

    # Look for entities discussed in conversation history
    historical_entities = {}
    for past_question in conversation_history[-5:]:  # Last 5 questions
        hist_entities = detect_entity_references(past_question)
        historical_entities.update(hist_entities)

    # Extract actual entity IDs from cached results
    entity_types_to_find = []
    for entity_type, data in current_entities.items():
        if data.get("referenced"):  # If using "this X" pattern
            entity_types_to_find.append(entity_type)

    # Get all recent SQL results to extract IDs
    all_sql_results = {}
    if "sql" in cached_results:
        # Add current cached SQL results
        all_sql_results.update(cached_results.get("sql", {}))

    # Extract IDs from results
    try:
        resolved_ids = extract_entity_ids_from_results(all_sql_results, entity_types_to_find)
    except Exception as e:
        # If entity extraction fails, return empty resolved_ids to prevent crashes
        resolved_ids = {}
        if st.session_state.get("debug_judge", False):
            st.warning(f"Entity ID extraction failed: {str(e)}")

    return {
        "current_entities": current_entities,
        "historical_entities": historical_entities,
        "resolved_ids": resolved_ids,
        "entity_schema": {etype: ENTITY_SCHEMA[etype] for etype in set(list(current_entities.keys()) + list(historical_entities.keys()))}
    }

def assess_context_relevance(current_question: str, available_context: dict) -> dict:
    """Assess which parts of the context are relevant to the current question."""
    question_lower = current_question.lower()

    # Question type classification
    if any(word in question_lower for word in ["this product", "this customer", "this order", "this seller"]):
        question_type = "specific_entity_reference"
    elif any(word in question_lower for word in ["all", "overview", "general", "total"]):
        question_type = "broad_analysis"
    elif any(word in question_lower for word in ["explain", "interpret", "what does", "why"]):
        question_type = "explanation"
    else:
        question_type = "new_analysis"

    # Update context relevance
    available_context["context_relevance"] = {
        "question_type": question_type,
        "requires_entity_continuity": question_type == "specific_entity_reference",
        "use_cached_results": question_type in ["specific_entity_reference", "explanation"],
        "allow_broad_analysis": question_type in ["broad_analysis", "new_analysis"]
    }

    return available_context

def get_table_schema_info() -> Dict[str, Dict[str, Any]]:
    """Get comprehensive schema information for all tables."""
    return {
        "olist_customers_dataset": {
            "columns": ["customer_id", "customer_unique_id", "customer_zip_code_prefix", "customer_city", "customer_state"],
            "row_count": 99441,
            "description": "Customer information including location data"
        },
        "olist_orders_dataset": {
            "columns": ["order_id", "customer_id", "order_status", "order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date", "order_delivered_customer_date", "order_estimated_delivery_date"],
            "row_count": 99441,
            "description": "Order information with status and timestamps"
        },
        "olist_order_items_dataset": {
            "columns": ["order_id", "order_item_id", "product_id", "seller_id", "shipping_limit_date", "price", "freight_value"],
            "row_count": 112650,
            "description": "Individual items within orders with pricing"
        },
        "olist_products_dataset": {
            "columns": ["product_id", "product_category_name", "product_name_lenght", "product_description_lenght", "product_photos_qty", "product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"],
            "row_count": 32951,
            "description": "Product catalog with physical dimensions"
        },
        "olist_sellers_dataset": {
            "columns": ["seller_id", "seller_zip_code_prefix", "seller_city", "seller_state"],
            "row_count": 3095,
            "description": "Seller information and location"
        },
        "olist_order_reviews_dataset": {
            "columns": ["review_id", "order_id", "review_score", "review_comment_title", "review_comment_message", "review_creation_date", "review_answer_timestamp"],
            "row_count": 99224,
            "description": "Customer reviews and ratings"
        },
        "olist_order_payments_dataset": {
            "columns": ["order_id", "payment_sequential", "payment_type", "payment_installments", "payment_value"],
            "row_count": 103886,
            "description": "Payment information for orders"
        },
        "product_category_name_translation": {
            "columns": ["product_category_name", "product_category_name_english"],
            "row_count": 71,
            "description": "Translation of category names to English"
        },
        "olist_geolocation_dataset": {
            "columns": ["geolocation_zip_code_prefix", "geolocation_lat", "geolocation_lng", "geolocation_city", "geolocation_state"],
            "row_count": 1000163,
            "description": "Geographic coordinates for zip codes"
        }
    }

def _identify_business_columns(columns: List[str]) -> Dict[str, List[str]]:
    """Identify business-relevant columns for different types of analysis."""
    business_columns = {
        "price_metrics": [],
        "sales_metrics": [],
        "quantity_metrics": [],
        "time_metrics": [],
        "geographic_metrics": [],
        "quality_metrics": [],
        "identifier_columns": [],
        "physical_dimensions": []
    }

    for col in columns:
        col_lower = col.lower()

        # Price and financial metrics
        if any(word in col_lower for word in ["price", "payment", "value", "cost", "revenue"]):
            business_columns["price_metrics"].append(col)
            business_columns["sales_metrics"].append(col)

        # Quantity metrics
        if any(word in col_lower for word in ["qty", "quantity", "count", "number", "total"]):
            business_columns["quantity_metrics"].append(col)

        # Time metrics
        if any(word in col_lower for word in ["date", "time", "timestamp", "created", "updated"]):
            business_columns["time_metrics"].append(col)

        # Geographic metrics
        if any(word in col_lower for word in ["city", "state", "zip", "location", "geo", "lat", "lng"]):
            business_columns["geographic_metrics"].append(col)

        # Quality metrics
        if any(word in col_lower for word in ["score", "rating", "review", "satisfaction"]):
            business_columns["quality_metrics"].append(col)

        # Identifiers
        if any(word in col_lower for word in ["id", "key", "code"]):
            business_columns["identifier_columns"].append(col)

        # Physical dimensions
        if any(word in col_lower for word in ["weight", "length", "height", "width", "cm", "kg"]):
            business_columns["physical_dimensions"].append(col)

    return business_columns

def suggest_columns_for_query(intent: str, table_schema: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
    """Suggest relevant columns based on query intent."""
    intent_lower = intent.lower()
    suggested_columns = {}

    for table_name, schema in table_schema.items():
        columns = schema.get("columns", [])
        business_cols = _identify_business_columns(columns)

        # Suggest columns based on intent
        relevant_cols = []

        if any(word in intent_lower for word in ["price", "sales", "revenue", "money"]):
            relevant_cols.extend(business_cols["price_metrics"])

        if any(word in intent_lower for word in ["top", "best", "highest", "lowest"]):
            relevant_cols.extend(business_cols["sales_metrics"])
            relevant_cols.extend(business_cols["quantity_metrics"])

        if any(word in intent_lower for word in ["location", "geographic", "region", "city", "state"]):
            relevant_cols.extend(business_cols["geographic_metrics"])

        if any(word in intent_lower for word in ["quality", "review", "rating", "satisfaction"]):
            relevant_cols.extend(business_cols["quality_metrics"])

        if any(word in intent_lower for word in ["time", "trend", "period", "date"]):
            relevant_cols.extend(business_cols["time_metrics"])

        # Always include key identifiers for joins
        relevant_cols.extend([col for col in business_cols["identifier_columns"] if "id" in col.lower()])

        if relevant_cols:
            suggested_columns[table_name] = list(set(relevant_cols))  # Remove duplicates

    return suggested_columns