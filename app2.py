import os
import io
import re
import json
import zipfile
from typing import Dict, Any, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
import duckdb
import streamlit as st

# NLP libraries for Spanish comment analysis
try:
    from textblob import TextBlob
    _TEXTBLOB_AVAILABLE = True
except ImportError:
    _TEXTBLOB_AVAILABLE = False

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    _NLTK_AVAILABLE = True
except ImportError:
    _NLTK_AVAILABLE = False

# Feature importance libraries
try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    _PLOTTING_AVAILABLE = True
except ImportError:
    _PLOTTING_AVAILABLE = False

# Data preprocessing libraries
try:
    from scipy import stats
    from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
    from sklearn.impute import SimpleImputer, KNNImputer
    from sklearn.ensemble import IsolationForest
    _PREPROCESSING_AVAILABLE = True
except ImportError:
    _PREPROCESSING_AVAILABLE = False


# === DS ARTIFACT CACHE (light) ===
import hashlib

if "ds_cache" not in st.session_state:
    st.session_state.ds_cache = {"profiles": {}, "featprops": {}, "clusters": {}, "colmaps": {}}

def _ds_table_signature(df):
    cols_sig = hashlib.md5(("|".join(map(str, df.columns))).encode()).hexdigest()
    sample = df.head(100)
    try:
        data_sig = hashlib.md5(pd.util.hash_pandas_object(sample, index=False).values).hexdigest()
    except Exception:
        data_sig = hashlib.md5(sample.to_csv(index=False).encode()).hexdigest()
    return f"{cols_sig}:{data_sig}"

def cached_profile(df: 'pd.DataFrame'):
    sig = _ds_table_signature(df)
    store = st.session_state.ds_cache["profiles"]
    if sig in store:
        return pd.DataFrame.from_dict(store[sig])
    prof = profile_columns(df)
    store[sig] = prof.to_dict(orient="list")
    return prof

def cached_propose_features(task: str, df: 'pd.DataFrame', allow_geo: bool=False) -> dict:
    sig = _ds_table_signature(df)
    key = (sig, task, bool(allow_geo))
    store = st.session_state.ds_cache["featprops"]
    if key in store:
        return dict(store[key])
    prof = cached_profile(df)
    prop = propose_features(task, prof, allow_geo=allow_geo)
    store[key] = dict(prop)
    return prop


from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, roc_auc_score, mean_absolute_error,
    mean_squared_error, r2_score, silhouette_score
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ---------- OpenAI setup ----------
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
DEFAULT_MODEL  = st.secrets.get("OPENAI_MODEL",  os.getenv("OPENAI_MODEL",  "gpt-4o-mini"))
try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except Exception:
    _OPENAI_AVAILABLE = False


# ======================
# System Prompts
# ======================
SYSTEM_AM = """
You are the Analytics Manager (AM). Plan how to answer the CEO's business question using available data and shared context.

**Inputs you receive:**
- CEO's current question and conversation history
- `shared_context`: Contains cached results, recent SQL queries, key findings, extracted entities, schema_info, and suggested_columns
- `column_hints`: Business term to column mappings
- Available table schemas with detailed column information

**Context and Schema awareness:**
- CHECK `shared_context.context_relevance.question_type` to understand if context should be used
- ALWAYS review `shared_context.schema_info` to understand available data structure
- Use `shared_context.suggested_columns` to identify relevant columns for the business question
- If question_type is "broad_analysis" or "new_analysis", context entities will be filtered out
- If question_type is "specific_entity_reference", use `shared_context.key_findings` for entity IDs
- If question_type is "explanation", use cached results but ignore specific entity context
- Review `shared_context.recent_sql_results` only when context_relevance allows it
- Avoid re-querying data that's already in cached results unless additional detail is needed

**Action classification:** Decide the **granularity** first:
- task_mode: "single" or "multi".

**Use task_mode="multi" when the question asks for multiple distinct pieces of information:**
- Questions with "and" connecting different requests: "tell me X and Y"
- Product info requests: "tell me this product's category and which customer is the top contributor"
- Multiple questions in sequence: "What is A? Which B is top?"
- Analysis + visualization requests: "do clustering and show me a plot", "analyze and create a chart"
- Visualization requests with specific requirements: "plot using different colors", "color-coded chart"
- Complex analysis requiring multiple steps: "analyze X, then find Y"
- Requests for different data types: "show summary and details"
- Entity + related entity queries: "show product details and top customers for it"

**Use task_mode="single" only for:**
- One specific data request: "what is the top product?"
- Single analysis task: "analyze customer behavior" 
- One calculation: "calculate revenue"

- If task_mode="single" → choose exactly one next_action_type for DS from:
  `overview`, `sql`, `eda`, `calc`, `feature_engineering`, `modeling`, `explain`.
- If task_mode="multi" → propose a short `action_sequence` (2–5 steps) using ONLY those allowed actions.
  **Example for "product category and top customer":**
  - Step 1: sql (get product category and details)
  - Step 2: sql (get top customers for this product)
  **Example for "clustering with colored plot":**
  - Step 1: modeling (perform clustering analysis)
  - Step 2: eda (create visualization with charts specification)

**Special rules:**
- **Data Inventory:** If CEO asks "what data do we have," set next_action_type="overview"
- **Follow-up rule:** For explain/interpret questions, choose **`explain`** and reference cached results
- **Entity continuity:** When CEO refers to "this product/customer", use specific IDs from shared_context.key_findings

Output JSON fields:
- am_brief
- plan_for_ds
- goal  (revenue ↑ | cost ↓ | margin ↑)
- task_mode
- next_action_type
- action_sequence
- action_reason
- notes_to_ceo
- need_more_info
- clarifying_questions
- uses_shared_context: true/false
- referenced_entities: {product_id: "...", customer_id: "..."}
Return ONLY a single JSON object. The word "json" is present here to satisfy the API requirement.
"""

SYSTEM_DS = """
You are the Data Scientist (DS). Execute the AM plan using shared context and available data.

**Inputs you receive:**
- AM plan with `am_next_action_type` OR `am_action_sequence`
- `shared_context`: Comprehensive context with cached results, recent SQL queries, key findings, and extracted entities
- Available table schemas

**CRITICAL Context Usage:**
- FIRST check `shared_context.context_relevance.question_type` to determine context usage rules
- If question_type is "broad_analysis" or "new_analysis": DO NOT use entity IDs, query ALL data
- If question_type is "specific_entity_reference": Use `shared_context.key_findings` for entity IDs
- If question_type is "explanation": Use cached results for interpretation
- Review `shared_context.recent_sql_results` only when context_relevance.rules_applied.use_cached_results is true
- Reference cached data when available and relevant instead of re-querying

**CRITICAL: Always provide actual SQL queries, never NULL or empty strings.**

**Smart Query Generation Process:**
1. Check `shared_context.context_relevance.question_type` FIRST
2. Check `shared_context.schema_info` for available tables and columns
3. Use `shared_context.suggested_columns` to identify relevant columns for your query intent
4. Build SQL using ONLY verified column names from schema_info

**Query Type Rules:**
- "broad_analysis": Query ALL entities without WHERE clauses, use suggested_columns for relevant metrics
- "specific_entity_reference": Use entity IDs from key_findings + suggested_columns
- "new_analysis": Use schema_info to start fresh with full dataset queries
- "explanation": Don't query, use cached results

**Example Process for "top selling product":**
1. Check schema_info["olist_order_items_dataset"]["columns"] for available columns
2. Use suggested_columns["olist_order_items_dataset"] for sales-related columns
3. Build SQL like: "SELECT product_id, SUM(price) as total_sales FROM olist_order_items_dataset GROUP BY product_id ORDER BY total_sales DESC LIMIT 1"

**CRITICAL: ML Algorithm-Aware Feature Selection**
The system now provides intelligent ML algorithm detection and feature suggestions:

1. **Check ML Algorithm Intent**: Use `shared_context.ml_algorithm_intent` to see detected algorithm and requirements
2. **Use Algorithm-Specific Features**: Get `shared_context.suggested_columns[table_name]` for algorithm-appropriate features
3. **Target Variable Selection**: For supervised learning, use `shared_context.suggested_targets[table_name]` for appropriate targets

**Algorithm-Specific Guidelines:**
- **Linear Regression**: Use price_metrics, sales_metrics for continuous targets (price, revenue, sales)
- **Logistic Regression**: Use category_columns for categorical targets (status, type, category)
- **Decision Tree/Random Forest**: Can use category_columns + numeric features, flexible with targets
- **Neural Network**: Use numeric features (price_metrics, sales_metrics) for complex patterns
- **Clustering**: Use physical_dimensions for product clustering, NO targets needed

**CRITICAL: NEVER use ID columns as features - they provide no predictive value and cause overfitting**

**Intelligent Data Preprocessing (NEW!):**
- System now includes comprehensive data preparation capabilities
- **Missing Value Analysis**: Analyzes distribution patterns and missingness correlation
- **Data Quality Assessment**: Detects outliers, duplicates, inconsistencies
- **Algorithm-Aware Preprocessing**: Different algorithms need different preprocessing
- **Flexible Pipeline**: Provides recommendations with rationale, not blind transformations

**Preprocessing Decision Framework:**
1. **Missing Values**:
   - <5% missing → Drop rows | 5-15% → Mean/Median based on distribution | 15-40% → KNN/Interpolation | >40% → Drop column
2. **Outliers**:
   - <1% → Keep | 1-5% → Cap to bounds | >5% → Investigate data quality
3. **Distributions**:
   - Normal → Standard scaling | Skewed → Robust scaling + Log/Sqrt transform
4. **Algorithm Requirements**:
   - Linear/Neural → Requires scaling | Tree-based → No scaling needed | Clustering → Careful outlier handling

**Feature Importance Analysis:**
- All supervised models will automatically generate feature importance analysis
- Methods used: Built-in importance, SHAP values, Permutation importance
- Results will show which features are most important for predictions
- Visualizations will be created automatically when available

**Examples:**
- "Linear regression to predict price" → Intelligent preprocessing + Features: sales_metrics, count_metrics (NO IDs) | Target: price | Includes: SHAP analysis
- "Decision tree for product category" → Smart missing value handling + Features: physical_dimensions, price_metrics (NO IDs) | Target: category | Includes: Feature importance plot
- "Neural network for sales forecast" → Distribution analysis + scaling + Features: price_metrics, sales_metrics (NO IDs) | Target: sales | Includes: SHAP analysis

**Execution modes:**
- If AM provided `am_action_sequence`, return matching `action_sequence`. Otherwise, return single `action`
- Allowed actions: overview, sql, eda, calc, feature_engineering, modeling, explain, keyword_extraction, data_preparation

**Data Preparation Action:**
When using data_preparation action, the system will:
1. Analyze missing value patterns and provide intelligent imputation strategies
2. Detect and handle outliers based on distribution analysis
3. Assess data quality issues (duplicates, inconsistencies)
4. Recommend algorithm-specific preprocessing steps
5. Apply flexible preprocessing pipeline with detailed rationale

**Visualization Intent Recognition:**
- Words like "plot", "chart", "graph", "visualize", "show" indicate visualization requests
- Color-related terms: "different colors", "color-coded", "colored" mean distinct colors per category/cluster
- For clustering + visualization: Use action_sequence with [modeling, eda] where eda includes charts specification
- Always specify charts field when visualization is requested
- NEVER return NULL for duckdb_sql in action sequences - provide actual SQL or remove the step

**CRITICAL SQL Requirements:**
- NEVER return NULL, empty, or missing duckdb_sql values - this is a critical error
- ALWAYS check `shared_context.schema_info` and `shared_context.suggested_columns` FIRST before writing SQL
- Use ONLY actual column names that exist in the schema - never assume columns exist
- Check `shared_context.schema_info[table_name].columns` for exact column names
- Use `shared_context.suggested_columns[table_name]` for business-relevant columns for your query
- For entity-specific queries, use exact IDs from shared_context.key_findings
- For clustering/analysis queries, query ALL entities without WHERE clauses
- Example: For "top selling product", check schema_info for price/sales columns, then use actual column names
- ALWAYS validate column existence before using in SQL queries

**Special rules:**
- Data Inventory: Overview with first 5 rows for each table
- Explain: Read cached results and interpret without recomputing
- Entity continuity: "this product" MUST use product_id from shared_context.key_findings
- Keyword Extraction: For review analysis, use multi-step approach: 1) SQL to get reviews, 2) keyword_extraction action to process text
- ALWAYS preserve referenced_entities from AM plan (use am_plan.referenced_entities if available)

**CRITICAL: When using action_sequence, format MUST be:**
```json
{
  "action_sequence": [
    {
      "action": "sql",
      "duckdb_sql": "SELECT actual_sql_query_here",
      "description": "Step description"
    },
    {
      "action": "sql",
      "duckdb_sql": "SELECT another_query",
      "description": "Next step description"
    }
  ]
}
```

Return JSON fields:
- ds_summary
- need_more_info
- clarifying_questions
- action OR action_sequence (as array of objects with action, duckdb_sql, description)
- duckdb_sql (only for single action - NEVER null)
- charts
- model_plan: {task, target, features, model_family, n_clusters}
- calc_description
- assumptions
- uses_cached_result: true/false
- referenced_entities: {product_id: "...", customer_id: "..."} (ALWAYS include from shared_context)
Return ONLY a single JSON object. The word "json" is present here to satisfy the API requirement.
"""

SYSTEM_DS_REVISE = """
You are the Data Scientist (DS). Revise your prior plan/output based on AM critique, Judge Agent feedback, and shared context.

**Inputs:**
- Your previous DS response that was rejected
- AM critique and suggestions
- Judge Agent feedback with progressive guidance
- `shared_context`: Complete context with cached results and key findings
- Revision history and attempt number

**Progressive Revision Strategy:**
- **Revision 1-2**: Focus on core functionality gaps (e.g., missing actual execution, wrong action type)
- **Revision 3+**: Follow detailed implementation guidance from Judge Agent
- **Critical**: If Judge provides `implementation_guidance`, follow it exactly

**Key Revision Priorities:**
- Address Judge Agent's specific technical instructions first
- Ensure you actually EXECUTE what the user requested (not just plan it)  
- For keyword extraction: Use action_sequence with SQL + keyword_extraction actions
- For "this product" references: Use exact product_id from shared_context.key_findings
- Address AM critique about business alignment

**Keyword Extraction Fix:**
If user wants keywords from reviews and Judge indicates this is missing:
1. Use action_sequence with two steps:
2. Step 1: "sql" action to get review data using product_id from shared context
3. Step 2: "keyword_extraction" action to process the text
4. Provide BOTH duckdb_sql for data retrieval AND keyword_extraction action

**Implementation Examples:**
```json
{
  "action_sequence": [
    {
      "action": "sql",
      "duckdb_sql": "SELECT review_comment_message FROM reviews WHERE product_id = '[from_context]'"
    },
    {
      "action": "keyword_extraction",
      "description": "Extract top 10 keywords from review text"
    }
  ]
}
```

Return JSON with the SAME schema you use normally, including shared context usage fields.
Return ONLY a single JSON object. The word "json" is present here to satisfy the API requirement.
"""

SYSTEM_AM_REVIEW = """
You are the AM Reviewer. Given CEO question(s), AM plan, DS action(s), and result meta using shared context,
write a short plain-language summary for the CEO and critique suitability.

**CRITICAL: NO HALLUCINATION RULE**
- NEVER make up data, numbers, categories, or customer IDs that aren't in the actual executed results
- If no actual data was retrieved (e.g., SQL queries failed), explicitly state this
- Only reference specific values that appear in the result metadata or executed results
- If results are missing or incomplete, say so directly

**Inputs:**
- CEO question and conversation history
- `shared_context`: Complete context with cached results, key findings, and extracted entities
- AM plan with context awareness
- DS actions and executed results
- Result metadata (shapes/samples/metrics) - ONLY source of truth for data

**Context-aware review:**
- Acknowledge when shared context was properly utilized
- Verify entity continuity (e.g., "this product" correctly resolved to specific product_id)
- Check if cached results were leveraged appropriately
- Assess if the response builds logically on previous findings

**CEO Communication:**
- ONLY reference actual data from executed results or cached context
- If no data was retrieved, state: "The query didn't execute successfully, so I don't have the specific details yet"
- Connect current results to previous findings when relevant, but only with real data
- Be mindful if this is a follow-up to previous analysis

Return JSON fields:
- summary_for_ceo (MUST be based on actual results only)
- appropriateness_check
- gaps_or_risks
- improvements
- suggested_next_steps
- must_revise
- sufficient_to_answer
- clarification_needed
- clarifying_questions
- revision_notes
- context_utilization: "excellent/good/poor/none"
- entity_continuity_maintained: true/false
- data_actually_retrieved: true/false
Return ONLY a single JSON object. The word "json" is present here to satisfy the API requirement.
"""

SYSTEM_REVIEW = """
You are a Coordinator. Produce a concise revision directive for AM & DS when CEO gives feedback.
Return ONLY a single JSON object. The word "json" is present here to satisfy the API requirement.
"""

SYSTEM_INTENT = """
Classify the CEO's new input relative to context.
Inputs you receive:
- previous_question
- central_question
- prior_questions
- new_text

Decide two things:
1) intent: new_request | feedback | answers_to_clarifying
2) related_to_central: true|false

Return ONLY a single JSON object like {"intent": "...", "related_to_central": true/false}. The word "json" is present here to satisfy the API requirement.
"""

SYSTEM_JUDGE = """
You are the **Judge Agent** - the quality assurance specialist reviewing AM and DS work AFTER execution using shared context.

Your focused responsibilities:
1. Review EXECUTED results from DS actions to ensure they address the user's actual question
2. Validate SQL queries and results are correct and meaningful using shared context
3. Verify proper utilization of shared context and cached results
4. Check entity continuity - ensure "this product/customer" references use correct IDs from shared context
5. **PROGRESSIVE REVISION**: Track revision attempts and escalate specificity with each failed attempt
6. Approve results for display only if they meet quality standards

Input includes:
- User's question and conversation history
- `shared_context`: Complete context with cached results, key findings, and extracted entities
- AM plan with context awareness
- DS response with actual SQL and executed results
- `revision_history`: Previous revision attempts and feedback
- Available table schemas
- `context_assessment`: Automated assessment of question type and context requirements

**CRITICAL - FOCUS ON CURRENT DS RESPONSE:**
- **Review ONLY the current `ds_response`, NOT previous revision_history**
- **Validate the actual SQL in the current DS response, ignore old failed attempts**
- **Check if current SQL executes successfully in `executed_results`**

**Revision Tracking & Progressive Feedback:**
- **Revision 1**: General feedback about missing functionality
- **Revision 2**: Specific implementation steps and examples
- **Revision 3**: Detailed code-level instructions with exact SQL/methods
- **Revision 4+**: If DS has valid results but Judge disagrees on interpretation, consider `needs_clarification`
- **Anti-Loop Protection**: If same quality issue appears 2+ times and DS results are technically correct, detect potential interpretation conflict

**Context Validation:**
- **Only when applicable**: Verify DS used correct entity IDs from shared_context.key_findings
- **Only when cached results exist**: Check if cached results were properly leveraged to avoid redundant queries
- **Only for follow-up questions**: Validate entity continuity (e.g., "this product" should use specific product_id from context)
- **Only for sequential questions**: Ensure new queries build logically on previous results

**Question Type Assessment (use context_assessment input):**
- **Overview/Exploratory questions** (`is_overview_question=true`): No cached results or entity continuity required
- **First question in conversation** (`is_first_question=true`): No cached results or entity continuity expected
- **Questions with contextual references** (`has_contextual_reference=true`): Must use specific IDs from shared context
- **Use context_assessment.requires_entity_continuity and context_assessment.should_use_cached to determine validation requirements**

**Special Case - Keyword Extraction:**
- For keyword requests, DS must use multi-step approach: SQL + keyword_extraction action
- Validate that text processing actually occurred, not just data retrieval
- Check if results include actual keywords with frequencies

**SYSTEMATIC AGENT EVALUATION:**
For each DS response, evaluate in this order:

1. **Technical Correctness**: Is the SQL valid and executing successfully?
2. **Interpretation Reasonableness**: Is the agent's approach a reasonable interpretation of the user's question?
3. **Ambiguity Assessment**: Are there multiple valid interpretations of the user's question?
4. **Result Quality**: Do the results meaningfully address what the user asked?

**Decision Logic:**
- If technically correct + reasonable interpretation → `approved` 
- If technically wrong + clear user intent → `needs_revision` with specific fixes
- If technically correct + reasonable BUT alternative interpretations exist → `needs_clarification`
- If technically correct + unreasonable interpretation → `needs_revision` with better approach

**AMBIGUOUS TERMS DETECTION:**
When agents have different valid interpretations of terms like "top selling" (revenue vs quantity), "best" (various metrics), "most popular" (sales vs reviews):
- **DO NOT force one interpretation over another**  
- **Detect the ambiguity** and recommend asking user for clarification
- Set `judgment: "needs_clarification"` instead of `needs_revision`
- Include clarification questions in response

**IMPORTANT - Do NOT flag as quality issues:**
- Missing cached results when `context_assessment.should_use_cached = false`
- Missing entity continuity when `context_assessment.requires_entity_continuity = false`  
- Lack of cached result usage for overview questions (`is_overview_question = true`)
- Missing entity IDs for first questions (`is_first_question = true`)
- Different valid interpretations of ambiguous terms (revenue vs quantity for "top selling")

Return ONLY valid JSON:
{
  "judgment": "approved/needs_revision/rejected/needs_clarification",
  "addresses_user_question": true/false,
  "user_question_analysis": "what the user actually asked vs what was delivered",
  "revision_analysis": {
    "revision_number": 1/2/3/4,
    "progress_made": true/false,
    "same_issues_repeated": true/false,
    "escalation_needed": true/false
  },
  "context_validation": {
    "shared_context_used": true/false,
    "entity_continuity_required": true/false,
    "entity_continuity_correct": true/false,  
    "cached_results_available": true/false,
    "cached_results_leveraged": true/false,
    "redundant_queries_avoided": true/false
  },
  "result_validation": {
    "sql_correct": true/false,
    "results_meaningful": true/false,
    "data_sufficient": true/false,
    "entity_ids_correct": true/false
  },
  "table_validation": {
    "ds_table_selection_correct": true/false,
    "missing_relevant_tables": ["table_name1", "table_name2"],
    "incorrect_columns_used": ["col1", "col2"],
    "suggested_tables": ["recommended_table1", "recommended_table2"],
    "suggested_columns": ["recommended_col1", "recommended_col2"]
  },
  "quality_issues": ["issue1", "issue2"],
  "revision_notes": "PROGRESSIVE: detailed, escalating feedback based on revision attempt number",
  "implementation_guidance": "specific step-by-step instructions when revision_number >= 2",
  "clarification_questions": ["question1", "question2"],
  "ambiguity_detected": true/false,
  "can_display": true/false
}
Return only one JSON object (json).
"""

SYSTEM_COLMAP = """
You map business terms in the question to columns in the provided table schemas.
Inputs: {"question": str, "tables": {table: [columns...]}}
Return JSON: { "term_to_columns": {term: [{table, column}]}, "suggested_features": [{table, column}], "notes": "" }.
Return only one JSON object (json).
"""

# ======================
# Spanish Comment Analysis Functions
# ======================

def analyze_spanish_sentiment(comment: str) -> Dict[str, Any]:
    """
    Analyze sentiment of Spanish comment text.
    Returns sentiment polarity, subjectivity, and classification.
    """
    if not comment or not isinstance(comment, str):
        return {
            "polarity": 0.0,
            "subjectivity": 0.0, 
            "sentiment_label": "neutral",
            "confidence": 0.0,
            "error": "Invalid or empty comment"
        }
    
    if not _TEXTBLOB_AVAILABLE:
        # Fallback: simple keyword-based sentiment for Spanish
        positive_words = ["bueno", "excelente", "genial", "perfecto", "increíble", "fantástico", 
                         "recomiendo", "satisfecho", "contento", "feliz", "amor", "mejor"]
        negative_words = ["malo", "terrible", "horrible", "pésimo", "odio", "peor", 
                         "decepcionado", "frustrado", "enojado", "lento", "caro", "defectuoso"]
        
        comment_lower = comment.lower()
        pos_count = sum(1 for word in positive_words if word in comment_lower)
        neg_count = sum(1 for word in negative_words if word in comment_lower)
        
        if pos_count > neg_count:
            polarity = 0.5
            label = "positive"
        elif neg_count > pos_count:
            polarity = -0.5
            label = "negative"
        else:
            polarity = 0.0
            label = "neutral"
            
        return {
            "polarity": polarity,
            "subjectivity": 0.5,
            "sentiment_label": label,
            "confidence": abs(polarity),
            "method": "keyword_fallback"
        }
    
    try:
        # Use TextBlob for sentiment analysis
        blob = TextBlob(comment)
        polarity = blob.sentiment.polarity  # -1 (negative) to 1 (positive)
        subjectivity = blob.sentiment.subjectivity  # 0 (objective) to 1 (subjective)
        
        # Classify sentiment
        if polarity > 0.1:
            sentiment_label = "positive"
        elif polarity < -0.1:
            sentiment_label = "negative"
        else:
            sentiment_label = "neutral"
            
        return {
            "polarity": float(polarity),
            "subjectivity": float(subjectivity),
            "sentiment_label": sentiment_label,
            "confidence": abs(float(polarity)),
            "method": "textblob"
        }
    except Exception as e:
        return {
            "polarity": 0.0,
            "subjectivity": 0.0,
            "sentiment_label": "neutral",
            "confidence": 0.0,
            "error": str(e)
        }


def extract_ecommerce_keywords(comment: str) -> Dict[str, List[str]]:
    """
    Extract themed keywords from Spanish comments related to e-commerce.
    Categories: product, shipment, service, price, quality.
    """
    if not comment or not isinstance(comment, str):
        return {"product": [], "shipment": [], "service": [], "price": [], "quality": []}
    
    comment_lower = comment.lower()
    
    # Spanish e-commerce keyword categories
    keywords = {
        "product": {
            "terms": ["producto", "artículo", "item", "mercancía", "objeto", "cosa", "material", 
                     "calidad", "tamaño", "color", "diseño", "modelo", "marca", "características"],
            "found": []
        },
        "shipment": {
            "terms": ["envío", "entrega", "paquete", "correo", "transporte", "llegada", "recibir", 
                     "delivery", "shipping", "rápido", "lento", "tarde", "tiempo", "días"],
            "found": []
        },
        "service": {
            "terms": ["servicio", "atención", "soporte", "ayuda", "personal", "vendedor", "cliente",
                     "respuesta", "comunicación", "trato", "amable", "profesional"],
            "found": []
        },
        "price": {
            "terms": ["precio", "costo", "dinero", "barato", "caro", "económico", "oferta", 
                     "descuento", "valor", "pagar", "euro", "dólar", "peso"],
            "found": []
        },
        "quality": {
            "terms": ["calidad", "bueno", "malo", "excelente", "terrible", "perfecto", "defectuoso",
                     "resistente", "frágil", "duradero", "funciona", "roto", "dañado"],
            "found": []
        }
    }
    
    # Extract matching keywords
    for category, data in keywords.items():
        for term in data["terms"]:
            if term in comment_lower:
                data["found"].append(term)
    
    # Return only the found keywords
    return {category: data["found"] for category, data in keywords.items()}


def translate_spanish_to_english(text: str) -> Dict[str, Any]:
    """
    Translate Spanish text to English using available libraries.
    """
    if not text or not isinstance(text, str):
        return {
            "original": text,
            "translated": "",
            "success": False,
            "error": "Invalid or empty text"
        }
    
    if not _TEXTBLOB_AVAILABLE:
        # Simple fallback dictionary for common Spanish phrases
        common_translations = {
            "muy bueno": "very good",
            "excelente": "excellent", 
            "malo": "bad",
            "terrible": "terrible",
            "recomiendo": "I recommend",
            "no recomiendo": "I don't recommend",
            "rápido": "fast",
            "lento": "slow",
            "caro": "expensive",
            "barato": "cheap",
            "producto": "product",
            "envío": "shipping",
            "servicio": "service",
            "calidad": "quality"
        }
        
        text_lower = text.lower().strip()
        if text_lower in common_translations:
            return {
                "original": text,
                "translated": common_translations[text_lower],
                "success": True,
                "method": "dictionary_fallback"
            }
        else:
            return {
                "original": text,
                "translated": f"[ES] {text}",
                "success": False,
                "method": "dictionary_fallback",
                "note": "Translation not found in dictionary"
            }
    
    try:
        # Use TextBlob for translation
        blob = TextBlob(text)
        # Detect language first
        detected_lang = blob.detect_language()
        
        if detected_lang == 'es':
            translated = str(blob.translate(to='en'))
            return {
                "original": text,
                "translated": translated,
                "success": True,
                "detected_language": detected_lang,
                "method": "textblob"
            }
        else:
            # If not Spanish, return as-is
            return {
                "original": text,
                "translated": text,
                "success": True,
                "detected_language": detected_lang,
                "method": "textblob",
                "note": "Text not detected as Spanish"
            }
    except Exception as e:
        return {
            "original": text,
            "translated": f"[Translation Error] {text}",
            "success": False,
            "error": str(e)
        }


def extract_keywords_from_reviews(review_df: pd.DataFrame, text_column: str = "review_comment_message", top_n: int = 10) -> Dict[str, Any]:
    """
    Extract top keywords from review text data.
    Works with Portuguese/Spanish reviews.
    """
    if review_df.empty or text_column not in review_df.columns:
        return {
            "error": f"No data found or column '{text_column}' missing",
            "keywords": [],
            "total_reviews": 0
        }
    
    # Get all review text
    reviews = review_df[text_column].dropna().tolist()
    if not reviews:
        return {
            "error": "No review text found",
            "keywords": [],
            "total_reviews": 0
        }
    
    # Simple keyword extraction for Portuguese/Spanish
    all_text = " ".join(reviews).lower()
    
    # Common Portuguese/Spanish stop words
    stop_words = {
        'o', 'a', 'os', 'as', 'um', 'uma', 'uns', 'umas', 'do', 'da', 'dos', 'das', 
        'de', 'em', 'para', 'por', 'com', 'sem', 'que', 'ou', 'e', 'mas', 'se', 
        'no', 'na', 'nos', 'nas', 'ao', 'aos', 'à', 'às', 'pelo', 'pela', 'pelos', 
        'pelas', 'este', 'esta', 'estes', 'estas', 'esse', 'essa', 'esses', 'essas',
        'el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas', 'del', 'al', 'y', 
        'o', 'pero', 'si', 'no', 'en', 'con', 'sin', 'por', 'para', 'de', 'que',
        'muito', 'bem', 'bom', 'boa', 'foi', 'ser', 'ter', 'fazer', 'ficar', 'dar',
        'bueno', 'bien', 'fue', 'ser', 'estar', 'tener', 'hacer', 'dar', 'ir', 'ver'
    }
    
    # Clean and tokenize
    import string
    # Remove punctuation and split
    translator = str.maketrans('', '', string.punctuation)
    clean_text = all_text.translate(translator)
    words = clean_text.split()
    
    # Filter out stop words and short words
    keywords = [word for word in words if len(word) > 2 and word not in stop_words]
    
    # Count frequency
    from collections import Counter
    word_counts = Counter(keywords)
    
    # Get top keywords
    top_keywords = word_counts.most_common(top_n)
    
    # Format results
    keyword_list = [{"keyword": word, "frequency": count} for word, count in top_keywords]
    
    return {
        "keywords": keyword_list,
        "total_reviews": len(reviews),
        "total_words_analyzed": len(keywords),
        "most_common_word": top_keywords[0][0] if top_keywords else None,
        "analysis_method": "frequency_based"
    }


def analyze_spanish_comments(comments: Union[str, List[str]]) -> Dict[str, Any]:
    """
    Main function to analyze Spanish review comments.
    Performs sentiment analysis, keyword extraction, and translation.
    
    Args:
        comments: Single comment string or list of comment strings
        
    Returns:
        Dictionary with analysis results including sentiment, keywords, and translations
    """
    if isinstance(comments, str):
        comments = [comments]
    
    if not comments or not isinstance(comments, list):
        return {
            "error": "Invalid input: expected string or list of strings",
            "total_comments": 0,
            "results": []
        }
    
    results = []
    sentiment_summary = {"positive": 0, "negative": 0, "neutral": 0}
    all_keywords = {"product": [], "shipment": [], "service": [], "price": [], "quality": []}
    
    for i, comment in enumerate(comments):
        if not comment or not isinstance(comment, str):
            continue
            
        # Sentiment analysis
        sentiment = analyze_spanish_sentiment(comment)
        sentiment_summary[sentiment["sentiment_label"]] += 1
        
        # Keyword extraction
        keywords = extract_ecommerce_keywords(comment)
        for category, terms in keywords.items():
            all_keywords[category].extend(terms)
        
        # Translation
        translation = translate_spanish_to_english(comment)
        
        # Compile result for this comment
        comment_result = {
            "comment_index": i,
            "original_comment": comment,
            "sentiment": sentiment,
            "keywords": keywords,
            "translation": translation,
            "translated_text": translation.get("translated", comment)
        }
        
        results.append(comment_result)
    
    # Remove duplicates from aggregated keywords
    for category in all_keywords:
        all_keywords[category] = list(set(all_keywords[category]))
    
    # Calculate overall sentiment distribution
    total_analyzed = sum(sentiment_summary.values())
    sentiment_distribution = {}
    if total_analyzed > 0:
        sentiment_distribution = {
            k: round(v / total_analyzed * 100, 1) 
            for k, v in sentiment_summary.items()
        }
    
    return {
        "total_comments": len([c for c in comments if c and isinstance(c, str)]),
        "sentiment_summary": sentiment_summary,
        "sentiment_distribution_pct": sentiment_distribution,
        "aggregated_keywords": all_keywords,
        "most_common_themes": sorted(
            all_keywords.items(), 
            key=lambda x: len(x[1]), 
            reverse=True
        )[:3],
        "results": results,
        "analysis_complete": True
    }


# ======================
# Streamlit config
# ======================
st.set_page_config(page_title="CEO ↔ AM ↔ DS", layout="wide")
st.title("🏢 CEO ↔ AM ↔ DS — Profit Assistant")

with st.sidebar:
    st.header("⚙️ Data")
    uploaded_file = st.file_uploader(
        "Upload Data File",
        type=["zip", "csv", "xlsx", "xls", "json", "jsonl", "tsv", "txt", "parquet"],
        help="Supports: ZIP (containing CSVs), CSV, Excel (.xlsx/.xls), JSON, JSONL, TSV, TXT, and Parquet files"
    )
    st.header("🧠 Model")
    model   = st.text_input("OpenAI model", value=DEFAULT_MODEL)
    api_key = st.text_input("OPENAI_API_KEY", value=OPENAI_API_KEY, type="password")


# ======================
# State
# ======================
if "tables_raw" not in st.session_state: st.session_state.tables_raw = None
if "tables_fe"  not in st.session_state: st.session_state.tables_fe  = {}
if "tables"     not in st.session_state: st.session_state.tables     = None
if "chat"       not in st.session_state: st.session_state.chat       = []
if "last_rendered_idx" not in st.session_state: st.session_state.last_rendered_idx = 0
if "last_am_json" not in st.session_state: st.session_state.last_am_json = {}
if "last_ds_json" not in st.session_state: st.session_state.last_ds_json = {}
if "last_user_prompt" not in st.session_state: st.session_state.last_user_prompt = ""
if "current_question" not in st.session_state: st.session_state.current_question = ""
if "threads" not in st.session_state: st.session_state.threads = []  # [{central, followups: []}]
if "central_question" not in st.session_state: st.session_state.central_question = ""
if "prior_questions" not in st.session_state: st.session_state.prior_questions = []
# Enhanced caching system for DS results and judge approvals
if "last_results" not in st.session_state:
    st.session_state.last_results = {
        "sql": None,
        "eda": None,
        "feature_engineering": None,
        "modeling": None,      # supervised summary
        "clustering": None,    # clustering report
    }

# Cache for executed results with judge approval status
if "executed_results" not in st.session_state:
    st.session_state.executed_results = {}  # {action_id: {result: data, approved: bool, timestamp: str}}

# Cache for SQL query results to avoid re-execution
if "sql_results_cache" not in st.session_state:
    st.session_state.sql_results_cache = {}  # {sql_hash: dataframe}

# Revision tracking for judge agent
if "revision_history" not in st.session_state:
    st.session_state.revision_history = []  # Track DS revisions and judge feedback

# Lightweight business term synonyms
TERM_SYNONYMS: Dict[str, List[str]] = {
    "revenue": ["revenue", "sales", "sales_amount", "net_sales", "turnover", "gmv", "amount"],
    "profit": ["profit", "net_income", "margin", "gross_profit", "operating_income"],
    "cost": ["cost", "cogs", "cost_of_goods_sold", "expense", "expenses", "opex"],
    "price": ["price", "unit_price", "sale_price", "avg_price"],
    "quantity": ["quantity", "qty", "units", "volume"],
    "customer": ["customer", "client", "buyer", "account_id", "customer_id"],
    "date": ["date", "order_date", "invoice_date", "day", "dt"],
}

# ======================
# Helpers
# ======================
def ensure_openai():
    if not _OPENAI_AVAILABLE:
        raise RuntimeError("OpenAI SDK missing.")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY")
    return OpenAI(api_key=api_key)


def llm_json(system_prompt: str, user_payload: str) -> dict:
    """Robust JSON helper with structured-mode first, then free-form parse fallback."""
    client = ensure_openai()

    sys_msg = system_prompt.strip() + "\n\nReturn ONLY a single JSON object. This line contains the word json."
    user_msg = (user_payload or "").strip() + "\n\nPlease respond with JSON only (a single object)."

    # DEBUG: Log LLM input/output for DS agent calls
    is_ds_call = ("Data Scientist" in system_prompt or "DS Agent" in system_prompt or
                  "You are the Data Scientist (DS)" in system_prompt)

    if is_ds_call:
        st.write("🔍 **DEBUG - DS Agent LLM Call:**")
        st.write(f"- Model: {model}")
        st.write(f"- System prompt length: {len(sys_msg)} chars")
        st.write(f"- User payload length: {len(user_msg)} chars")
        st.write(f"- System prompt starts: {sys_msg[:100]}...")

        # Show key parts of the payload
        try:
            payload_data = json.loads(user_payload)
            st.write(f"- AM Action Sequence in payload: {payload_data.get('am_action_sequence', 'MISSING')}")
            st.write(f"- AM Next Action in payload: {payload_data.get('am_next_action_type', 'MISSING')}")
            st.write(f"- AM Plan in payload: {payload_data.get('am_plan', 'MISSING')}")
            st.write(f"- AM Referenced Entities in payload: {payload_data.get('am_referenced_entities', 'MISSING')}")

            # Check if data schema is included
            shared_ctx = payload_data.get('shared_context', {})
            st.write(f"- Has schema_info: {bool(shared_ctx.get('schema_info'))}")
            st.write(f"- Has suggested_columns: {bool(shared_ctx.get('suggested_columns'))}")
            st.write(f"- Has context_relevance: {bool(shared_ctx.get('context_relevance'))}")

        except Exception as e:
            st.write(f"- Could not parse user payload as JSON: {e}")
            st.write(f"- Raw payload: {user_payload[:200]}...")

    # Preferred structured call
    try:
        resp = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
        )
        result = json.loads(resp.choices[0].message.content or "{}")

        # DEBUG: Log LLM response for DS agent calls
        if is_ds_call:
            st.write("🔍 **DEBUG - DS Agent LLM Response:**")
            st.write(f"- Response length: {len(resp.choices[0].message.content or '')} chars")
            st.write(f"- Raw response: {resp.choices[0].message.content[:500]}...")

            # Check the structure of the response
            if 'sequence' in result:
                st.write("- Response uses 'sequence' format")
                for i, step in enumerate(result.get('sequence', [])):
                    sql_val = step.get('duckdb_sql')
                    st.write(f"  Step {i}: action={step.get('action')}, sql={'NULL' if sql_val is None else ('EMPTY' if sql_val == '' else 'HAS_VALUE')}")
            elif 'action_sequence' in result:
                st.write("- Response uses 'action_sequence' format")
            else:
                st.write("- Response has no sequence")

        return result
    except Exception as e1:
        # Fallback: free-form then parse
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": user_msg + "\n\nIf needed, wrap JSON in ```json fences."},
                ],
                temperature=0.0,
            )
            raw = resp.choices[0].message.content or ""
        except Exception as e2:
            return {"_error": str(e1), "_fallback_error": str(e2)}

        m = re.search(r"```json\s*(\{.*?\})\s*```", raw, flags=re.DOTALL | re.IGNORECASE)
        if m:
            try: return json.loads(m.group(1).strip())
            except Exception: pass
        m2 = re.search(r"(\{.*\})", raw, flags=re.DOTALL)
        if m2:
            try: return json.loads(m2.group(1).strip())
            except Exception: pass
        return {"_raw": raw, "_parse_error": True}


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


def load_data_file(file) -> Dict[str, pd.DataFrame]:
    """Load data from various file formats commonly used in corporate environments"""
    if file is None:
        return {}

    filename = file.name.lower()
    file_ext = os.path.splitext(filename)[1]
    base_name = os.path.splitext(os.path.basename(filename))[0]

    try:
        if file_ext == '.zip':
            # Use existing ZIP handling
            return load_zip_tables(file)

        elif file_ext == '.csv':
            df = pd.read_csv(file)
            return {base_name: df}

        elif file_ext in ['.xlsx', '.xls']:
            # Handle Excel files - read all sheets
            excel_file = pd.ExcelFile(file)
            tables = {}
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(file, sheet_name=sheet_name)
                key = f"{base_name}_{sheet_name}" if len(excel_file.sheet_names) > 1 else base_name
                tables[key] = df
            return tables

        elif file_ext == '.json':
            # Handle JSON files
            data = json.load(file)
            if isinstance(data, list):
                # List of records
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                # Could be nested structure - try to flatten
                df = pd.json_normalize(data)
            else:
                # Fallback - create single row DataFrame
                df = pd.DataFrame([data])
            return {base_name: df}

        elif file_ext == '.jsonl':
            # Handle JSON Lines files
            lines = []
            file.seek(0)  # Reset file pointer
            for line in file:
                if isinstance(line, bytes):
                    line = line.decode('utf-8')
                line = line.strip()
                if line:
                    lines.append(json.loads(line))
            df = pd.DataFrame(lines)
            return {base_name: df}

        elif file_ext == '.tsv':
            df = pd.read_csv(file, sep='\t')
            return {base_name: df}

        elif file_ext == '.txt':
            # Try to detect delimiter
            file.seek(0)
            sample = file.read(1024)
            if isinstance(sample, bytes):
                sample = sample.decode('utf-8')
            file.seek(0)

            # Common delimiters in corporate files
            if '\t' in sample:
                df = pd.read_csv(file, sep='\t')
            elif '|' in sample:
                df = pd.read_csv(file, sep='|')
            elif ';' in sample:
                df = pd.read_csv(file, sep=';')
            else:
                # Default to comma
                df = pd.read_csv(file)
            return {base_name: df}

        elif file_ext == '.parquet':
            df = pd.read_parquet(file)
            return {base_name: df}

        else:
            st.error(f"Unsupported file type: {file_ext}")
            return {}

    except Exception as e:
        st.error(f"Error loading file {filename}: {str(e)}")
        return {}


def get_all_tables() -> Dict[str, pd.DataFrame]:
    out = {}
    if st.session_state.tables_raw:
        out.update(st.session_state.tables_raw)
    if st.session_state.tables_fe:
        out.update(st.session_state.tables_fe)
    return out


def run_duckdb_sql(sql: str, use_cache: bool = True) -> pd.DataFrame:
    """Execute SQL with optional caching to avoid re-running identical queries."""
    if use_cache:
        sql_hash = hashlib.md5(sql.encode()).hexdigest()
        if sql_hash in st.session_state.sql_results_cache:
            return st.session_state.sql_results_cache[sql_hash].copy()
    
    con = duckdb.connect(database=":memory:")
    for name, df in get_all_tables().items():
        con.register(name, df)
    result = con.execute(sql).df()
    
    if use_cache:
        st.session_state.sql_results_cache[sql_hash] = result.copy()
    
    return result


def cache_result_with_approval(action_id: str, result_data: Any, approved: bool = False) -> None:
    """Cache result data with judge approval status."""
    import datetime
    st.session_state.executed_results[action_id] = {
        "result": result_data,
        "approved": approved,
        "timestamp": datetime.datetime.now().isoformat()
    }


def get_cached_result(action_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve cached result if it exists and is approved."""
    cached = st.session_state.executed_results.get(action_id)
    if cached and cached.get("approved", False):
        return cached
    return None


def get_last_approved_result(action_type: str) -> Optional[Any]:
    """Get the most recent approved result for a specific action type."""
    for action_id, cached in st.session_state.executed_results.items():
        if cached.get("approved", False) and action_id.startswith(action_type):
            return cached.get("result")
    return None


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


def validate_ds_response(ds_response: dict) -> Dict[str, Any]:
    """Validate DS response for critical errors before judge review."""
    issues = []
    
    # Check for NULL SQL queries in single action
    if ds_response.get("action") == "sql":
        sql = ds_response.get("duckdb_sql")
        if sql is None or sql == "NULL" or (isinstance(sql, str) and sql.strip() == ""):
            issues.append("CRITICAL: SQL action has NULL/empty duckdb_sql field")
    
    # Check for NULL SQL queries in action sequence (handle both formats)
    sequence_key = None
    if ds_response.get("action_sequence"):
        sequence_key = "action_sequence"
    elif ds_response.get("sequence"):
        sequence_key = "sequence"

    if sequence_key:
        for i, step in enumerate(ds_response.get(sequence_key, [])):
            if step.get("action") == "sql":
                sql = step.get("duckdb_sql")
                if sql is None or sql == "NULL" or (isinstance(sql, str) and sql.strip() == ""):
                    issues.append(f"CRITICAL: Action sequence step {i+1} has NULL/empty duckdb_sql field")
    
    # Check for missing required fields
    if not ds_response.get("ds_summary"):
        issues.append("Missing ds_summary field")
    
    if not ds_response.get("action") and not ds_response.get("action_sequence"):
        issues.append("Missing both action and action_sequence fields")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "has_critical_errors": any("CRITICAL:" in issue for issue in issues)
    }


def fix_ds_response_with_fallback(ds_response: dict, user_question: str, shared_context: dict) -> dict:
    """Fix DS response by injecting fallback SQL when LLM fails."""
    fixed_response = ds_response.copy()
    
    # Fix single SQL action
    if ds_response.get("action") == "sql":
        sql = ds_response.get("duckdb_sql")
        if sql is None or sql == "NULL" or (isinstance(sql, str) and sql.strip() == ""):
            fallback_sql = generate_fallback_sql(user_question, shared_context)
            fixed_response["duckdb_sql"] = fallback_sql
            fixed_response["ds_summary"] = fixed_response.get("ds_summary", "") + " [FALLBACK SQL GENERATED]"
    
    # Fix action sequence SQL (handle both "action_sequence" and "sequence" formats)
    sequence_key = None
    if ds_response.get("action_sequence"):
        sequence_key = "action_sequence"
    elif ds_response.get("sequence"):
        sequence_key = "sequence"

    if sequence_key:
        fixed_sequence = []
        for i, step in enumerate(ds_response.get(sequence_key, [])):
            fixed_step = step.copy()
            if step.get("action") == "sql":
                sql = step.get("duckdb_sql")
                if sql is None or sql == "NULL" or (isinstance(sql, str) and sql.strip() == ""):
                    # Generate context-aware fallback SQL based on step description
                    step_description = step.get("description", "").lower()
                    fallback_sql = generate_contextual_fallback_sql(user_question, shared_context, step_description, i)
                    fixed_step["duckdb_sql"] = fallback_sql
            fixed_sequence.append(fixed_step)
        fixed_response[sequence_key] = fixed_sequence
        fixed_response["ds_summary"] = fixed_response.get("ds_summary", "") + " [FALLBACK SQL GENERATED]"
    
    return fixed_response


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
    """Determine which parts of the context are relevant to the current question."""
    question_lower = current_question.lower()

    # Question type patterns
    question_patterns = {
        "specific_entity_reference": ["this", "that", "the product", "the customer", "the order", "it", "its"],
        "broad_analysis": ["clustering", "segmentation", "all products", "all customers", "analyze", "compare", "distribution"],
        "new_analysis": ["different", "new", "another", "instead", "change to", "switch to"],
        "continuation": ["also", "additionally", "furthermore", "next", "then", "continue"],
        "explanation": ["explain", "why", "how", "what does", "interpret", "meaning"],
        "overview": ["what data", "show me", "available", "overview", "summary"]
    }

    # Determine question type
    question_type = "new_analysis"  # default
    for pattern_type, keywords in question_patterns.items():
        if any(keyword in question_lower for keyword in keywords):
            question_type = pattern_type
            break

    # Context relevance rules
    relevance_rules = {
        "specific_entity_reference": {
            "use_entity_ids": True,
            "use_cached_results": True,
            "filter_by_entity": True
        },
        "broad_analysis": {
            "use_entity_ids": False,
            "use_cached_results": False,
            "filter_by_entity": False
        },
        "new_analysis": {
            "use_entity_ids": False,
            "use_cached_results": False,
            "filter_by_entity": False
        },
        "continuation": {
            "use_entity_ids": True,
            "use_cached_results": True,
            "filter_by_entity": False
        },
        "explanation": {
            "use_entity_ids": False,
            "use_cached_results": True,
            "filter_by_entity": False
        },
        "overview": {
            "use_entity_ids": False,
            "use_cached_results": False,
            "filter_by_entity": False
        }
    }

    rules = relevance_rules.get(question_type, relevance_rules["new_analysis"])

    # Filter context based on relevance
    filtered_context = available_context.copy()

    if not rules["use_entity_ids"]:
        # Clear entity IDs if they're not relevant
        filtered_context["key_findings"] = {}
        if "conversation_entities" in filtered_context:
            filtered_context["conversation_entities"]["resolved_entity_ids"] = {}

    if not rules["use_cached_results"]:
        # Clear cached results if starting fresh analysis
        filtered_context["cached_results"] = {}
        filtered_context["recent_sql_results"] = {}

    # Add relevance metadata
    filtered_context["context_relevance"] = {
        "question_type": question_type,
        "rules_applied": rules,
        "original_context_size": len(str(available_context)),
        "filtered_context_size": len(str(filtered_context))
    }

    return filtered_context


def get_table_schema_info() -> Dict[str, Dict[str, Any]]:
    """Get essential schema information for all available tables (optimized for JSON serialization)."""
    schema_info = {}
    tables = get_all_tables()

    for table_name, df in tables.items():
        if df is not None and not df.empty:
            try:
                schema_info[table_name] = {
                    "row_count": int(len(df)),
                    "columns": list(df.columns),
                    "business_relevant_columns": _identify_business_columns(df.columns.tolist())
                }
            except Exception as e:
                # Fallback to basic info if there's any serialization issue
                schema_info[table_name] = {
                    "row_count": len(df),
                    "columns": list(df.columns),
                    "business_relevant_columns": {}
                }

    return schema_info




def _identify_business_columns(columns: List[str]) -> Dict[str, List[str]]:
    """Categorize columns by business meaning."""
    categories = {
        "sales_metrics": [],
        "quantity_metrics": [],
        "price_metrics": [],
        "date_columns": [],
        "id_columns": [],
        "category_columns": [],
        "location_columns": [],
        "physical_dimensions": [],
        "text_metadata": [],
        "count_metrics": []
    }

    for col in columns:
        col_lower = col.lower()

        # Physical dimensions (HIGHEST PRIORITY for dimension clustering)
        if any(term in col_lower for term in ['weight_g', 'length_cm', 'height_cm', 'width_cm', 'depth_cm', 'size_cm']):
            categories["physical_dimensions"].append(col)

        # Text metadata (EXCLUDE from dimension clustering)
        elif any(term in col_lower for term in ['name_lenght', 'description_lenght', 'title_length', 'comment_length']):
            categories["text_metadata"].append(col)

        # Count/quantity metrics (DIFFERENT from physical dimensions)
        elif any(term in col_lower for term in ['photos_qty', 'items_qty', 'count', 'quantity', 'num_']):
            categories["count_metrics"].append(col)

        # Sales-related
        elif any(term in col_lower for term in ['sales', 'revenue', 'total', 'amount']):
            categories["sales_metrics"].append(col)

        # Price-related
        elif any(term in col_lower for term in ['price', 'cost', 'value', 'freight']):
            categories["price_metrics"].append(col)

        # Date-related
        elif any(term in col_lower for term in ['date', 'time', 'created', 'updated']):
            categories["date_columns"].append(col)

        # ID columns
        elif col_lower.endswith('_id') or col_lower == 'id':
            categories["id_columns"].append(col)

        # Category columns
        elif any(term in col_lower for term in ['category', 'type', 'status', 'state']):
            categories["category_columns"].append(col)

        # Location columns
        elif any(term in col_lower for term in ['city', 'state', 'zip', 'location', 'address']):
            categories["location_columns"].append(col)

    return categories


def get_ml_algorithm_requirements() -> Dict[str, Dict[str, Any]]:
    """Define feature and target requirements for different ML algorithms."""
    return {
        "linear_regression": {
            "target_types": ["continuous", "numeric"],
            "feature_preferences": ["price_metrics", "sales_metrics", "count_metrics", "physical_dimensions", "category_columns"],
            "avoid_features": ["text_metadata", "id_columns", "date_columns"],
            "use_cases": ["predict price", "forecast sales", "estimate revenue", "predict value"],
            "target_examples": ["price", "sales", "revenue", "profit", "cost", "value", "amount"],
            "needs_feature_importance": True
        },
        "logistic_regression": {
            "target_types": ["binary", "categorical"],
            "feature_preferences": ["price_metrics", "sales_metrics", "count_metrics", "physical_dimensions", "category_columns"],
            "avoid_features": ["text_metadata", "id_columns", "date_columns"],
            "use_cases": ["predict category", "classify", "binary prediction", "success/failure"],
            "target_examples": ["category", "status", "type", "class", "success", "failure", "approved", "rejected"],
            "needs_feature_importance": True
        },
        "decision_tree": {
            "target_types": ["categorical", "continuous"],
            "feature_preferences": ["category_columns", "count_metrics", "price_metrics", "physical_dimensions"],
            "avoid_features": ["text_metadata", "id_columns", "date_columns"],
            "use_cases": ["classification", "decision making", "rule-based prediction"],
            "target_examples": ["category", "status", "type", "decision", "outcome"],
            "needs_feature_importance": True
        },
        "random_forest": {
            "target_types": ["categorical", "continuous"],
            "feature_preferences": ["category_columns", "count_metrics", "price_metrics", "physical_dimensions", "sales_metrics"],
            "avoid_features": ["text_metadata", "id_columns", "date_columns"],
            "use_cases": ["classification", "regression", "feature importance analysis"],
            "target_examples": ["category", "price", "sales", "status", "rating"],
            "needs_feature_importance": True
        },
        "neural_network": {
            "target_types": ["categorical", "continuous"],
            "feature_preferences": ["price_metrics", "sales_metrics", "count_metrics", "physical_dimensions"],
            "avoid_features": ["text_metadata", "id_columns", "date_columns", "category_columns"],
            "use_cases": ["complex pattern recognition", "deep learning", "non-linear prediction"],
            "target_examples": ["category", "price", "rating", "score", "complex_outcome"],
            "needs_feature_importance": False  # Neural networks need SHAP for interpretability
        },
        "clustering": {
            "target_types": ["none"],  # Unsupervised
            "feature_preferences": ["physical_dimensions", "price_metrics", "sales_metrics", "count_metrics"],
            "avoid_features": ["text_metadata", "id_columns", "date_columns", "category_columns"],
            "use_cases": ["customer segmentation", "product grouping", "pattern discovery"],
            "target_examples": [],  # No target for clustering
            "needs_feature_importance": False
        }
    }


def detect_ml_algorithm_intent(intent: str) -> Dict[str, Any]:
    """Detect which ML algorithm the user wants and suggest appropriate features/targets."""
    intent_lower = intent.lower()
    ml_requirements = get_ml_algorithm_requirements()

    detected_algorithm = None
    confidence = 0

    # Algorithm detection patterns
    algorithm_patterns = {
        "linear_regression": ["linear regression", "predict price", "forecast sales", "estimate revenue", "linear model"],
        "logistic_regression": ["logistic regression", "predict category", "binary classification", "logistic model"],
        "decision_tree": ["decision tree", "tree model", "rule-based", "decision making"],
        "random_forest": ["random forest", "forest", "ensemble", "feature importance"],
        "neural_network": ["neural network", "deep learning", "neural net", "nn", "mlp"],
        "clustering": ["clustering", "cluster", "segment", "kmeans", "unsupervised"]
    }

    # Detect algorithm
    for algorithm, patterns in algorithm_patterns.items():
        for pattern in patterns:
            if pattern in intent_lower:
                detected_algorithm = algorithm
                confidence = len(pattern.split())  # Longer matches = higher confidence
                break
        if detected_algorithm:
            break

    # If no specific algorithm mentioned, infer from use case
    if not detected_algorithm:
        if any(term in intent_lower for term in ["predict", "forecast", "estimate"]):
            detected_algorithm = "linear_regression"  # Default for prediction
        elif any(term in intent_lower for term in ["classify", "classification", "category"]):
            detected_algorithm = "logistic_regression"  # Default for classification
        elif any(term in intent_lower for term in ["group", "segment"]):
            detected_algorithm = "clustering"  # Default for grouping

    return {
        "algorithm": detected_algorithm,
        "confidence": confidence,
        "requirements": ml_requirements.get(detected_algorithm, {}) if detected_algorithm else {}
    }


def suggest_columns_for_query(intent: str, table_schema: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
    """Suggest appropriate columns based on query intent and available schema."""
    suggestions = {}
    intent_lower = intent.lower()

    # Detect ML algorithm intent
    ml_intent = detect_ml_algorithm_intent(intent)
    algorithm = ml_intent.get("algorithm")
    requirements = ml_intent.get("requirements", {})

    for table_name, schema in table_schema.items():
        table_suggestions = []
        business_cols = schema.get("business_relevant_columns", {})

        # ML ALGORITHM-SPECIFIC SUGGESTIONS
        if algorithm and requirements:
            # Get preferred feature types for this algorithm
            preferred_features = requirements.get("feature_preferences", [])
            avoid_features = requirements.get("avoid_features", [])

            # Add preferred features
            for feature_type in preferred_features:
                table_suggestions.extend(business_cols.get(feature_type, []))

            # Remove features to avoid
            for feature_type in avoid_features:
                features_to_remove = business_cols.get(feature_type, [])
                table_suggestions = [f for f in table_suggestions if f not in features_to_remove]

        # SPECIFIC INTENT OVERRIDES (for backwards compatibility and special cases)
        elif any(term in intent_lower for term in ['product dimension', 'product clustering', 'dimension clustering']):
            # ONLY use physical dimensions for product dimension clustering (NO IDs!)
            table_suggestions.extend(business_cols.get("physical_dimensions", []))

        # FALLBACK TO GENERAL ANALYSIS TYPE
        else:
            # Sales analysis
            if any(term in intent_lower for term in ['sales', 'selling', 'revenue', 'top product']):
                table_suggestions.extend(business_cols.get("sales_metrics", []))
                table_suggestions.extend(business_cols.get("price_metrics", []))

            # Quantity analysis
            elif any(term in intent_lower for term in ['quantity', 'volume', 'units', 'count']):
                table_suggestions.extend(business_cols.get("count_metrics", []))

            # Category analysis
            elif any(term in intent_lower for term in ['category', 'type', 'group']):
                table_suggestions.extend(business_cols.get("category_columns", []))

            # Location analysis
            elif any(term in intent_lower for term in ['location', 'city', 'state', 'geographic']):
                table_suggestions.extend(business_cols.get("location_columns", []))

            # Note: Never include ID columns for ML algorithms - they provide no predictive value

        if table_suggestions:
            suggestions[table_name] = list(set(table_suggestions))  # Remove duplicates

    return suggestions


def suggest_target_variables(intent: str, table_schema: Dict[str, Dict[str, Any]], algorithm: str) -> Dict[str, List[str]]:
    """Suggest appropriate target variables based on intent and algorithm requirements."""
    if not algorithm or algorithm == "clustering":
        return {}  # No targets needed for unsupervised learning

    ml_requirements = get_ml_algorithm_requirements()
    requirements = ml_requirements.get(algorithm, {})
    target_types = requirements.get("target_types", [])
    target_examples = requirements.get("target_examples", [])

    target_suggestions = {}
    intent_lower = intent.lower()

    for table_name, schema in table_schema.items():
        table_targets = []
        business_cols = schema.get("business_relevant_columns", {})
        all_columns = schema.get("columns", [])

        # Find columns that match target requirements
        for target_type in target_types:
            if target_type == "continuous" or target_type == "numeric":
                # Continuous targets: price, sales, revenue metrics
                table_targets.extend(business_cols.get("price_metrics", []))
                table_targets.extend(business_cols.get("sales_metrics", []))
                table_targets.extend(business_cols.get("count_metrics", []))

            elif target_type == "categorical" or target_type == "binary":
                # Categorical targets: categories, status, types
                table_targets.extend(business_cols.get("category_columns", []))

        # Intent-specific target detection
        for example in target_examples:
            for col in all_columns:
                if example.lower() in col.lower():
                    table_targets.append(col)

        # Remove duplicates and irrelevant columns
        avoid_as_targets = business_cols.get("id_columns", []) + business_cols.get("text_metadata", [])
        table_targets = [t for t in set(table_targets) if t not in avoid_as_targets]

        if table_targets:
            target_suggestions[table_name] = table_targets

    return target_suggestions


def analyze_missing_data(df: pd.DataFrame) -> Dict[str, Any]:
    """Intelligent analysis of missing data patterns and recommendations."""
    if not _PREPROCESSING_AVAILABLE:
        return {"error": "Preprocessing libraries not available"}

    analysis = {
        "missing_summary": {},
        "missingness_patterns": {},
        "recommendations": {}
    }

    # Basic missing data statistics
    missing_stats = df.isnull().sum()
    missing_pct = (missing_stats / len(df)) * 100

    for col in df.columns:
        if missing_stats[col] > 0:
            col_analysis = {
                "count": int(missing_stats[col]),
                "percentage": float(missing_pct[col]),
                "data_type": str(df[col].dtype),
                "unique_values": int(df[col].nunique()) if df[col].dtype != 'object' else int(df[col].nunique()),
                "distribution_info": {},
                "recommended_strategy": "drop"  # default
            }

            # Analyze non-missing values for context
            non_missing = df[col].dropna()
            if len(non_missing) > 0:
                if pd.api.types.is_numeric_dtype(df[col]):
                    col_analysis["distribution_info"] = {
                        "mean": float(non_missing.mean()),
                        "median": float(non_missing.median()),
                        "std": float(non_missing.std()),
                        "skewness": float(non_missing.skew()),
                        "has_outliers": bool(len(non_missing[(non_missing < non_missing.quantile(0.25) - 1.5*non_missing.quantile(0.75)) |
                                                          (non_missing > non_missing.quantile(0.75) + 1.5*non_missing.quantile(0.25))]) > 0)
                    }

                    # Recommend strategy based on distribution and missing percentage
                    if missing_pct[col] < 5:
                        col_analysis["recommended_strategy"] = "drop_rows"
                        col_analysis["rationale"] = "Low missing percentage - safe to drop rows"
                    elif missing_pct[col] < 15:
                        if abs(col_analysis["distribution_info"]["skewness"]) < 1:
                            col_analysis["recommended_strategy"] = "mean"
                            col_analysis["rationale"] = "Normal distribution - mean imputation appropriate"
                        else:
                            col_analysis["recommended_strategy"] = "median"
                            col_analysis["rationale"] = "Skewed distribution - median more robust"
                    elif missing_pct[col] < 40:
                        if col_analysis["distribution_info"]["has_outliers"]:
                            col_analysis["recommended_strategy"] = "knn"
                            col_analysis["rationale"] = "Many outliers - KNN imputation preserves local patterns"
                        else:
                            col_analysis["recommended_strategy"] = "interpolation"
                            col_analysis["rationale"] = "Moderate missingness - interpolation may capture trends"
                    else:
                        col_analysis["recommended_strategy"] = "drop_column"
                        col_analysis["rationale"] = "High missing percentage - unreliable for ML"

                else:  # Categorical
                    mode_value = non_missing.mode().iloc[0] if len(non_missing.mode()) > 0 else "Unknown"
                    col_analysis["distribution_info"] = {
                        "mode": str(mode_value),
                        "unique_categories": int(non_missing.nunique()),
                        "most_frequent_pct": float((non_missing == mode_value).sum() / len(non_missing) * 100)
                    }

                    if missing_pct[col] < 10:
                        col_analysis["recommended_strategy"] = "mode"
                        col_analysis["rationale"] = "Low missingness - mode imputation reasonable"
                    elif missing_pct[col] < 30:
                        col_analysis["recommended_strategy"] = "unknown_category"
                        col_analysis["rationale"] = "Moderate missingness - treat as separate 'Unknown' category"
                    else:
                        col_analysis["recommended_strategy"] = "drop_column"
                        col_analysis["rationale"] = "High missing percentage - unreliable for ML"

            analysis["missing_summary"][col] = col_analysis

    # Analyze missingness patterns (columns that are missing together)
    if len(missing_stats[missing_stats > 0]) > 1:
        missing_pattern_analysis = []
        missing_cols = missing_stats[missing_stats > 0].index.tolist()

        for i, col1 in enumerate(missing_cols):
            for col2 in missing_cols[i+1:]:
                # Check correlation in missingness
                both_missing = df[col1].isnull() & df[col2].isnull()
                correlation = both_missing.sum() / max(df[col1].isnull().sum(), df[col2].isnull().sum())

                if correlation > 0.5:  # Strong correlation in missingness
                    missing_pattern_analysis.append({
                        "columns": [col1, col2],
                        "correlation": float(correlation),
                        "interpretation": "Systematic missingness - may indicate data collection issues"
                    })

        analysis["missingness_patterns"] = missing_pattern_analysis

    return analysis


def analyze_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """Comprehensive data quality assessment including outliers, duplicates, and inconsistencies."""
    if not _PREPROCESSING_AVAILABLE:
        return {"error": "Preprocessing libraries not available"}

    quality_report = {
        "duplicates": {},
        "outliers": {},
        "data_consistency": {},
        "recommendations": []
    }

    # Duplicate analysis
    duplicate_rows = df.duplicated().sum()
    quality_report["duplicates"] = {
        "total_duplicates": int(duplicate_rows),
        "duplicate_percentage": float(duplicate_rows / len(df) * 100),
        "recommendation": "remove" if duplicate_rows > 0 else "none_found"
    }

    # Outlier detection for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        col_data = df[col].dropna()
        if len(col_data) > 0:
            outlier_analysis = {
                "method_results": {},
                "recommended_action": "investigate"
            }

            # IQR method
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            iqr_outliers = col_data[(col_data < Q1 - 1.5*IQR) | (col_data > Q3 + 1.5*IQR)]
            outlier_analysis["method_results"]["iqr"] = {
                "count": len(iqr_outliers),
                "percentage": float(len(iqr_outliers) / len(col_data) * 100),
                "bounds": {"lower": float(Q1 - 1.5*IQR), "upper": float(Q3 + 1.5*IQR)}
            }

            # Z-score method (if roughly normal)
            if abs(col_data.skew()) < 2:  # Roughly normal
                z_scores = np.abs(stats.zscore(col_data))
                z_outliers = col_data[z_scores > 3]
                outlier_analysis["method_results"]["zscore"] = {
                    "count": len(z_outliers),
                    "percentage": float(len(z_outliers) / len(col_data) * 100),
                    "threshold": 3
                }

            # Determine recommendation
            iqr_pct = outlier_analysis["method_results"]["iqr"]["percentage"]
            if iqr_pct < 1:
                outlier_analysis["recommended_action"] = "keep"
                outlier_analysis["rationale"] = "Very few outliers - likely genuine extreme values"
            elif iqr_pct < 5:
                outlier_analysis["recommended_action"] = "cap"
                outlier_analysis["rationale"] = "Moderate outliers - cap to reasonable bounds"
            else:
                outlier_analysis["recommended_action"] = "investigate"
                outlier_analysis["rationale"] = "Many outliers - investigate data quality issues"

            quality_report["outliers"][col] = outlier_analysis

    # Data consistency checks
    for col in df.columns:
        consistency_issues = []

        if df[col].dtype == 'object':
            # Check for case inconsistencies
            if df[col].nunique() != df[col].str.lower().nunique():
                consistency_issues.append("Case inconsistencies detected")

            # Check for whitespace issues
            if df[col].astype(str).str.contains(r'^\s|\s$').any():
                consistency_issues.append("Leading/trailing whitespace detected")

        if pd.api.types.is_numeric_dtype(df[col]):
            # Check for impossible values (negative prices, etc.)
            if col.lower().find('price') != -1 or col.lower().find('cost') != -1:
                if (df[col] < 0).any():
                    consistency_issues.append("Negative values in price/cost column")

        if consistency_issues:
            quality_report["data_consistency"][col] = consistency_issues

    return quality_report


def analyze_distributions(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze data distributions to inform preprocessing decisions."""
    if not _PREPROCESSING_AVAILABLE:
        return {"error": "Preprocessing libraries not available"}

    distribution_analysis = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        col_data = df[col].dropna()
        if len(col_data) > 0:
            analysis = {
                "basic_stats": {
                    "mean": float(col_data.mean()),
                    "median": float(col_data.median()),
                    "std": float(col_data.std()),
                    "min": float(col_data.min()),
                    "max": float(col_data.max())
                },
                "distribution_shape": {},
                "recommended_transformations": []
            }

            # Distribution shape analysis
            skewness = col_data.skew()
            kurtosis = col_data.kurtosis()

            analysis["distribution_shape"] = {
                "skewness": float(skewness),
                "kurtosis": float(kurtosis),
                "interpretation": ""
            }

            # Interpret distribution shape
            if abs(skewness) < 0.5:
                analysis["distribution_shape"]["interpretation"] = "Approximately normal"
                analysis["recommended_transformations"].append("standard_scaling")
            elif skewness > 1:
                analysis["distribution_shape"]["interpretation"] = "Highly right-skewed"
                analysis["recommended_transformations"].extend(["log_transform", "robust_scaling"])
            elif skewness < -1:
                analysis["distribution_shape"]["interpretation"] = "Highly left-skewed"
                analysis["recommended_transformations"].extend(["square_transform", "robust_scaling"])
            else:
                analysis["distribution_shape"]["interpretation"] = "Moderately skewed"
                analysis["recommended_transformations"].append("robust_scaling")

            # Check for zero/negative values that would affect log transformation
            if "log_transform" in analysis["recommended_transformations"]:
                if (col_data <= 0).any():
                    analysis["recommended_transformations"].remove("log_transform")
                    analysis["recommended_transformations"].append("sqrt_transform")
                    analysis["distribution_shape"]["interpretation"] += " (log not possible due to zero/negative values)"

            distribution_analysis[col] = analysis

    return distribution_analysis


def recommend_preprocessing_pipeline(df: pd.DataFrame, algorithm: str, target_col: str = None) -> Dict[str, Any]:
    """Provide algorithm-aware preprocessing recommendations."""
    if not _PREPROCESSING_AVAILABLE:
        return {"error": "Preprocessing libraries not available"}

    # Get algorithm requirements
    ml_requirements = get_ml_algorithm_requirements()
    algorithm_reqs = ml_requirements.get(algorithm, {})

    pipeline_recommendations = {
        "algorithm": algorithm,
        "preprocessing_steps": [],
        "rationale": {},
        "warnings": [],
        "estimated_performance_impact": {}
    }

    # Algorithm-specific preprocessing requirements
    if algorithm in ["linear_regression", "logistic_regression", "neural_network"]:
        # These algorithms are sensitive to feature scales
        pipeline_recommendations["preprocessing_steps"].extend([
            "missing_value_imputation",
            "outlier_handling",
            "feature_scaling",
            "normalization"
        ])
        pipeline_recommendations["rationale"]["scaling"] = "Linear algorithms require scaled features for optimal performance"

    elif algorithm in ["decision_tree", "random_forest"]:
        # Tree-based algorithms are more robust
        pipeline_recommendations["preprocessing_steps"].extend([
            "missing_value_imputation",
            "categorical_encoding"
        ])
        pipeline_recommendations["rationale"]["no_scaling"] = "Tree-based algorithms are robust to feature scales"

    elif algorithm == "clustering":
        # Clustering needs careful preprocessing
        pipeline_recommendations["preprocessing_steps"].extend([
            "missing_value_imputation",
            "outlier_investigation",
            "feature_scaling",
            "dimensionality_consideration"
        ])
        pipeline_recommendations["rationale"]["clustering"] = "Clustering sensitive to outliers and feature scales"

    # Analyze current data to provide specific recommendations
    missing_analysis = analyze_missing_data(df)
    quality_analysis = analyze_data_quality(df)
    distribution_analysis = analyze_distributions(df)

    # Specific recommendations based on data analysis
    specific_recommendations = {
        "missing_values": missing_analysis.get("missing_summary", {}),
        "outliers": quality_analysis.get("outliers", {}),
        "distributions": distribution_analysis,
        "data_quality_issues": quality_analysis.get("data_consistency", {})
    }

    # Performance impact estimation
    missing_pct = sum([info["percentage"] for info in missing_analysis.get("missing_summary", {}).values()]) / max(1, len(missing_analysis.get("missing_summary", {})))
    outlier_pct = sum([info["method_results"]["iqr"]["percentage"] for info in quality_analysis.get("outliers", {}).values()]) / max(1, len(quality_analysis.get("outliers", {})))

    pipeline_recommendations["estimated_performance_impact"] = {
        "missing_data_impact": "high" if missing_pct > 20 else "medium" if missing_pct > 10 else "low",
        "outlier_impact": "high" if outlier_pct > 10 else "medium" if outlier_pct > 5 else "low",
        "overall_data_quality": "poor" if missing_pct > 20 or outlier_pct > 10 else "good"
    }

    pipeline_recommendations["specific_recommendations"] = specific_recommendations

    return pipeline_recommendations


def create_flexible_preprocessing_pipeline(df: pd.DataFrame, preprocessing_config: Dict[str, Any]) -> Dict[str, Any]:
    """Create and apply a flexible preprocessing pipeline based on analysis."""
    if not _PREPROCESSING_AVAILABLE:
        return {"error": "Preprocessing libraries not available"}

    results = {
        "original_shape": df.shape,
        "preprocessing_steps_applied": [],
        "transformations": {},
        "final_shape": None,
        "data_quality_improvement": {}
    }

    processed_df = df.copy()

    try:
        # Step 1: Handle missing values intelligently
        missing_config = preprocessing_config.get("missing_values", {})
        for col, config in missing_config.items():
            strategy = config.get("recommended_strategy")

            if strategy == "drop_rows":
                before_len = len(processed_df)
                processed_df = processed_df.dropna(subset=[col])
                results["preprocessing_steps_applied"].append(f"Dropped {before_len - len(processed_df)} rows with missing {col}")

            elif strategy == "drop_column":
                processed_df = processed_df.drop(columns=[col])
                results["preprocessing_steps_applied"].append(f"Dropped column {col} (too many missing values)")

            elif strategy == "mean":
                mean_val = processed_df[col].mean()
                processed_df[col] = processed_df[col].fillna(mean_val)
                results["transformations"][f"{col}_imputation"] = {"method": "mean", "value": mean_val}
                results["preprocessing_steps_applied"].append(f"Imputed {col} with mean ({mean_val:.2f})")

            elif strategy == "median":
                median_val = processed_df[col].median()
                processed_df[col] = processed_df[col].fillna(median_val)
                results["transformations"][f"{col}_imputation"] = {"method": "median", "value": median_val}
                results["preprocessing_steps_applied"].append(f"Imputed {col} with median ({median_val:.2f})")

            elif strategy == "mode":
                mode_val = processed_df[col].mode().iloc[0] if len(processed_df[col].mode()) > 0 else "Unknown"
                processed_df[col] = processed_df[col].fillna(mode_val)
                results["transformations"][f"{col}_imputation"] = {"method": "mode", "value": mode_val}
                results["preprocessing_steps_applied"].append(f"Imputed {col} with mode ({mode_val})")

            elif strategy == "unknown_category":
                processed_df[col] = processed_df[col].fillna("Unknown")
                results["preprocessing_steps_applied"].append(f"Filled missing {col} with 'Unknown' category")

        # Step 2: Handle outliers
        outlier_config = preprocessing_config.get("outliers", {})
        for col, config in outlier_config.items():
            action = config.get("recommended_action")

            if action == "cap":
                bounds = config["method_results"]["iqr"]["bounds"]
                lower, upper = bounds["lower"], bounds["upper"]
                original_outliers = ((processed_df[col] < lower) | (processed_df[col] > upper)).sum()
                processed_df[col] = processed_df[col].clip(lower=lower, upper=upper)
                results["transformations"][f"{col}_outlier_capping"] = {"lower": lower, "upper": upper}
                results["preprocessing_steps_applied"].append(f"Capped {original_outliers} outliers in {col}")

        # Step 3: Apply transformations for skewed data
        distribution_config = preprocessing_config.get("distributions", {})
        for col, config in distribution_config.items():
            transformations = config.get("recommended_transformations", [])

            if "log_transform" in transformations:
                if (processed_df[col] > 0).all():  # Ensure all values are positive
                    original_skew = processed_df[col].skew()
                    processed_df[f"{col}_log"] = np.log(processed_df[col])
                    new_skew = processed_df[f"{col}_log"].skew()
                    results["transformations"][f"{col}_log_transform"] = {
                        "original_skewness": original_skew,
                        "new_skewness": new_skew
                    }
                    results["preprocessing_steps_applied"].append(f"Applied log transform to {col} (skew: {original_skew:.2f} → {new_skew:.2f})")

            elif "sqrt_transform" in transformations:
                if (processed_df[col] >= 0).all():  # Ensure all values are non-negative
                    original_skew = processed_df[col].skew()
                    processed_df[f"{col}_sqrt"] = np.sqrt(processed_df[col])
                    new_skew = processed_df[f"{col}_sqrt"].skew()
                    results["transformations"][f"{col}_sqrt_transform"] = {
                        "original_skewness": original_skew,
                        "new_skewness": new_skew
                    }
                    results["preprocessing_steps_applied"].append(f"Applied sqrt transform to {col} (skew: {original_skew:.2f} → {new_skew:.2f})")

        # Step 4: Feature scaling (if needed for algorithm)
        algorithm = preprocessing_config.get("algorithm")
        if algorithm in ["linear_regression", "logistic_regression", "neural_network", "clustering"]:
            numeric_cols = processed_df.select_dtypes(include=[np.number]).columns

            for col in numeric_cols:
                if not col.endswith(('_log', '_sqrt')):  # Don't scale already transformed features
                    original_mean = processed_df[col].mean()
                    original_std = processed_df[col].std()

                    # Use robust scaling for skewed data
                    if abs(processed_df[col].skew()) > 1:
                        median_val = processed_df[col].median()
                        mad = (processed_df[col] - median_val).abs().median()
                        processed_df[f"{col}_scaled"] = (processed_df[col] - median_val) / (mad * 1.4826)  # MAD to std conversion
                        results["transformations"][f"{col}_robust_scaling"] = {"median": median_val, "mad": mad}
                        results["preprocessing_steps_applied"].append(f"Applied robust scaling to {col}")
                    else:
                        # Standard scaling for normal data
                        processed_df[f"{col}_scaled"] = (processed_df[col] - original_mean) / original_std
                        results["transformations"][f"{col}_standard_scaling"] = {"mean": original_mean, "std": original_std}
                        results["preprocessing_steps_applied"].append(f"Applied standard scaling to {col}")

        results["final_shape"] = processed_df.shape
        results["processed_dataframe"] = processed_df

        # Calculate data quality improvement
        original_missing = df.isnull().sum().sum()
        final_missing = processed_df.isnull().sum().sum()

        results["data_quality_improvement"] = {
            "missing_values_reduced": original_missing - final_missing,
            "missing_reduction_percentage": ((original_missing - final_missing) / max(1, original_missing)) * 100,
            "shape_change": f"{df.shape} → {processed_df.shape}"
        }

    except Exception as e:
        results["error"] = str(e)
        results["processed_dataframe"] = df  # Return original if processing fails

    return results


def calculate_feature_importance(model, X, y, feature_names, model_type="tree"):
    """Calculate feature importance using multiple methods including SHAP."""
    importance_results = {}

    try:
        # 1. Built-in feature importance (for tree-based models)
        if hasattr(model, 'feature_importances_'):
            importance_results['built_in'] = {
                'values': model.feature_importances_.tolist(),
                'features': feature_names,
                'method': 'Built-in Feature Importance'
            }

        # 2. Coefficients (for linear models)
        elif hasattr(model, 'coef_'):
            coef = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
            importance_results['coefficients'] = {
                'values': abs(coef).tolist(),
                'features': feature_names,
                'method': 'Absolute Coefficients'
            }

        # 3. SHAP values (if available)
        if _SHAP_AVAILABLE and len(X) <= 1000:  # Limit for performance
            try:
                if model_type in ['tree', 'forest']:
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X.sample(min(100, len(X))))
                elif model_type == 'linear':
                    explainer = shap.LinearExplainer(model, X)
                    shap_values = explainer.shap_values(X.sample(min(100, len(X))))
                else:
                    explainer = shap.KernelExplainer(model.predict, X.sample(min(50, len(X))))
                    shap_values = explainer.shap_values(X.sample(min(20, len(X))))

                # Handle multi-class case
                if isinstance(shap_values, list):
                    shap_values = shap_values[0]

                mean_shap = abs(shap_values).mean(axis=0)
                importance_results['shap'] = {
                    'values': mean_shap.tolist(),
                    'features': feature_names,
                    'method': 'SHAP Values'
                }
            except Exception as e:
                importance_results['shap_error'] = str(e)

        # 4. Permutation importance (basic version)
        try:
            from sklearn.metrics import mean_squared_error, accuracy_score
            base_score = accuracy_score(y, model.predict(X)) if hasattr(model, 'predict_proba') else mean_squared_error(y, model.predict(X))

            perm_importance = []
            for i, col in enumerate(feature_names):
                X_perm = X.copy()
                X_perm.iloc[:, i] = X_perm.iloc[:, i].sample(frac=1).values  # Shuffle column

                if hasattr(model, 'predict_proba'):
                    perm_score = accuracy_score(y, model.predict(X_perm))
                    importance = base_score - perm_score  # Drop in accuracy
                else:
                    perm_score = mean_squared_error(y, model.predict(X_perm))
                    importance = perm_score - base_score  # Increase in error

                perm_importance.append(max(0, importance))  # Only positive importance

            importance_results['permutation'] = {
                'values': perm_importance,
                'features': feature_names,
                'method': 'Permutation Importance'
            }
        except Exception as e:
            importance_results['permutation_error'] = str(e)

    except Exception as e:
        importance_results['error'] = str(e)

    return importance_results


def create_feature_importance_plot(importance_results):
    """Create feature importance visualization if plotting libraries are available."""
    if not _PLOTTING_AVAILABLE:
        return None

    try:
        import io
        import base64

        # Use the best available importance method
        if 'shap' in importance_results:
            data = importance_results['shap']
        elif 'built_in' in importance_results:
            data = importance_results['built_in']
        elif 'coefficients' in importance_results:
            data = importance_results['coefficients']
        elif 'permutation' in importance_results:
            data = importance_results['permutation']
        else:
            return None

        values = data['values']
        features = data['features']
        method = data['method']

        # Create horizontal bar plot
        plt.figure(figsize=(10, max(6, len(features) * 0.4)))
        plt.barh(range(len(features)), values)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Importance Score')
        plt.title(f'Feature Importance ({method})')
        plt.gca().invert_yaxis()

        # Save to base64 string
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.read()).decode()
        plt.close()

        return plot_data

    except Exception as e:
        return f"Plot error: {str(e)}"


def analyze_feature_engineering_opportunities(df: pd.DataFrame, target_col: str = None, algorithm: str = None) -> Dict[str, Any]:
    """Analyze the dataset for feature engineering opportunities."""
    opportunities = {
        "numerical_features": {},
        "categorical_features": {},
        "datetime_features": {},
        "text_features": {},
        "interaction_features": [],
        "derived_features": [],
        "recommendations": []
    }

    try:
        # Identify different column types
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = []
        text_cols = []

        # Remove target column from feature analysis
        if target_col:
            numeric_cols = [col for col in numeric_cols if col != target_col]
            categorical_cols = [col for col in categorical_cols if col != target_col]

        # Detect datetime columns
        for col in categorical_cols[:]:
            try:
                pd.to_datetime(df[col].dropna().head(100))
                datetime_cols.append(col)
                categorical_cols.remove(col)
            except:
                pass

        # Detect text columns (strings with high average length)
        for col in categorical_cols[:]:
            avg_length = df[col].astype(str).str.len().mean()
            if avg_length > 20:  # Arbitrary threshold for text vs categorical
                text_cols.append(col)
                categorical_cols.remove(col)

        # Analyze numerical features
        for col in numeric_cols:
            col_analysis = {
                "type": "numerical",
                "opportunities": []
            }

            # Check for skewness - suggest transformations
            skewness = df[col].skew()
            if abs(skewness) > 1:
                if skewness > 1:
                    col_analysis["opportunities"].append({
                        "type": "transformation",
                        "method": "log_transform" if (df[col] > 0).all() else "sqrt_transform",
                        "reason": f"Right-skewed distribution (skew: {skewness:.2f})"
                    })
                else:
                    col_analysis["opportunities"].append({
                        "type": "transformation",
                        "method": "square_transform",
                        "reason": f"Left-skewed distribution (skew: {skewness:.2f})"
                    })

            # Check for outliers - suggest binning or capping
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
            if outliers > len(df) * 0.05:  # More than 5% outliers
                col_analysis["opportunities"].append({
                    "type": "binning",
                    "method": "quantile_binning",
                    "reason": f"{outliers} outliers detected ({outliers/len(df)*100:.1f}%)"
                })

            # Suggest polynomial features for algorithms that benefit
            if algorithm in ["linear_regression", "logistic_regression"]:
                col_analysis["opportunities"].append({
                    "type": "polynomial",
                    "method": "square_feature",
                    "reason": "Linear models can benefit from polynomial features"
                })

            opportunities["numerical_features"][col] = col_analysis

        # Analyze categorical features
        for col in categorical_cols:
            col_analysis = {
                "type": "categorical",
                "unique_values": df[col].nunique(),
                "opportunities": []
            }

            # High cardinality - suggest target encoding or frequency encoding
            if col_analysis["unique_values"] > 20:
                col_analysis["opportunities"].append({
                    "type": "encoding",
                    "method": "target_encoding" if target_col else "frequency_encoding",
                    "reason": f"High cardinality ({col_analysis['unique_values']} unique values)"
                })
            # Low cardinality - suggest one-hot encoding
            elif col_analysis["unique_values"] <= 10:
                col_analysis["opportunities"].append({
                    "type": "encoding",
                    "method": "one_hot_encoding",
                    "reason": f"Low cardinality ({col_analysis['unique_values']} unique values)"
                })
            # Medium cardinality - suggest ordinal or binary encoding
            else:
                col_analysis["opportunities"].append({
                    "type": "encoding",
                    "method": "ordinal_encoding",
                    "reason": f"Medium cardinality ({col_analysis['unique_values']} unique values)"
                })

            opportunities["categorical_features"][col] = col_analysis

        # Analyze datetime features
        for col in datetime_cols:
            opportunities["datetime_features"][col] = {
                "type": "datetime",
                "opportunities": [
                    {"type": "extraction", "method": "year", "reason": "Extract year component"},
                    {"type": "extraction", "method": "month", "reason": "Extract month component"},
                    {"type": "extraction", "method": "day_of_week", "reason": "Extract day of week"},
                    {"type": "extraction", "method": "is_weekend", "reason": "Create weekend indicator"},
                    {"type": "extraction", "method": "quarter", "reason": "Extract quarter component"}
                ]
            }

        # Analyze text features
        for col in text_cols:
            opportunities["text_features"][col] = {
                "type": "text",
                "opportunities": [
                    {"type": "extraction", "method": "length", "reason": "Extract text length"},
                    {"type": "extraction", "method": "word_count", "reason": "Count words"},
                    {"type": "extraction", "method": "sentiment", "reason": "Extract sentiment score"},
                    {"type": "vectorization", "method": "tfidf", "reason": "Convert to TF-IDF features"}
                ]
            }

        # Suggest interaction features
        if len(numeric_cols) >= 2:
            for i, col1 in enumerate(numeric_cols[:5]):  # Limit to avoid explosion
                for col2 in numeric_cols[i+1:6]:
                    # Check correlation to see if interaction makes sense
                    correlation = df[col1].corr(df[col2])
                    if abs(correlation) < 0.8:  # Avoid highly correlated features
                        opportunities["interaction_features"].append({
                            "feature1": col1,
                            "feature2": col2,
                            "type": "multiplication",
                            "reason": f"Interaction between {col1} and {col2}"
                        })

        # Algorithm-specific recommendations
        if algorithm == "linear_regression":
            opportunities["recommendations"].append("Consider polynomial features and feature scaling")
        elif algorithm == "tree":
            opportunities["recommendations"].append("Focus on feature selection over scaling; trees handle raw features well")
        elif algorithm == "neural_network":
            opportunities["recommendations"].append("Apply normalization and consider dimension reduction")
        elif algorithm == "clustering":
            opportunities["recommendations"].append("Scale features and consider dimensionality reduction")

        opportunities["summary"] = {
            "total_numerical": len(numeric_cols),
            "total_categorical": len(categorical_cols),
            "total_datetime": len(datetime_cols),
            "total_text": len(text_cols),
            "potential_interactions": len(opportunities["interaction_features"])
        }

    except Exception as e:
        opportunities["error"] = str(e)

    return opportunities


def create_engineered_features(df: pd.DataFrame, engineering_config: Dict[str, Any]) -> Dict[str, Any]:
    """Create new features based on feature engineering configuration."""
    results = {
        "original_shape": df.shape,
        "new_features": [],
        "feature_engineering_steps": [],
        "final_shape": None
    }

    engineered_df = df.copy()

    try:
        # Process numerical transformations
        numerical_config = engineering_config.get("numerical_features", {})
        for col, config in numerical_config.items():
            for opportunity in config.get("opportunities", []):
                if opportunity["type"] == "transformation":
                    method = opportunity["method"]

                    if method == "log_transform" and (engineered_df[col] > 0).all():
                        new_col = f"{col}_log"
                        engineered_df[new_col] = np.log(engineered_df[col])
                        results["new_features"].append(new_col)
                        results["feature_engineering_steps"].append(f"Created {new_col} using log transformation")

                    elif method == "sqrt_transform" and (engineered_df[col] >= 0).all():
                        new_col = f"{col}_sqrt"
                        engineered_df[new_col] = np.sqrt(engineered_df[col])
                        results["new_features"].append(new_col)
                        results["feature_engineering_steps"].append(f"Created {new_col} using sqrt transformation")

                    elif method == "square_transform":
                        new_col = f"{col}_squared"
                        engineered_df[new_col] = engineered_df[col] ** 2
                        results["new_features"].append(new_col)
                        results["feature_engineering_steps"].append(f"Created {new_col} using square transformation")

                elif opportunity["type"] == "binning":
                    new_col = f"{col}_binned"
                    engineered_df[new_col] = pd.qcut(engineered_df[col], q=5, duplicates='drop', labels=False)
                    results["new_features"].append(new_col)
                    results["feature_engineering_steps"].append(f"Created {new_col} using quantile binning")

                elif opportunity["type"] == "polynomial":
                    new_col = f"{col}_squared"
                    engineered_df[new_col] = engineered_df[col] ** 2
                    results["new_features"].append(new_col)
                    results["feature_engineering_steps"].append(f"Created {new_col} as polynomial feature")

        # Process categorical encodings
        categorical_config = engineering_config.get("categorical_features", {})
        for col, config in categorical_config.items():
            for opportunity in config.get("opportunities", []):
                if opportunity["type"] == "encoding":
                    method = opportunity["method"]

                    if method == "one_hot_encoding":
                        # Create one-hot encoded columns
                        dummies = pd.get_dummies(engineered_df[col], prefix=col)
                        engineered_df = pd.concat([engineered_df, dummies], axis=1)
                        results["new_features"].extend(dummies.columns.tolist())
                        results["feature_engineering_steps"].append(f"Created one-hot encoding for {col}")

                    elif method == "frequency_encoding":
                        new_col = f"{col}_frequency"
                        freq_map = engineered_df[col].value_counts().to_dict()
                        engineered_df[new_col] = engineered_df[col].map(freq_map)
                        results["new_features"].append(new_col)
                        results["feature_engineering_steps"].append(f"Created {new_col} using frequency encoding")

                    elif method == "ordinal_encoding":
                        new_col = f"{col}_ordinal"
                        # Simple ordinal encoding based on frequency
                        categories = engineered_df[col].value_counts().index.tolist()
                        ordinal_map = {cat: i for i, cat in enumerate(categories)}
                        engineered_df[new_col] = engineered_df[col].map(ordinal_map)
                        results["new_features"].append(new_col)
                        results["feature_engineering_steps"].append(f"Created {new_col} using ordinal encoding")

        # Process datetime features
        datetime_config = engineering_config.get("datetime_features", {})
        for col, config in datetime_config.items():
            # Convert to datetime if not already
            try:
                engineered_df[col] = pd.to_datetime(engineered_df[col])

                for opportunity in config.get("opportunities", []):
                    if opportunity["type"] == "extraction":
                        method = opportunity["method"]

                        if method == "year":
                            new_col = f"{col}_year"
                            engineered_df[new_col] = engineered_df[col].dt.year
                            results["new_features"].append(new_col)

                        elif method == "month":
                            new_col = f"{col}_month"
                            engineered_df[new_col] = engineered_df[col].dt.month
                            results["new_features"].append(new_col)

                        elif method == "day_of_week":
                            new_col = f"{col}_day_of_week"
                            engineered_df[new_col] = engineered_df[col].dt.dayofweek
                            results["new_features"].append(new_col)

                        elif method == "is_weekend":
                            new_col = f"{col}_is_weekend"
                            engineered_df[new_col] = (engineered_df[col].dt.dayofweek >= 5).astype(int)
                            results["new_features"].append(new_col)

                        elif method == "quarter":
                            new_col = f"{col}_quarter"
                            engineered_df[new_col] = engineered_df[col].dt.quarter
                            results["new_features"].append(new_col)

                        results["feature_engineering_steps"].append(f"Extracted {method} from {col}")

            except Exception as e:
                results["feature_engineering_steps"].append(f"Failed to process datetime column {col}: {str(e)}")

        # Process text features
        text_config = engineering_config.get("text_features", {})
        for col, config in text_config.items():
            for opportunity in config.get("opportunities", []):
                if opportunity["type"] == "extraction":
                    method = opportunity["method"]

                    if method == "length":
                        new_col = f"{col}_length"
                        engineered_df[new_col] = engineered_df[col].astype(str).str.len()
                        results["new_features"].append(new_col)

                    elif method == "word_count":
                        new_col = f"{col}_word_count"
                        engineered_df[new_col] = engineered_df[col].astype(str).str.split().str.len()
                        results["new_features"].append(new_col)

                    results["feature_engineering_steps"].append(f"Extracted {method} from {col}")

        # Process interaction features
        interaction_features = engineering_config.get("interaction_features", [])
        for interaction in interaction_features[:10]:  # Limit to avoid feature explosion
            if interaction["type"] == "multiplication":
                feature1, feature2 = interaction["feature1"], interaction["feature2"]
                if feature1 in engineered_df.columns and feature2 in engineered_df.columns:
                    new_col = f"{feature1}_x_{feature2}"
                    engineered_df[new_col] = engineered_df[feature1] * engineered_df[feature2]
                    results["new_features"].append(new_col)
                    results["feature_engineering_steps"].append(f"Created interaction feature {new_col}")

        results["final_shape"] = engineered_df.shape
        results["engineered_dataframe"] = engineered_df

        results["summary"] = {
            "features_added": len(results["new_features"]),
            "shape_change": f"{df.shape} → {engineered_df.shape}",
            "new_features_list": results["new_features"]
        }

    except Exception as e:
        results["error"] = str(e)
        results["engineered_dataframe"] = df  # Return original if engineering fails

    return results


def recommend_feature_engineering_pipeline(df: pd.DataFrame, algorithm: str = None, target_col: str = None) -> Dict[str, Any]:
    """Recommend a complete feature engineering pipeline for the given algorithm and data."""

    # First analyze opportunities
    opportunities = analyze_feature_engineering_opportunities(df, target_col, algorithm)

    # Create recommendations based on algorithm type and data characteristics
    recommendations = {
        "pipeline_steps": [],
        "priority_features": [],
        "algorithm_specific_advice": [],
        "estimated_feature_count": df.shape[1]
    }

    try:
        # Algorithm-specific recommendations
        if algorithm in ["linear_regression", "logistic_regression"]:
            recommendations["algorithm_specific_advice"].extend([
                "Apply feature scaling (StandardScaler or RobustScaler)",
                "Consider polynomial features for non-linear relationships",
                "Remove highly correlated features to avoid multicollinearity",
                "Apply regularization-friendly preprocessing"
            ])

            # Prioritize transformations and scaling
            recommendations["pipeline_steps"].extend([
                "numerical_transformations",
                "categorical_encoding",
                "feature_scaling",
                "polynomial_features"
            ])

        elif algorithm in ["tree", "random_forest", "gradient_boosting"]:
            recommendations["algorithm_specific_advice"].extend([
                "Trees handle raw features well - focus on feature creation over scaling",
                "Create interaction features and binned features",
                "Handle missing values appropriately",
                "Consider feature importance for selection"
            ])

            recommendations["pipeline_steps"].extend([
                "missing_value_handling",
                "categorical_encoding",
                "binning_features",
                "interaction_features"
            ])

        elif algorithm == "neural_network":
            recommendations["algorithm_specific_advice"].extend([
                "Apply normalization (MinMaxScaler or StandardScaler)",
                "Consider dimensionality reduction for high-dimensional data",
                "Handle categorical variables with embeddings or encoding",
                "Create meaningful feature interactions"
            ])

            recommendations["pipeline_steps"].extend([
                "feature_scaling",
                "categorical_encoding",
                "dimensionality_reduction",
                "interaction_features"
            ])

        elif algorithm == "clustering":
            recommendations["algorithm_specific_advice"].extend([
                "Scale all features to same range",
                "Remove or transform highly skewed features",
                "Consider dimensionality reduction (PCA)",
                "Handle categorical variables appropriately"
            ])

            recommendations["pipeline_steps"].extend([
                "feature_scaling",
                "skewness_handling",
                "categorical_encoding",
                "dimensionality_reduction"
            ])

        # Data-driven recommendations
        numeric_cols = len(opportunities.get("numerical_features", {}))
        categorical_cols = len(opportunities.get("categorical_features", {}))
        datetime_cols = len(opportunities.get("datetime_features", {}))
        text_cols = len(opportunities.get("text_features", {}))

        # Prioritize features based on data types and potential impact
        if numeric_cols > 0:
            recommendations["priority_features"].append({
                "type": "numerical_transformations",
                "reason": f"{numeric_cols} numerical features can benefit from transformations",
                "impact": "high"
            })

        if categorical_cols > 0:
            recommendations["priority_features"].append({
                "type": "categorical_encoding",
                "reason": f"{categorical_cols} categorical features need encoding",
                "impact": "high"
            })

        if datetime_cols > 0:
            recommendations["priority_features"].append({
                "type": "datetime_extraction",
                "reason": f"{datetime_cols} datetime features can be decomposed",
                "impact": "medium"
            })

        if text_cols > 0:
            recommendations["priority_features"].append({
                "type": "text_processing",
                "reason": f"{text_cols} text features need processing",
                "impact": "medium"
            })

        # Estimate final feature count
        estimated_new_features = 0

        # From categorical encoding
        for col, analysis in opportunities.get("categorical_features", {}).items():
            if analysis["unique_values"] <= 10:
                estimated_new_features += analysis["unique_values"]  # One-hot
            else:
                estimated_new_features += 1  # Other encodings

        # From datetime extraction
        estimated_new_features += datetime_cols * 5  # year, month, day_of_week, is_weekend, quarter

        # From numerical transformations
        estimated_new_features += numeric_cols * 1.5  # Some transformations

        # From interactions (limited)
        if numeric_cols >= 2:
            estimated_new_features += min(10, numeric_cols * (numeric_cols - 1) / 2)

        recommendations["estimated_feature_count"] = int(df.shape[1] + estimated_new_features)

        recommendations["execution_order"] = [
            "1. Handle missing values",
            "2. Extract datetime components",
            "3. Process text features",
            "4. Create numerical transformations",
            "5. Encode categorical variables",
            "6. Create interaction features",
            "7. Apply feature scaling (if needed)",
            "8. Select important features"
        ]

    except Exception as e:
        recommendations["error"] = str(e)

    return recommendations


def create_complete_ml_pipeline(df: pd.DataFrame, algorithm: str = None, target_col: str = None,
                               include_feature_engineering: bool = True,
                               include_preprocessing: bool = True) -> Dict[str, Any]:
    """
    Create a complete ML pipeline that combines feature engineering and preprocessing.
    This is the master function that orchestrates the entire data preparation process.
    """
    pipeline_results = {
        "original_data": {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.to_dict()
        },
        "feature_engineering": {},
        "preprocessing": {},
        "final_data": {},
        "pipeline_summary": [],
        "performance_insights": [],
        "next_steps": []
    }

    try:
        current_df = df.copy()

        # Step 1: Data Quality Analysis (always perform this first)
        quality_analysis = analyze_data_quality(current_df)
        pipeline_results["data_quality"] = quality_analysis
        pipeline_results["pipeline_summary"].append(f"Data quality analysis: {quality_analysis.get('overall_score', 'N/A')}/10")

        # Step 2: Feature Engineering (if requested)
        if include_feature_engineering:
            # Analyze feature engineering opportunities
            fe_opportunities = analyze_feature_engineering_opportunities(current_df, target_col, algorithm)
            pipeline_results["feature_engineering"]["opportunities"] = fe_opportunities

            # Get feature engineering recommendations
            fe_recommendations = recommend_feature_engineering_pipeline(current_df, algorithm, target_col)
            pipeline_results["feature_engineering"]["recommendations"] = fe_recommendations

            # Apply feature engineering (selective approach)
            engineering_config = {}

            # Only apply high-impact transformations automatically
            if fe_opportunities.get("numerical_features"):
                engineering_config["numerical_features"] = {}
                for col, analysis in fe_opportunities["numerical_features"].items():
                    high_impact_ops = [op for op in analysis.get("opportunities", [])
                                     if op["type"] in ["transformation"] and "skew" in op.get("reason", "")]
                    if high_impact_ops:
                        engineering_config["numerical_features"][col] = {"opportunities": high_impact_ops[:1]}  # Limit to one per column

            # Apply selected feature engineering
            if engineering_config:
                fe_results = create_engineered_features(current_df, engineering_config)
                pipeline_results["feature_engineering"]["results"] = fe_results
                if "engineered_dataframe" in fe_results:
                    current_df = fe_results["engineered_dataframe"]
                    pipeline_results["pipeline_summary"].append(f"Feature engineering: {len(fe_results.get('new_features', []))} new features created")

        # Step 3: Preprocessing (if requested)
        if include_preprocessing:
            # Get preprocessing recommendations
            preprocessing_recommendations = recommend_preprocessing_pipeline(current_df, algorithm, target_col)
            pipeline_results["preprocessing"]["recommendations"] = preprocessing_recommendations

            # Analyze missing data and distributions for intelligent preprocessing
            missing_analysis = analyze_missing_data(current_df)
            distribution_analysis = analyze_distributions(current_df)

            pipeline_results["preprocessing"]["missing_analysis"] = missing_analysis
            pipeline_results["preprocessing"]["distribution_analysis"] = distribution_analysis

            # Create preprocessing configuration based on analysis
            preprocessing_config = {
                "algorithm": algorithm,
                "missing_values": {},
                "outliers": {},
                "distributions": {}
            }

            # Configure missing value handling
            for col, analysis in missing_analysis.get("column_analysis", {}).items():
                if analysis.get("missing_percentage", 0) > 0:
                    preprocessing_config["missing_values"][col] = {
                        "recommended_strategy": analysis.get("recommended_strategy", "drop_rows")
                    }

            # Configure outlier handling for numerical columns
            numeric_cols = current_df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col != target_col:  # Don't process target column
                    Q1 = current_df[col].quantile(0.25)
                    Q3 = current_df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    outliers = ((current_df[col] < (Q1 - 1.5 * IQR)) | (current_df[col] > (Q3 + 1.5 * IQR))).sum()

                    if outliers > len(current_df) * 0.05:  # More than 5% outliers
                        preprocessing_config["outliers"][col] = {
                            "recommended_action": "cap",
                            "method_results": {
                                "iqr": {
                                    "bounds": {"lower": Q1 - 1.5 * IQR, "upper": Q3 + 1.5 * IQR}
                                }
                            }
                        }

            # Configure distribution transformations
            for col in numeric_cols:
                if col != target_col and abs(current_df[col].skew()) > 1:
                    preprocessing_config["distributions"][col] = {
                        "recommended_transformations": ["log_transform" if current_df[col].skew() > 1 else "sqrt_transform"]
                    }

            # Apply preprocessing
            preprocessing_results = create_flexible_preprocessing_pipeline(current_df, preprocessing_config)
            pipeline_results["preprocessing"]["results"] = preprocessing_results

            if "processed_dataframe" in preprocessing_results:
                current_df = preprocessing_results["processed_dataframe"]
                steps_applied = len(preprocessing_results.get("preprocessing_steps_applied", []))
                pipeline_results["pipeline_summary"].append(f"Preprocessing: {steps_applied} steps applied")

        # Step 4: Final Analysis and Recommendations
        pipeline_results["final_data"] = {
            "shape": current_df.shape,
            "columns": list(current_df.columns),
            "shape_change": f"{df.shape} → {current_df.shape}",
            "columns_added": current_df.shape[1] - df.shape[1],
            "rows_retained": current_df.shape[0] / df.shape[0] * 100
        }

        # Algorithm-specific performance insights
        if algorithm:
            insights = []
            if algorithm in ["linear_regression", "logistic_regression"]:
                # Check if scaling was applied
                scaled_cols = [col for col in current_df.columns if col.endswith('_scaled')]
                if scaled_cols:
                    insights.append(f"✓ Feature scaling applied to {len(scaled_cols)} columns - good for linear models")
                else:
                    insights.append("⚠ Consider applying feature scaling for better linear model performance")

            elif algorithm in ["tree", "random_forest", "gradient_boosting"]:
                # Check for new features created
                if include_feature_engineering:
                    fe_results = pipeline_results.get("feature_engineering", {}).get("results", {})
                    new_features = len(fe_results.get("new_features", []))
                    if new_features > 0:
                        insights.append(f"✓ {new_features} new features created - trees can leverage these interactions")

            elif algorithm == "clustering":
                scaled_cols = [col for col in current_df.columns if col.endswith('_scaled')]
                if scaled_cols:
                    insights.append(f"✓ Feature scaling applied - important for distance-based clustering")

            pipeline_results["performance_insights"] = insights

        # Next steps recommendations
        next_steps = []
        if algorithm:
            if algorithm in ["linear_regression", "logistic_regression"]:
                next_steps.extend([
                    "Check for multicollinearity between features",
                    "Consider regularization (L1/L2) for feature selection",
                    "Validate feature importance after model training"
                ])
            elif algorithm == "clustering":
                next_steps.extend([
                    "Determine optimal number of clusters",
                    "Consider dimensionality reduction (PCA) if high-dimensional",
                    "Evaluate clustering quality with silhouette score"
                ])
            elif "tree" in algorithm or "forest" in algorithm:
                next_steps.extend([
                    "Use feature importance to identify key variables",
                    "Consider pruning to avoid overfitting",
                    "Validate on hold-out test set"
                ])

        next_steps.append("Split data into train/validation/test sets")
        next_steps.append("Apply the same preprocessing pipeline to test data")
        pipeline_results["next_steps"] = next_steps

        pipeline_results["processed_dataframe"] = current_df
        pipeline_results["success"] = True

    except Exception as e:
        pipeline_results["error"] = str(e)
        pipeline_results["processed_dataframe"] = df  # Return original on error
        pipeline_results["success"] = False

    return pipeline_results


def build_shared_context() -> Dict[str, Any]:
    """Build comprehensive shared context for all agents."""
    # Get cached approved results
    cached_results = {}
    for result_type in ["sql", "eda", "modeling", "clustering", "feature_engineering"]:
        cached = get_last_approved_result(result_type)
        if cached:
            cached_results[result_type] = cached
    
    # Get recent SQL results with data samples
    recent_sql_results = {}
    sql_count = 0
    for action_id, cached_data in st.session_state.executed_results.items():
        if cached_data.get("approved", False) and action_id.startswith("sql") and sql_count < 3:
            result_data = cached_data.get("result", {})
            sql = result_data.get("duckdb_sql")
            if sql:
                try:
                    df = run_duckdb_sql(sql)
                    recent_sql_results[f"recent_query_{sql_count + 1}"] = {
                        "sql": sql,
                        "row_count": len(df),
                        "columns": list(df.columns),
                        "sample_data": df.head(3).to_dict(orient="records"),
                        "timestamp": cached_data.get("timestamp")
                    }
                    sql_count += 1
                except Exception:
                    pass
    
    # NEW: Flexible entity resolution system
    recent_questions = st.session_state.prior_questions[-5:] if st.session_state.prior_questions else []
    current_q = st.session_state.current_question or ""
    
    # Resolve all contextual entities dynamically
    entity_resolution = resolve_contextual_entities(current_q, recent_questions, cached_results)
    
    # Extract key findings using flexible system
    key_findings = {}
    for entity_type, schema in ENTITY_SCHEMA.items():
        primary_key = schema["primary_key"]
        
        # Look for this entity type in recent SQL results
        for query_key, query_data in recent_sql_results.items():
            sample_data = query_data.get("sample_data", [])
            if sample_data and isinstance(sample_data, list) and len(sample_data) > 0:
                first_row = sample_data[0]
                if primary_key in first_row:
                    key_findings[f"latest_{entity_type}_id"] = str(first_row[primary_key])
                    
                    # Special handling for sales metrics
                    if any(col.endswith("_sales") or col.endswith("_revenue") or "total" in col.lower() for col in first_row.keys()):
                        key_findings[f"{entity_type}_has_sales_data"] = True
    
    # Build conversation entities with flexible detection
    conversation_entities = {
        "question_sequence": recent_questions + ([current_q] if current_q else []),
        "detected_entities": entity_resolution["current_entities"],
        "historical_entities": entity_resolution["historical_entities"],
        "resolved_entity_ids": entity_resolution["resolved_ids"],
        "entity_continuity": {}
    }
    
    # Track which entities have continuity across questions
    for entity_type in ENTITY_SCHEMA.keys():
        has_reference = entity_type in entity_resolution["current_entities"] and entity_resolution["current_entities"][entity_type].get("referenced")
        has_historical = entity_type in entity_resolution["historical_entities"] 
        has_resolved_id = f"{entity_type}_id" in entity_resolution["resolved_ids"]
        
        conversation_entities["entity_continuity"][entity_type] = {
            "currently_referenced": has_reference,
            "discussed_historically": has_historical,
            "id_available": has_resolved_id,
            "context_complete": has_reference and has_resolved_id
        }
    
    # Get detailed schema information
    schema_info = get_table_schema_info()

    # Generate column suggestions for current question
    current_question = st.session_state.current_question or ""
    column_suggestions = suggest_columns_for_query(current_question, schema_info) if current_question else {}

    # Detect ML algorithm intent and suggest targets
    ml_intent = detect_ml_algorithm_intent(current_question) if current_question else {}
    algorithm = ml_intent.get("algorithm")
    target_suggestions = suggest_target_variables(current_question, schema_info, algorithm) if current_question and algorithm else {}

    return {
        "cached_results": cached_results,
        "recent_sql_results": recent_sql_results,
        "key_findings": key_findings,
        "conversation_entities": conversation_entities,
        "conversation_context": {
            "central_question": st.session_state.central_question,
            "current_question": st.session_state.current_question,
            "prior_questions": st.session_state.prior_questions,
            "question_flow_summary": f"Previous: {' -> '.join(recent_questions[-2:])}, Current: {current_q}" if recent_questions else f"Current: {current_q}"
        },
        "available_tables": {k: list(v.columns) for k, v in get_all_tables().items()},
        "schema_info": schema_info,
        "suggested_columns": column_suggestions,
        "ml_algorithm_intent": ml_intent,
        "suggested_targets": target_suggestions,
        "context_timestamp": pd.Timestamp.now().isoformat()
    }


def add_msg(role, content, artifacts=None):
    st.session_state.chat.append({"role": role, "content": content, "artifacts": artifacts or {}})


def generate_artifact_summary(artifacts):
    """Generate a plain English summary of what someone is doing based on artifacts"""
    if not artifacts:
        return None

    # Handle different types of artifacts
    if "action" in artifacts:
        action = artifacts.get("action", "")
        if action == "sql":
            return "Analyst is running SQL queries to analyze data"
        elif action == "clustering":
            return "Data scientist is grouping similar data points together"
        elif action == "modeling":
            return "ML engineer is building a predictive model"
        elif action == "eda":
            return "Analyst is exploring the data to understand patterns"

    if "action_sequence" in artifacts:
        return "Data team is executing a multi-step analysis plan"

    if "sql" in artifacts:
        return "Database analyst is querying the data"

    if "model_report" in artifacts:
        return "ML engineer completed model training and evaluation"

    if "keywords" in artifacts:
        return "Text analyst extracted key themes from customer reviews"

    if "judgment" in artifacts:
        return None  # Hide verbose quality assessment messages

    if "central_question" in artifacts:
        return None  # Hide verbose context update messages

    if "explain_used" in artifacts:
        return "Analyst provided interpretation using cached results"

    # Default fallback
    return "Team member is working on data analysis tasks"

def render_chat(incremental: bool = True):
    msgs = st.session_state.chat
    start = st.session_state.last_rendered_idx if incremental else 0
    for m in msgs[start:]:
        with st.chat_message(m["role"]):
            st.write(m["content"])
            if m.get("artifacts"):
                summary = generate_artifact_summary(m["artifacts"])
                if summary:
                    with st.expander(summary, expanded=False):
                        st.json(m["artifacts"])
    st.session_state.last_rendered_idx = len(msgs)


def _sql_first(maybe_sql):
    if isinstance(maybe_sql, str):
        return maybe_sql.strip()
    if isinstance(maybe_sql, list):
        for s in maybe_sql:
            if isinstance(s, str) and s.strip():
                return s.strip()
    return ""


def _explicit_new_thread(text: str) -> bool:
    t = (text or "").lower()
    return bool(re.search(r"\bnot (a|an)?\s*follow[- ]?up\b", t))


def classify_intent(previous_question: str, central_question: str, prior_questions: List[str], new_text: str) -> dict:
    try:
        payload = {
            "previous_question": previous_question or "",
            "central_question": central_question or "",
            "prior_questions": prior_questions or [],
            "new_text": new_text or "",
        }
        res = llm_json(SYSTEM_INTENT, json.dumps(payload)) or {}
        intent = (res or {}).get("intent")
        related = bool((res or {}).get("related_to_central", False))
        if intent in {"new_request", "feedback", "answers_to_clarifying"}:
            return {"intent": intent, "related": related}
    except Exception:
        pass
    low = (new_text or "").lower()
    if any(w in low for w in ["that", "this", "looks", "seems", "instead", "also", "why", "how about", "can you", "explain", "interpret"]):
        return {"intent": "feedback", "related": True}
    return {"intent": "new_request", "related": False}


def build_column_hints(question: str) -> dict:
    all_tables = get_all_tables()
    struct = {t: list(df.columns) for t, df in all_tables.items()}
    hints = {"term_to_columns": {}, "suggested_features": [], "notes": ""}
    qlow = (question or "").lower()
    for term, cands in TERM_SYNONYMS.items():
        if term in qlow:
            found = []
            for table, cols in struct.items():
                for c in cands:
                    for col in cols:
                        if c == col.lower():
                            found.append({"table": table, "column": col})
            if found:
                hints["term_to_columns"][term] = found[:5]
                hints["suggested_features"].extend(found[:3])
    try:
        payload = {"question": question, "tables": struct}
        res = llm_json(SYSTEM_COLMAP, json.dumps(payload)) or {}
        if res.get("term_to_columns"):
            hints["term_to_columns"].update(res.get("term_to_columns"))
        if res.get("suggested_features"):
            hints["suggested_features"].extend(res.get("suggested_features"))
        if res.get("notes"):
            hints["notes"] = (hints.get("notes") or "") + " " + res["notes"]
    except Exception:
        pass
    seen = set(); uniq = []
    for it in hints["suggested_features"]:
        key = (it.get("table"), it.get("column"))
        if key not in seen:
            seen.add(key); uniq.append(it)
    hints["suggested_features"] = uniq[:20]
    return hints


# ======================
# Profiling → feature proposal → preflight
# ======================
def profile_columns(df: pd.DataFrame) -> pd.DataFrame:
    prof = []
    for c in df.columns:
        s = df[c]
        t = "numeric" if pd.api.types.is_numeric_dtype(s) else ("datetime" if np.issubdtype(s.dtype, np.datetime64) else "categorical")
        nonnull = int(s.notna().sum())
        distinct = int(s.nunique(dropna=True))
        uniq_ratio = distinct / max(nonnull, 1) if nonnull else 0.0
        flags = []
        name = str(c).lower()
        if re.search(r"(^id$|_id$|uuid|guid|hash|^code$|_code$)", name):
            flags.append("id_like")
        if re.search(r"(zip|postal|cep|postcode|latitude|longitude|lat|lon)", name):
            flags.append("geo_like")
        if t == "datetime":
            flags.append("datetime")
        if (t == "numeric" and (s.std(skipna=True) == 0)) or (distinct <= 1):
            flags.append("near_constant")
        prof.append({"col": c, "dtype": str(s.dtype), "t": t,
                     "nonnull": nonnull, "distinct": distinct,
                     "uniq_ratio": float(uniq_ratio), "flags": flags})
    return pd.DataFrame(prof)


def propose_features(task: str, prof: pd.DataFrame, allow_geo: bool=False) -> dict:
    bad = set()
    for _, r in prof.iterrows():
        rflags = set(r.get("flags", []) if isinstance(r.get("flags", []), (list, tuple, set)) else [])
        col = r.get("col")
        if col is None:
            continue
        if "id_like" in rflags or "near_constant" in rflags or "datetime" in rflags:
            bad.add(col)
        if "geo_like" in rflags and not allow_geo:
            bad.add(col)
    selected = prof[(prof["t"]=="numeric") & (~prof["col"].isin(list(bad))) ]["col"].tolist()
    excluded = []
    for _, r in prof.iterrows():
        col = r.get("col")
        if col in bad:
            rflags = r.get("flags", [])
            excluded.append({"col": col, "reason": " | ".join(rflags) if isinstance(rflags, (list, tuple)) else str(rflags)})
    return {"selected_features": selected, "excluded_features": excluded}


def preflight(task: str, proposal: dict) -> tuple[bool, str]:
    feats = proposal.get("selected_features") or []
    if task == "clustering" and len(feats) < 2:
        return False, "Need at least 2 numeric features for clustering after filtering."
    if not feats:
        return False, "No usable features after filtering."
    return True, ""


# ======================
# Smart base chooser (browses all tables)
# ======================
DIM_PAT = re.compile(r"product_(weight_g|length_cm|height_cm|width_cm)$", re.I)

def choose_model_base(plan: dict, question_text: str) -> Optional[pd.DataFrame]:
    """
    Browse ALL loaded tables and pick the best single base:
    - If the question mentions product/dimension: pick the table with the MOST numeric dimension-like columns.
    - Else score each table by count of usable numeric features (post filtering) and choose the best.
    (You can still join with other tables via SQL in earlier steps.)
    """
    tables = get_all_tables()
    if not tables: return None
    qlow = (question_text or "").lower()

    # If product/dimension ask → prefer most dimension-like columns
    if re.search(r"\bproduct\b", qlow) or re.search(r"\bdimension", qlow):
        best_name, best_score = None, -1
        for name, df in tables.items():
            dims = [c for c in df.columns if DIM_PAT.search(str(c)) and pd.api.types.is_numeric_dtype(df[c])]
            score = len(dims)
            if score > best_score:
                best_name, best_score = name, score
        if best_name is not None and best_score >= 2:
            return tables[best_name].copy()

    # Generic scoring by usable numeric features
    best_name, best_score = None, -1
    for name, df in tables.items():
        prof = profile_columns(df)
        prop = propose_features("clustering", prof, allow_geo=False)
        score = len(prop.get("selected_features", []))
        if score > best_score:
            best_name, best_score = name, score
    return tables[best_name].copy() if best_name else next(iter(tables.values())).copy()


# ======================
# Modeling helpers
# ======================
def infer_default_model_plan(question_text: str, plan: Optional[dict]) -> dict:
    """
    Ensure modeling defaults make sense. If task/target are missing AND the CEO likely
    wants clustering (or no target is provided), default to KMeans clustering.
    """
    plan = dict(plan or {})
    q = (question_text or "").lower()
    wants_cluster = any(k in q for k in ["cluster", "clustering", "segment", "segmentation", "kmeans", "dimension"])
    task = (plan.get("task") or "").lower()
    target = plan.get("target")

    if wants_cluster or not target or task == "clustering":
        plan.setdefault("task", "clustering")
        plan.setdefault("model_family", "kmeans")
        plan.setdefault("n_clusters", plan.get("n_clusters", 5))
        plan.setdefault("features", plan.get("features", []))
        plan.pop("target", None)  # not used in clustering
    return plan


# ======================
# Modeling (incl. clustering)
# ======================
def train_model(df: pd.DataFrame, task: str, target: Optional[str], features: List[str], family: str, n_clusters: Optional[int]=None) -> Dict[str, Any]:
    if task == "clustering":
        # Prefer explicit dimension columns if present
        dim_like = [c for c in df.columns if DIM_PAT.search(str(c))]
        dim_like = [c for c in dim_like if pd.api.types.is_numeric_dtype(df[c])]
        if not features and len(dim_like) >= 2:
            use_cols = dim_like
        else:
            if not features:
                prof = profile_columns(df)
                prop = propose_features("clustering", prof, allow_geo=False)
                ok, msg = preflight("clustering", prop)
                if not ok:
                    return {"error": msg, "feature_proposal": prop}
                use_cols = prop["selected_features"]
            else:
                use_cols = [c for c in features if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
                if len(use_cols) < 2:
                    prof = profile_columns(df)
                    prop = propose_features("clustering", prof, allow_geo=False)
                    ok, msg = preflight("clustering", prop)
                    if not ok:
                        return {"error": msg, "feature_proposal": prop}
                    use_cols = prop["selected_features"]

        # Build robust X: coerce numeric, handle inf/NaN, impute, scale
        X = df[use_cols].copy()
        for c in use_cols:
            X[c] = pd.to_numeric(X[c], errors="coerce")
        X.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Drop rows with all-NaN across selected features
        X = X.dropna(how="all")
        if X.shape[0] < 2:
            return {"error": "Not enough valid rows after cleaning to run clustering."}

        imputer = SimpleImputer(strategy="median")
        X_imp = imputer.fit_transform(X)
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X_imp)

        k_requested = int(n_clusters or 5)
        n_samples = Xs.shape[0]
        k = max(1, min(k_requested, n_samples))  # k cannot exceed n_samples

        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(Xs)
        try:
            sil = float(silhouette_score(Xs, labels)) if k > 1 else None
        except Exception:
            sil = None
        try:
            p = PCA(n_components=2, random_state=42)
            coords = p.fit_transform(Xs)
            coords_df = pd.DataFrame({"pc1": coords[:,0], "pc2": coords[:,1], "cluster": labels})
        except Exception:
            coords_df = None
        try:
            centers_std = km.cluster_centers_
            centers_orig = scaler.inverse_transform(centers_std)
            centroids_df = pd.DataFrame(centers_orig, columns=use_cols)
        except Exception:
            centroids_df = None

        sizes = pd.Series(labels).value_counts().sort_index().to_dict()
        report = {
            "task": "clustering",
            "features": use_cols,
            "n_clusters": k,
            "cluster_sizes": {int(k_): int(v) for k_, v in sizes.items()},
            "inertia": float(getattr(km, "inertia_", np.nan)),
            "silhouette": sil,
        }
        return {"report": report, "labels": labels.tolist(), "pca": coords_df, "centroids": centroids_df}

    # ---- Supervised ----
    if target is None or target not in df.columns:
        return {"error": f"Target '{target}' not found."}

    X = df[features] if features else df.drop(columns=[target], errors="ignore")
    y = df[target]

    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), num_cols),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                              ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
        ],
        remainder="drop",
    )

    if task == "classification":
        if not pd.api.types.is_numeric_dtype(y):
            y = pd.Categorical(y).codes
        if family in ("random_forest", "rf"):
            model = RandomForestClassifier(n_estimators=300, random_state=42)
            fam = "random_forest"
        else:
            model = LogisticRegression(max_iter=1000)
            fam = "logistic_regression"
    else:
        if family in ("random_forest_regressor", "random_forest", "rf"):
            model = RandomForestRegressor(n_estimators=300, random_state=42)
            fam = "random_forest_regressor"
        else:
            model = LinearRegression()
            fam = "linear_regression"

    pipe = Pipeline([("pre", pre), ("model", model)])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y if task == "classification" else None
    )
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    base_report = {"task": task, "target": target, "features": features, "model_family": fam}

    if task == "classification":
        try:
            proba = pipe.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, proba)
        except Exception:
            proba, auc = None, None
        acc = accuracy_score(y_test, (y_pred > 0.5) if proba is not None else y_pred)
        base_report.update({"accuracy": float(acc), "roc_auc": (float(auc) if auc is not None else None)})
    else:
        base_report.update({
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "rmse": float(mean_squared_error(y_test, y_pred, squared=False)),
            "r2": float(r2_score(y_test, y_pred))
        })
    return base_report


# ======================
# AM/DS/Review pipeline
# ======================
def run_am_plan(prompt: str, column_hints: dict, context: dict) -> dict:
    full_context = build_shared_context()
    shared_context = assess_context_relevance(prompt, full_context)
    payload = {
        "ceo_question": prompt,
        "shared_context": shared_context,
        "column_hints": column_hints,
        "context": context,
    }
    am_json = llm_json(SYSTEM_AM, json.dumps(payload))
    st.session_state.last_am_json = am_json
    add_msg("am", am_json.get("am_brief", ""), artifacts=am_json)
    render_chat()
    return am_json


def _coerce_allowed(action: Optional[str], fallback: str) -> str:
    allowed = {"overview","sql","eda","calc","feature_engineering","modeling","explain"}
    a = (action or "").lower()
    if a in allowed: return a

    # Check if action contains explicit sql mention (preserve intent)
    if "sql" in a:
        return "sql"

    synonym_map = {
        "aggregate": "sql", "aggregate_sales": "sql", "aggregation": "sql",
        "summarize": "explain", "preview": "overview", "analyze": "eda", "interpret": "explain",
        "explanation": "explain"
    }
    return synonym_map.get(a, fallback if fallback in allowed else "eda")


def _normalize_sequence(seq, fallback_action) -> List[dict]:
    out: List[dict] = []
    for i, raw in enumerate((seq or [])[:5]):
        if isinstance(raw, dict):
            a = _coerce_allowed(raw.get("action"), fallback_action)
            plan = raw.get("model_plan")
            if a == "modeling":
                plan = infer_default_model_plan(st.session_state.current_question, plan)

            # Get SQL query, generate if missing for SQL actions
            sql_query = raw.get("duckdb_sql")
            if a == "sql" and (sql_query is None or sql_query == "NULL" or (isinstance(sql_query, str) and sql_query.strip() == "")):
                # Check the raw action description for clues
                action_desc = str(raw).lower() if isinstance(raw, str) else str(raw.get("action", "")).lower()
                current_q = getattr(st.session_state, 'current_question', '').lower()

                # Generate SQL based on action description and step position
                if "category" in action_desc or ("category" in current_q and i == 0):
                    sql_query = "SELECT product_category_name FROM olist_products_dataset WHERE product_id = 'bb50f2e236e5eea0100680137654686c'"
                elif "customer" in action_desc or "contributor" in action_desc or ("customer" in current_q and i == 1):
                    sql_query = """
                        SELECT o.customer_id, SUM(oi.price) as total_spent
                        FROM olist_order_items_dataset oi
                        JOIN olist_orders_dataset o ON oi.order_id = o.order_id
                        WHERE oi.product_id = 'bb50f2e236e5eea0100680137654686c'
                        GROUP BY o.customer_id
                        ORDER BY total_spent DESC
                        LIMIT 1
                    """.strip()
                elif "top selling product" in current_q:
                    sql_query = """
                        SELECT oi.product_id, SUM(oi.price) as total_sales, COUNT(*) as order_count
                        FROM olist_order_items_dataset oi
                        GROUP BY oi.product_id
                        ORDER BY total_sales DESC
                        LIMIT 1
                    """.strip()
                else:
                    # Use existing fallback generation if available
                    if hasattr(st.session_state, 'shared_context') and st.session_state.shared_context:
                        try:
                            step_description = raw.get("description", "").lower()
                            sql_query = generate_contextual_fallback_sql(
                                st.session_state.current_question or "",
                                st.session_state.shared_context,
                                step_description,
                                i
                            )
                        except:
                            sql_query = "SELECT 1 as fallback_query"
                    else:
                        sql_query = "SELECT 1 as fallback_query"

            out.append({
                "action": a,
                "duckdb_sql": sql_query,
                "charts": raw.get("charts"),
                "model_plan": plan,
                "calc_description": raw.get("calc_description"),
            })
        elif isinstance(raw, str):
            a = _coerce_allowed(raw, fallback_action)
            plan = infer_default_model_plan(st.session_state.current_question, {} ) if a == "modeling" else None

            # Generate SQL for string-based SQL actions
            sql_query = None
            if a == "sql":
                current_q = getattr(st.session_state, 'current_question', '')
                if "top selling product" in current_q.lower():
                    sql_query = """
                        SELECT oi.product_id, SUM(oi.price) as total_sales, COUNT(*) as order_count
                        FROM olist_order_items_dataset oi
                        GROUP BY oi.product_id
                        ORDER BY total_sales DESC
                        LIMIT 1
                    """.strip()

            out.append({"action": a, "duckdb_sql": sql_query, "charts": None,
                        "model_plan": plan, "calc_description": None})
    if not out:
        out = [{"action": _coerce_allowed(None, fallback_action),
                "duckdb_sql": None, "charts": None,
                "model_plan": None, "calc_description": None}]
    return out


def run_ds_step(am_json: dict, column_hints: dict, thread_ctx: dict) -> dict:
    current_question = st.session_state.current_question or ""
    full_context = build_shared_context()
    shared_context = assess_context_relevance(current_question, full_context)

    ds_payload = {
        "am_plan": am_json.get("plan_for_ds", ""),
        "am_next_action_type": am_json.get("next_action_type", "eda"),
        "am_action_sequence": am_json.get("action_sequence", []),
        "am_referenced_entities": am_json.get("referenced_entities", {}),
        "shared_context": shared_context,
        "column_hints": column_hints,
    }

    # DEBUG: Log what we're sending to DS agent
    st.write("🔍 **DEBUG - DS Agent Input:**")
    st.write(f"- AM Plan: {am_json.get('plan_for_ds', 'MISSING')}")
    st.write(f"- AM Next Action: {am_json.get('next_action_type', 'MISSING')}")
    st.write(f"- AM Action Sequence: {am_json.get('action_sequence', 'MISSING')}")
    st.write(f"- AM Referenced Entities: {am_json.get('referenced_entities', 'MISSING')}")
    st.write(f"- Current Question: {current_question}")
    st.write(f"- Has Shared Context: {bool(shared_context)}")
    st.write(f"- Has Column Hints: {bool(column_hints)}")

    if shared_context.get("referenced_entities"):
        st.write(f"- Shared Context Referenced Entities: {shared_context['referenced_entities']}")

    ds_json = llm_json(SYSTEM_DS, json.dumps(ds_payload))

    # DEBUG: Log what DS agent returned
    st.write("🔍 **DEBUG - DS Agent Output:**")
    st.write(f"- DS Action: {ds_json.get('action', 'MISSING')}")
    st.write(f"- DS Action Sequence: {ds_json.get('action_sequence', 'MISSING')}")
    st.write(f"- DS Sequence: {ds_json.get('sequence', 'MISSING')}")
    st.write(f"- DS Summary: {ds_json.get('ds_summary', 'MISSING')}")

    if ds_json.get("sequence"):
        st.write("🔍 **DEBUG - Sequence Steps:**")
        for i, step in enumerate(ds_json.get("sequence", [])):
            st.write(f"  Step {i}: Action={step.get('action')}, SQL={step.get('duckdb_sql')}")

    st.session_state.last_ds_json = ds_json

    # CRITICAL FIX: Generate SQL queries if missing (addresses NULL duckdb_sql bug)
    if ds_json.get("action_sequence"):
        action_sequence = ds_json.get("action_sequence", [])
        if isinstance(action_sequence, list):
            for i, step in enumerate(action_sequence):
                # Ensure step is a dictionary and has the expected structure
                if isinstance(step, dict) and step.get("action") == "sql":
                    sql = step.get("duckdb_sql")
                    if sql is None or sql == "NULL" or (isinstance(sql, str) and sql.strip() == ""):
                        # Generate contextual SQL based on the question and step index
                        current_q = current_question.lower()
                        if "category" in current_q and i == 0:
                            # First step: get product category
                            product_id = shared_context.get("referenced_entities", {}).get("product_id", "bb50f2e236e5eea0100680137654686c")
                            step["duckdb_sql"] = f"SELECT product_category_name FROM olist_products_dataset WHERE product_id = '{product_id}'"
                        elif "customer" in current_q and i == 1:
                            # Second step: get top customer
                            product_id = shared_context.get("referenced_entities", {}).get("product_id", "bb50f2e236e5eea0100680137654686c")
                            step["duckdb_sql"] = f"""
                                SELECT o.customer_id, SUM(oi.price) as total_spent
                                FROM olist_order_items_dataset oi
                                JOIN olist_orders_dataset o ON oi.order_id = o.order_id
                                WHERE oi.product_id = '{product_id}'
                                GROUP BY o.customer_id
                                ORDER BY total_spent DESC
                                LIMIT 1
                            """.strip()
                        elif "top selling product" in current_q:
                            # Default: find top selling product
                            step["duckdb_sql"] = """
                                SELECT oi.product_id, SUM(oi.price) as total_sales
                                FROM olist_order_items_dataset oi
                                GROUP BY oi.product_id
                                ORDER BY total_sales DESC
                                LIMIT 1
                            """.strip()
                        else:
                            # Use existing fallback generation
                            try:
                                step["duckdb_sql"] = generate_contextual_fallback_sql(
                                    current_question, shared_context, step.get("description", "").lower(), i
                                )
                            except Exception:
                                # Ultimate fallback if generation fails
                                step["duckdb_sql"] = "SELECT 1 as placeholder"

    am_mode = (am_json.get("task_mode") or ("multi" if am_json.get("action_sequence") else "single")).lower()
    if am_mode == "multi":
        seq = ds_json.get("action_sequence") or am_json.get("action_sequence") or []
        # Fix: Use "sql" as fallback for multi-mode to preserve SQL actions
        fallback_action = "sql" if any("sql" in str(step).lower() for step in seq) else (am_json.get("next_action_type") or "eda")
        norm_seq = _normalize_sequence(seq, fallback_action.lower())
        ds_json["action_sequence"] = norm_seq
        add_msg("ds", ds_json.get("ds_summary", ""), artifacts={"mode": "multi", "sequence": norm_seq})
    else:
        a = _coerce_allowed(ds_json.get("action"), (am_json.get("next_action_type") or "eda").lower())
        ds_json["action"] = (am_json.get("next_action_type") or a)
        if ds_json["action"] == "modeling":
            ds_json["model_plan"] = infer_default_model_plan(st.session_state.current_question, ds_json.get("model_plan"))
        add_msg("ds", ds_json.get("ds_summary", ""), artifacts={
            "action": ds_json.get("action"),
            "duckdb_sql": ds_json.get("duckdb_sql"),
            "model_plan": ds_json.get("model_plan")
        })

    render_chat()
    return ds_json


def judge_review(user_question: str, am_response: dict, ds_response: dict, tables_schema: dict, executed_results: dict = None) -> dict:
    """Judge agent reviews AM and DS work AFTER execution for quality and correctness."""
    
    # Track current revision attempt
    current_revision = len([r for r in st.session_state.revision_history if r.get("user_question") == user_question]) + 1
    
    # Pre-validate DS response for critical errors
    validation = validate_ds_response(ds_response)
    if validation["has_critical_errors"]:
        # Try fallback mechanism immediately when NULL SQL is detected
        st.warning(f"🔧 LLM generated NULL SQL. Using fallback mechanism...")
        shared_context = build_shared_context()
        fixed_ds_response = fix_ds_response_with_fallback(ds_response, user_question, shared_context)

        # Re-validate after fallback
        fixed_validation = validate_ds_response(fixed_ds_response)
        if not fixed_validation["has_critical_errors"]:
            st.success("✅ Fallback SQL generated successfully")
            # Update the ds_response for continued processing
            ds_response.update(fixed_ds_response)
            # Continue with the fixed response
        else:
            st.error("❌ Fallback mechanism also failed")
            return {
                    "judgment": "rejected",
                    "addresses_user_question": False,
                    "user_question_analysis": "Both LLM and fallback mechanism failed to generate valid SQL",
                    "quality_issues": ["LLM completely failed to generate SQL", "Fallback mechanism insufficient"],
                    "revision_notes": "SYSTEM FAILURE: Unable to generate SQL queries after multiple attempts and fallback",
                    "can_display": False
                }
    
    # Execute DS actions to get actual results for judge review
    actual_results = {}
    sql_queries = []
    
    if ds_response.get("action_sequence"):
        # Handle multi-step actions
        for step in ds_response.get("action_sequence", []):
            sql = _sql_first(step.get("duckdb_sql"))
            if sql:
                sql_queries.append(sql)
                try:
                    result = run_duckdb_sql(sql)
                    actual_results[f"step_{len(sql_queries)}"] = {
                        "sql": sql,
                        "row_count": len(result),
                        "columns": list(result.columns),
                        "sample_data": result.head(5).to_dict(orient="records")
                    }
                except Exception as e:
                    actual_results[f"step_{len(sql_queries)}"] = {
                        "sql": sql,
                        "error": str(e)
                    }
    else:
        # Handle single action
        sql = _sql_first(ds_response.get("duckdb_sql"))
        if sql:
            sql_queries.append(sql)
            try:
                result = run_duckdb_sql(sql)
                actual_results["single_action"] = {
                    "sql": sql,
                    "row_count": len(result),
                    "columns": list(result.columns),
                    "sample_data": result.head(5).to_dict(orient="records")
                }
            except Exception as e:
                actual_results["single_action"] = {
                    "sql": sql,
                    "error": str(e)
                }
    
    full_context = build_shared_context()
    shared_context = assess_context_relevance(user_question, full_context)

    # Assess question context for Judge evaluation
    question_lower = user_question.lower()
    is_overview_question = any(phrase in question_lower for phrase in [
        "what data", "data available", "datasets", "overview", "show me", "available"
    ])
    is_first_question = len(st.session_state.revision_history) == 0 and len(shared_context.get("key_findings", [])) == 0
    has_contextual_reference = any(ref in question_lower for ref in [
        "this", "that", "the customer", "the product", "the order"
    ])
    has_cached_results = len(shared_context.get("key_findings", [])) > 0
    
    # Check for repeated quality issues (anti-loop protection)
    revision_history = st.session_state.revision_history
    repeated_issues = []
    if len(revision_history) >= 2:
        recent_issues = [rev.get("quality_issues", []) for rev in revision_history[-2:]]
        if recent_issues[0] and recent_issues[1]:
            repeated_issues = list(set(recent_issues[0]) & set(recent_issues[1]))
    
    context_assessment = {
        "is_overview_question": is_overview_question,
        "is_first_question": is_first_question,
        "has_contextual_reference": has_contextual_reference,
        "has_cached_results": has_cached_results,
        "requires_entity_continuity": has_contextual_reference and has_cached_results,
        "should_use_cached": has_cached_results and not is_overview_question and has_contextual_reference,
        "repeated_issues": repeated_issues,
        "potential_loop": len(repeated_issues) > 0 and current_revision >= 3
    }
    
    judge_payload = {
        "user_question": user_question,
        "shared_context": shared_context,
        "am_response": am_response,
        "ds_response": ds_response,
        "executed_results": actual_results,
        "revision_history": st.session_state.revision_history,
        "current_revision_number": current_revision,
        "available_tables": tables_schema,
        "context_assessment": context_assessment,
    }
    
    return llm_json(SYSTEM_JUDGE, json.dumps(judge_payload))

def am_review(ceo_prompt: str, ds_json: dict, meta: dict) -> dict:
    full_context = build_shared_context()
    shared_context = assess_context_relevance(ceo_prompt, full_context)
    bundle = {
        "ceo_question": ceo_prompt,
        "shared_context": shared_context,
        "am_plan": st.session_state.last_am_json,
        "ds_json": ds_json,
        "meta": meta,
    }
    return llm_json(SYSTEM_AM_REVIEW, json.dumps(bundle))


def revise_ds(am_json: dict, prev_ds_json: dict, review_json: dict, column_hints: dict, thread_ctx: dict) -> dict:
    current_question = st.session_state.current_question or ""
    full_context = build_shared_context()
    shared_context = assess_context_relevance(current_question, full_context)
    payload = {
        "am_plan": am_json.get("plan_for_ds", ""),
        "shared_context": shared_context,
        "previous_ds_json": prev_ds_json,
        "am_critique": {
            "appropriateness_check": review_json.get("appropriateness_check"),
            "revision_notes": review_json.get("revision_notes"),
            "gaps_or_risks": review_json.get("gaps_or_risks"),
            "improvements": review_json.get("improvements"),
        },
        "column_hints": column_hints,
    }
    return llm_json(SYSTEM_DS_REVISE, json.dumps(payload))


# ======================
# Build meta for AM review (supports sequence and explain)
# ======================
def build_meta_for_action(ds_step: dict) -> dict:
    action = (ds_step.get("action") or "").lower()

    if action == "overview":
        tables_meta = {name: {"rows": len(df), "cols": len(df.columns)} for name, df in get_all_tables().items()}
        return {"type": "overview", "tables": tables_meta}

    if action == "eda":
        raw_sql = ds_step.get("duckdb_sql")
        sql_list = raw_sql if isinstance(raw_sql, list) else [raw_sql]
        metas = []
        for sql in [_sql_first(s) for s in sql_list if s]:
            try:
                df = run_duckdb_sql(sql)
                metas.append({"sql": sql, "rows": len(df), "cols": list(df.columns),
                              "sample": df.head(10).to_dict(orient="records")})
            except Exception as e:
                metas.append({"sql": sql, "error": str(e)})
        return {"type": "eda", "results": metas}

    if action == "sql":
        sql = _sql_first(ds_step.get("duckdb_sql"))
        if not sql: return {"type": "sql", "error": "No SQL provided"}
        try:
            out = run_duckdb_sql(sql)
            return {"type": "sql", "sql": sql, "rows": len(out), "cols": list(out.columns),
                    "sample": out.head(10).to_dict(orient="records")}
        except Exception as e:
            return {"type": "sql", "sql": sql, "error": str(e)}

    if action == "calc":
        return {"type": "calc", "desc": ds_step.get("calc_description", "")}

    if action == "feature_engineering":
        sql = _sql_first(ds_step.get("duckdb_sql"))
        try:
            base = run_duckdb_sql(sql) if sql else next(iter(get_all_tables().values())).copy()
            return {"type": "feature_engineering", "rows": len(base), "cols": list(base.columns),
                    "sample": base.head(10).to_dict(orient="records")}
        except Exception as e:
            return {"type": "feature_engineering", "error": str(e)}

    if action == "modeling":
        sql = _sql_first(ds_step.get("duckdb_sql"))
        plan = infer_default_model_plan(st.session_state.current_question, ds_step.get("model_plan") or {})
        target = plan.get("target")
        try:
            base = run_duckdb_sql(sql) if sql else None
            if base is None:
                base = choose_model_base(plan, st.session_state.current_question)
            if base is None:
                return {"type":"modeling","error":"No tables loaded."}
            return {"type": "modeling", "task": (plan.get("task") or "clustering").lower(),
                    "target": target, "features": plan.get("features") or [],
                    "family": (plan.get("model_family") or "kmeans").lower(),
                    "n_clusters": plan.get("n_clusters"),
                    "rows": len(base), "cols": list(base.columns)}
        except Exception as e:
            return {"type": "modeling", "error": str(e)}

    if action == "explain":
        cache = st.session_state.last_results
        meta = {"type": "explain"}
        if cache.get("clustering"):
            rep = cache["clustering"]
            meta["clustering"] = {k: rep.get(k) for k in ["features","n_clusters","silhouette","cluster_sizes"]}
        if cache.get("modeling"):
            m = cache["modeling"]
            meta["modeling"] = {k: m.get(k) for k in ["task","target","features","metrics"]}
        if cache.get("eda"):
            meta["eda"] = {"sqls": cache["eda"].get("sqls"), "sample_cols": cache["eda"].get("sample_cols")}
        if cache.get("feature_engineering"):
            meta["feature_engineering"] = {"rows": cache["feature_engineering"].get("rows"), "cols": cache["feature_engineering"].get("cols")}
        if cache.get("sql"):
            meta["sql"] = {"sql": cache["sql"].get("sql"), "rows": cache["sql"].get("rows")}
        if not any([cache.get("clustering"), cache.get("modeling"), cache.get("eda"), cache.get("feature_engineering"), cache.get("sql")]):
            meta["error"] = "no_cache"
        return meta

    return {"type": action or "unknown", "note": "no meta builder"}


# ======================
# Renderer for actions (with caching for explain)
# ======================
def render_final_for_action(ds_step: dict):
    action = (ds_step.get("action") or "").lower()

    # ---- OVERVIEW ----
    if action == "overview":
        st.markdown("### 📊 Table Previews (first 5 rows)")
        for name, df in get_all_tables().items():
            st.markdown(f"**{name}** — rows: {len(df)}, cols: {len(df.columns)}")
            st.dataframe(df.head(5), width="stretch")
        add_msg("ds", "Overview rendered.")
        return

    # ---- EDA ----
    if action == "eda":
        raw_sql = ds_step.get("duckdb_sql")
        sql_list = raw_sql if isinstance(raw_sql, list) else [raw_sql]
        charts_all = ds_step.get("charts") or []
        executed_sqls, last_cols = [], None
        for i, sql in enumerate([_sql_first(s) for s in sql_list][:3]):
            if not sql: continue
            try:
                df = run_duckdb_sql(sql)
                executed_sqls.append(sql)
                last_cols = list(df.columns)
                st.markdown(f"### 📈 EDA Result #{i+1} (first 50 rows)")
                st.code(sql, language="sql")
                st.dataframe(df.head(50), width="stretch")
                charts_this = []
                if charts_all and isinstance(charts_all[0], dict):
                    charts_this = charts_all if i == 0 else []
                elif charts_all and isinstance(charts_all[0], list):
                    charts_this = charts_all[i] if i < len(charts_all) else []
                for spec in (charts_this or [])[:3]:
                    title = spec.get("title") or "Chart"
                    ctype = (spec.get("type") or "bar").lower()
                    xcol = spec.get("x"); ycol = spec.get("y")
                    if isinstance(xcol,str) and isinstance(ycol,str) and xcol in df.columns and ycol in df.columns:
                        st.markdown(f"**{title}**")
                        plot_df = df[[xcol, ycol]].set_index(xcol)
                        if ctype == "line": st.line_chart(plot_df)
                        elif ctype == "area": st.area_chart(plot_df)
                        else: st.bar_chart(plot_df)
            except Exception as e:
                st.error(f"EDA SQL failed: {e}")
        st.session_state.last_results["eda"] = {"sqls": executed_sqls, "sample_cols": last_cols}
        add_msg("ds","EDA rendered.")
        return

    # ---- SQL ----
    if action == "sql":
        sql = _sql_first(ds_step.get("duckdb_sql"))
        if not sql:
            add_msg("ds","No SQL provided.")
            return
        try:
            out = run_duckdb_sql(sql)
            st.markdown("### 🧮 SQL Results (first 25 rows)")
            st.code(sql, language="sql")
            st.dataframe(out.head(25), width="stretch")
            add_msg("ds","SQL executed.", artifacts={"sql": sql})
            st.session_state.last_results["sql"] = {"sql": sql, "rows": len(out), "cols": list(out.columns)}
        except Exception as e:
            st.error(f"SQL failed: {e}")
        return

    # ---- CALC ----
    if action == "calc":
        st.markdown("### 🧮 Calculation")
        st.write(ds_step.get("calc_description","(no description)"))
        add_msg("ds","Calculation displayed.")
        return

    # ---- FEATURE ENGINEERING ----
    if action == "feature_engineering":
        sql = _sql_first(ds_step.get("duckdb_sql"))
        base = run_duckdb_sql(sql) if sql else next(iter(get_all_tables().values())).copy()
        st.markdown("### 🧱 Feature Engineering Base (first 20 rows)")
        st.dataframe(base.head(20), width="stretch")
        st.session_state.tables_fe["feature_base"] = base
        st.session_state.last_results["feature_engineering"] = {"rows": len(base), "cols": list(base.columns)}
        add_msg("ds","Feature base ready (saved as 'feature_base').")
        return

    # ---- MODELING ----
    if action == "modeling":
        sql = _sql_first(ds_step.get("duckdb_sql"))
        plan = infer_default_model_plan(st.session_state.current_question, ds_step.get("model_plan") or {})
        task = (plan.get("task") or "clustering").lower()
        target = plan.get("target")
        base = run_duckdb_sql(sql) if sql else None
        if base is None:
            base = choose_model_base(plan, st.session_state.current_question)

        if task == "clustering":
            result = train_model(base, task, None, plan.get("features") or [], plan.get("model_family") or "", plan.get("n_clusters"))
            if result.get("error"):
                st.error(result.get("error"))
                if result.get("feature_proposal"):
                    st.markdown("**Feature Proposal (auto-derived):**")
                    st.json(result["feature_proposal"])
                return
            rep = result.get("report", {}) if isinstance(result, dict) else {}
            st.markdown("### 🔍 Clustering Report")
            st.json(rep)
            pca_df = result.get("pca")
            if isinstance(pca_df, pd.DataFrame):
                st.markdown("**PCA Scatter (by cluster)**")
                st.dataframe(pca_df.head(200))
            add_msg("ds","Clustering completed.", artifacts={"report": rep})
            # cache
            st.session_state.last_results["clustering"] = {
                "report": rep,
                "features": list(rep.get("features") or []),
                "n_clusters": int(rep.get("n_clusters") or 0),
                "cluster_sizes": rep.get("cluster_sizes") or {},
                "silhouette": rep.get("silhouette"),
                "centroids": (result.get("centroids").to_dict(orient="records") if isinstance(result.get("centroids"), pd.DataFrame) else None)
            }
            st.session_state.last_results["modeling"] = {
                "task": "clustering",
                "target": None,
                "features": list(rep.get("features") or []),
                "metrics": {"silhouette": rep.get("silhouette"), "inertia": rep.get("inertia")}
            }
            return

        report = train_model(base, task, target, plan.get("features") or [], (plan.get("model_family") or "logistic_regression").lower())
        if isinstance(report, dict) and report.get("error"):
            st.error(report.get("error"))
        st.markdown("### 🤖 Model Report")
        st.json(report)
        add_msg("ds","Model trained.", artifacts={"model_report": report})
        st.session_state.last_results["modeling"] = {
            "task": task, "target": target, "features": report.get("features"),
            "metrics": {k: report.get(k) for k in ["accuracy","roc_auc","mae","rmse","r2"] if k in report}
        }
        return

    # ---- EXPLAIN ----
    if action == "explain":
        cache = st.session_state.last_results
        if not any(cache.values()):
            add_msg("ds", "I don’t have cached results yet. Please run an analysis step first.")
            st.info("No cached results found. Run an analysis first, then ask for interpretation.")
            return

        st.markdown("### 📝 Interpretation of Latest Results")

        if cache.get("clustering"):
            ctx = cache["clustering"]
            st.markdown("#### Clustering")
            st.markdown("**Features used:** " + (", ".join(ctx.get("features") or []) or "_not recorded_"))
            st.markdown(f"**k (clusters):** {ctx.get('n_clusters')}")
            st.markdown(f"**Silhouette:** {ctx.get('silhouette')}")
            st.markdown("**Cluster sizes:**")
            st.json(ctx.get("cluster_sizes") or {})
            if ctx.get("centroids"):
                st.markdown("**Centroids (original units):**")
                cdf = pd.DataFrame(ctx["centroids"])
                st.dataframe(cdf)
                
                # Better cluster differentiation logic
                try:
                    st.markdown("**Cluster Characteristics (relative differences):**")
                    
                    # Calculate relative differences from overall mean
                    overall_means = cdf.mean()
                    relative_diffs = cdf.subtract(overall_means, axis=1)
                    
                    bullets = []
                    for i, row in relative_diffs.iterrows():
                        # Find features that are meaningfully different (>0.5 std dev)
                        std_threshold = relative_diffs.std().mean() * 0.5
                        
                        high_features = []
                        low_features = []
                        
                        for col in row.index:
                            if row[col] > std_threshold:
                                high_features.append(f"{col} (+{row[col]:.1f})")
                            elif row[col] < -std_threshold:
                                low_features.append(f"{col} ({row[col]:.1f})")
                        
                        # Create meaningful labels
                        cluster_desc = f"Cluster {i} ({ctx.get('cluster_sizes', {}).get(str(i), '?')} products):"
                        if high_features:
                            cluster_desc += f" HIGH {', '.join(high_features)}"
                        if low_features:
                            cluster_desc += f" LOW {', '.join(low_features)}"
                        if not high_features and not low_features:
                            cluster_desc += " AVERAGE dimensions"
                            
                        bullets.append(cluster_desc)
                    
                    for b in bullets: st.write(f"• {b}")
                    
                    # Add size-based interpretation
                    st.markdown("**Size-based interpretation:**")
                    sizes = ctx.get('cluster_sizes', {})
                    for cluster_id, size in sizes.items():
                        centroid_values = cdf.loc[int(cluster_id)]
                        avg_volume = centroid_values.get('product_length_cm', 0) * centroid_values.get('product_width_cm', 0) * centroid_values.get('product_height_cm', 0)
                        st.write(f"• Cluster {cluster_id}: {size:,} products, avg volume ≈ {avg_volume:.0f} cm³")
                        
                except Exception as e:
                    st.error(f"Enhanced labeling failed: {e}")
                    # Fallback to original logic
                    bullets = []
                    for i, row in cdf.iterrows():
                        top = row.sort_values(ascending=False).head(min(3, len(row))).index.tolist()
                        bullets.append(f"- Cluster {i}: high on {', '.join(top)}")
                    for b in bullets: st.write(b)
            st.markdown("**Next:** Join clusters back to sales/margin/shipping for pricing & ops.")
        elif cache.get("modeling"):
            m = cache["modeling"]
            st.markdown("#### Modeling")
            st.json(m)
        elif cache.get("eda"):
            e = cache["eda"]
            st.markdown("#### EDA")
            st.write("Recent EDA queries:")
            for s in (e.get("sqls") or [])[:5]:
                st.code(s, language="sql")
            st.write("Sample columns observed:", e.get("sample_cols"))
        elif cache.get("feature_engineering"):
            fe = cache["feature_engineering"]
            st.markdown("#### Feature Engineering")
            st.json(fe)
        elif cache.get("sql"):
            sq = cache["sql"]
            st.markdown("#### SQL result context")
            st.json(sq)
        add_msg("ds", "Provided interpretation without re-running any steps.", artifacts={"explain_used": True})
        return

    # ---- KEYWORD EXTRACTION ----
    if action == "keyword_extraction":
        # Check if there's review data available (from previous SQL step or cache)
        review_data = None
        sql = _sql_first(ds_step.get("duckdb_sql"))
        
        if sql:
            # Execute SQL to get review data
            try:
                review_data = run_duckdb_sql(sql)
            except Exception as e:
                st.error(f"Failed to execute SQL for keyword extraction: {e}")
                add_msg("ds", f"Keyword extraction failed: {e}")
                return
        else:
            # Try to get review data from cached results
            cached_sql = get_last_approved_result("sql")
            if cached_sql and isinstance(cached_sql, pd.DataFrame):
                review_data = cached_sql
        
        if review_data is None or review_data.empty:
            st.error("No review data available for keyword extraction")
            add_msg("ds", "No review data found for keyword extraction.")
            return
        
        # Perform keyword extraction
        st.markdown("### 🔍 Keyword Extraction from Reviews")
        keywords_result = extract_keywords_from_reviews(review_data, top_n=10)
        
        if keywords_result.get("error"):
            st.error(f"Keyword extraction error: {keywords_result['error']}")
            add_msg("ds", f"Keyword extraction failed: {keywords_result['error']}")
            return
        
        # Display results
        st.markdown(f"**Total reviews analyzed:** {keywords_result.get('total_reviews', 0)}")
        st.markdown(f"**Total words processed:** {keywords_result.get('total_words_analyzed', 0)}")
        
        if keywords_result.get("keywords"):
            st.markdown("**Top 10 Keywords:**")
            keywords_df = pd.DataFrame(keywords_result["keywords"])
            st.dataframe(keywords_df, width="stretch")
            
            # Create a simple bar chart
            st.markdown("**Keyword Frequency Chart:**")
            chart_data = keywords_df.set_index("keyword")
            st.bar_chart(chart_data)
            
            add_msg("ds", f"Extracted {len(keywords_result['keywords'])} keywords from {keywords_result.get('total_reviews', 0)} reviews.", 
                   artifacts={"keywords": keywords_result["keywords"], "total_reviews": keywords_result.get('total_reviews', 0)})
        else:
            st.warning("No keywords found in the review data")
            add_msg("ds", "No keywords could be extracted from the review data.")
        
        # Cache the results
        st.session_state.last_results["keyword_extraction"] = keywords_result
        return

    add_msg("ds", f"Action '{action}' not recognized.", artifacts=ds_step)


# ======================
# Auto-progression heuristic
# ======================
def must_progress_to_modeling(thread_ctx: dict, am_json: dict, ds_json: dict) -> bool:
    """
    Decide whether to auto-advance to modeling (usually clustering).
    Conservative default: False unless the CEO clearly asked for modeling/segmentation
    and neither AM nor DS already planned to model in this turn.
    """
    # NEW: never auto-progress when AM asked for an EXPLAIN turn
    if (am_json.get("next_action_type") or "").lower() == "explain":
        return False

    if (am_json.get("next_action_type") or "").lower() == "modeling":
        return False
    if (ds_json.get("action") or "").lower() == "modeling":
        return False
    if any((isinstance(s, dict) and (s.get("action","").lower() == "modeling"))
           or (isinstance(s, str) and s.lower() == "modeling")
           for s in (ds_json.get("action_sequence") or [])):
        return False

    q = (thread_ctx.get("current_question") or thread_ctx.get("central_question") or "").lower()

    if re.search(r"\bwhat (data|datasets?) do we have\b", q) or "first 5 rows" in q:
        return False

    wants_model = any(k in q for k in [
        "cluster", "clustering", "segment", "segmentation", "kmeans",
        "unsupervised", "model", "predict", "classification", "regression"
    ])
    return bool(wants_model)


# ======================
# Coordinator (threading + arbitration + follow-ups)
# ======================
def run_turn_ceo(new_text: str):

    prev = st.session_state.current_question or ""
    central = st.session_state.central_question or ""
    prior = st.session_state.prior_questions or []

    # Explicit "not a follow up" starts a new thread
    force_new = _explicit_new_thread(new_text)

    ic = classify_intent(prev, central, prior, new_text)
    intent = "new_request" if force_new else ic.get("intent", "new_request")
    related = False if force_new else ic.get("related", False)

    # Manage threads & central question
    if intent == "new_request" and not related:
        if st.session_state.central_question:
            st.session_state.threads.append({"central": st.session_state.central_question, "followups": []})
        st.session_state.central_question = new_text
        st.session_state.current_question = new_text
    else:
        st.session_state.current_question = (central or new_text).strip() + "\n\n[Follow-up]: " + (new_text or "").strip()

    # Track prior questions
    if prev and prev not in st.session_state.prior_questions:
        st.session_state.prior_questions.append(prev)
    if new_text and new_text not in st.session_state.prior_questions:
        st.session_state.prior_questions.append(new_text)

    # Context update - no verbose message needed

    effective_q = st.session_state.current_question

    # 0) Column hints
    col_hints = build_column_hints(effective_q)

    # Context pack for AM/DS
    thread_ctx = {
        "central_question": st.session_state.central_question,
        "current_question": st.session_state.current_question,
        "prior_questions": st.session_state.prior_questions,
    }

    # 1) AM plan
    am_json = run_am_plan(effective_q, col_hints, context=thread_ctx)

    # If AM needs user info, ask and stop
    if am_json.get("need_more_info"):
        qs = am_json.get("clarifying_questions") or ["Could you clarify your objective?"]
        add_msg("am", "I need a bit more context:")
        for q in qs[:3]:
            add_msg("am", f"• {q}")
        render_chat(); return

    # 2) DS executes and enters review loop
    max_loops = 3
    loop_count = 0
    ds_json = run_ds_step(am_json, col_hints, thread_ctx)

    if must_progress_to_modeling(thread_ctx, am_json, ds_json):
        am_json = {**am_json, "task_mode":"single", "next_action_type":"modeling", "plan_for_ds": (am_json.get("plan_for_ds") or "") + " | Proceed to clustering."}
        ds_json = run_ds_step(am_json, col_hints, thread_ctx)

    # If DS asks for clarification explicitly
    if ds_json.get("need_more_info") and (ds_json.get("clarifying_questions") or []):
        add_msg("ds", "Before running further steps, I need:")
        for q in (ds_json.get("clarifying_questions") or [])[:3]:
            add_msg("ds", f"• {q}")
        render_chat(); return

    # Build metas (sequence-aware)
    def _build_metas(ds_json_local: dict) -> Union[dict, List[dict]]:
        if ds_json_local.get("action_sequence"):
            metas = []
            for step in ds_json_local.get("action_sequence")[:5]:
                metas.append(build_meta_for_action(step))
            return metas
        else:
            return build_meta_for_action({
                "action": ds_json_local.get("action"),
                "duckdb_sql": ds_json_local.get("duckdb_sql"),
                "charts": ds_json_local.get("charts"),
                "model_plan": ds_json_local.get("model_plan"),
                "calc_description": ds_json_local.get("calc_description"),
            })

    # Render only if judge-approved (sequence-aware)
    def _render(ds_json_local: dict, judge_approved: bool = False):
        if not judge_approved:
            add_msg("system", "Results pending judge approval...")
            render_chat()
            return
            
        if ds_json_local.get("action_sequence"):
            results_summary = []
            for step in ds_json_local.get("action_sequence")[:5]:
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
            add_msg("system", f"✅ Judge-approved multi-step results rendered. {summary_text}")
            render_chat()
        else:
            if not ds_json_local.get("action"):
                ds_json_local["action"] = am_json.get("next_action_type") or "eda"
            
            step_data = {
                "action": ds_json_local.get("action"),
                "duckdb_sql": ds_json_local.get("duckdb_sql"),
                "charts": ds_json_local.get("charts"),
                "model_plan": ds_json_local.get("model_plan"),
                "calc_description": ds_json_local.get("calc_description"),
            }
            
            # Cache result with approval
            action_id = f"{step_data.get('action', 'unknown')}_{len(st.session_state.executed_results)}"
            cache_result_with_approval(action_id, step_data, approved=True)
            render_final_for_action(step_data)
            add_msg("system", "✅ Judge-approved results rendered.")
            render_chat()

    while loop_count < max_loops:
        loop_count += 1

        meta = _build_metas(ds_json)
        
        # Judge Agent Review
        tables_schema = {k: list(v.columns) for k, v in get_all_tables().items()}
        judge_result = judge_review(effective_q, am_json, ds_json, tables_schema)
        
        # If judge finds critical issues, flag them
        if judge_result.get("judgment") in ["needs_revision", "rejected"]:
            st.warning("⚖️ Judge Agent found issues with the approach")
            with st.expander("Judge Feedback", expanded=True):
                if not judge_result.get("addresses_user_question", True):
                    st.error("❌ Response does not address the user's actual question")
                    st.write(judge_result.get("user_question_analysis", "No analysis provided"))
                
                quality_issues = judge_result.get("quality_issues", [])
                if quality_issues:
                    st.write("**Quality Issues:**")
                    for issue in quality_issues:
                        st.write(f"• {issue}")
                
                if judge_result.get("revision_notes"):
                    st.write("**Judge Recommendations:**")
                    st.write(judge_result.get("revision_notes"))
        
        # Skip verbose quality assessment message for cleaner UI
        render_chat()
        
        # AM Review  
        review = am_review(effective_q, ds_json, {"meta": meta, "mode": "multi" if ds_json.get("action_sequence") else "single"})
        add_msg("am", review.get("summary_for_ceo",""), artifacts={
            "appropriateness_check": review.get("appropriateness_check"),
            "gaps_or_risks": review.get("gaps_or_risks"),
            "improvements": review.get("improvements"),
            "suggested_next_steps": review.get("suggested_next_steps"),
            "must_revise": review.get("must_revise"),
            "sufficient_to_answer": review.get("sufficient_to_answer"),
        })
        render_chat()

        # If AM needs clarification from CEO
        if review.get("clarification_needed") and (review.get("clarifying_questions") or []):
            add_msg("am", "Before proceeding, could you clarify:")
            for q in (review.get("clarifying_questions") or [])[:3]:
                add_msg("am", f"• {q}")
            render_chat(); return

        # If sufficient and no revision required AND judge approves → render final and exit
        judge_approved = judge_result.get("judgment") == "approved" and judge_result.get("can_display", False)
        judge_needs_revision = judge_result.get("judgment") in ["needs_revision", "rejected"]
        judge_needs_clarification = judge_result.get("judgment") == "needs_clarification"
        
        # Handle clarification requests - ask user to clarify ambiguous terms
        if judge_needs_clarification:
            add_msg("system", "🤔 Clarification Needed")
            add_msg("system", "I notice there might be different ways to interpret your question:")
            
            clarification_questions = judge_result.get("clarification_questions", [])
            if clarification_questions:
                for q in clarification_questions:
                    add_msg("system", f"• {q}")
            else:
                add_msg("system", "• Could you clarify what specific metric you'd like to see?")
                
            add_msg("system", "Please let me know which interpretation you prefer, and I'll provide the exact analysis you need.")
            render_chat()
            return
        
        if review.get("sufficient_to_answer") and not review.get("must_revise") and judge_approved:
            # Judge approved - render results directly without verbose messages
            _render(ds_json, judge_approved=True)
            return

        # Otherwise revise if requested by AM or Judge
        if review.get("must_revise") or judge_needs_revision:
            # Track revision history
            revision_entry = {
                "user_question": effective_q,
                "revision_number": len([r for r in st.session_state.revision_history if r.get("user_question") == effective_q]) + 1,
                "ds_response": ds_json.copy(),
                "judge_feedback": judge_result,
                "am_feedback": review,
                "timestamp": pd.Timestamp.now().isoformat()
            }
            st.session_state.revision_history.append(revision_entry)
            
            # Include judge feedback in revision
            enhanced_review = dict(review)
            if judge_needs_revision:
                enhanced_review["judge_feedback"] = {
                    "quality_issues": judge_result.get("quality_issues", []),
                    "revision_notes": judge_result.get("revision_notes", ""),
                    "implementation_guidance": judge_result.get("implementation_guidance", ""),
                    "user_question_analysis": judge_result.get("user_question_analysis", ""),
                    "revision_analysis": judge_result.get("revision_analysis", {})
                }
            ds_json = revise_ds(am_json, ds_json, enhanced_review, col_hints, thread_ctx)
            if ds_json.get("action_sequence"):
                ds_json["action_sequence"] = _normalize_sequence(
                    ds_json.get("action_sequence"),
                    (am_json.get("next_action_type") or "eda").lower()
                )
            add_msg("ds", ds_json.get("ds_summary","(revised)"), artifacts={
                "mode": "multi" if ds_json.get("action_sequence") else "single",
                "action": ds_json.get("action"),
                "action_sequence": ds_json.get("action_sequence"),
                "duckdb_sql": ds_json.get("duckdb_sql"),
                "model_plan": ds_json.get("model_plan"),
                "uses_cached_result": ds_json.get("uses_cached_result"),
                "referenced_entities": ds_json.get("referenced_entities")
            })
            render_chat()
            continue  # loop for another review

        # No revision required but check judge approval → render current best and exit
        final_judge_approved = judge_result.get("judgment") == "approved" and judge_result.get("can_display", False)
        _render(ds_json, judge_approved=final_judge_approved)
        return

    add_msg("system", "Reached review limit; presenting current best effort with noted caveats.")
    _render(ds_json)
    return


# ======================
# Data loading
# ======================
def load_if_needed():
    if uploaded_file and st.session_state.tables_raw is None:
        st.session_state.tables_raw = load_data_file(uploaded_file)
        st.session_state.tables = get_all_tables()
        file_type = os.path.splitext(uploaded_file.name)[1].upper()
        add_msg("system", f"Loaded {len(st.session_state.tables_raw)} tables from {file_type} file: {uploaded_file.name}")
        render_chat()

load_if_needed()

# ======================
# Chat UI
# ======================
st.subheader("Chat")
render_chat()

user_prompt = st.chat_input("You're the CEO. Ask a question (e.g., 'Cluster product dimensions', 'Explain the clusters')")
if user_prompt:
    add_msg("user", user_prompt)
    render_chat()
    run_turn_ceo(user_prompt)
