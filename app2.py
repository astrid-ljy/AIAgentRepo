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
- `shared_context`: Contains cached results, recent SQL queries, key findings, and extracted entities
- `column_hints`: Business term to column mappings
- Available table schemas

**Context awareness:**
- FIRST check `shared_context.key_findings` for relevant entities (top_selling_product_id, etc.)
- Review `shared_context.recent_sql_results` to understand what data was already queried
- For follow-up questions about "this product" or "this customer", use specific IDs from key findings
- Avoid re-querying data that's already in cached results unless additional detail is needed

**Action classification:** Decide the **granularity** first:
- task_mode: "single" or "multi".
- If task_mode="single" → choose exactly one next_action_type for DS from:
  `overview`, `sql`, `eda`, `calc`, `feature_engineering`, `modeling`, `explain`.
- If task_mode="multi" → propose a short `action_sequence` (2–5 steps) using ONLY those allowed actions.

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
- ALWAYS check `shared_context.key_findings` for relevant entities (top_selling_product_id, etc.)
- Review `shared_context.recent_sql_results` to avoid duplicate queries
- For follow-up questions, use specific IDs from key findings (e.g., "this product" = shared_context.key_findings.top_selling_product_id)
- Reference cached data when available instead of re-querying

**CRITICAL: Always provide actual SQL queries, never NULL or empty strings.**

**Smart Query Generation:**
- For follow-ups about "this product/customer", use the specific ID from shared_context.key_findings
- Build on previous results - if top product is already known, query product details directly
- Example follow-up SQL: "SELECT * FROM products WHERE product_id = '[specific_id_from_context]'"
- Only run new aggregation queries if the specific data isn't already cached

**Execution modes:**
- If AM provided `am_action_sequence`, return matching `action_sequence`. Otherwise, return single `action`
- Allowed actions: overview, sql, eda, calc, feature_engineering, modeling, explain, keyword_extraction

**CRITICAL SQL Requirements:**
- NEVER return NULL, empty, or missing duckdb_sql values - this is a critical error
- ALWAYS provide complete, executable SQL queries in duckdb_sql field  
- Use proper table/column names from schema
- For entity-specific queries, use exact IDs from shared_context.key_findings
- Example valid SQL: "SELECT p.product_category_name FROM products p WHERE p.product_id = 'bb50f2e236e5eea0100680137654686c'"
- Example multi-table: "SELECT c.customer_id, SUM(oi.price * oi.quantity) as total FROM orders o JOIN order_items oi ON o.order_id = oi.order_id JOIN customers c ON o.customer_id = c.customer_id WHERE oi.product_id = 'bb50f2e236e5eea0100680137654686c' GROUP BY c.customer_id ORDER BY total DESC LIMIT 1"

**Special rules:**
- Data Inventory: Overview with first 5 rows for each table
- Explain: Read cached results and interpret without recomputing
- Entity continuity: "this product" MUST use product_id from shared_context.key_findings
- Keyword Extraction: For review analysis, use multi-step approach: 1) SQL to get reviews, 2) keyword_extraction action to process text

Return JSON fields:
- ds_summary
- need_more_info
- clarifying_questions  
- action OR action_sequence
- duckdb_sql (NEVER null - always provide actual SQL)
- charts
- model_plan: {task, target, features, model_family, n_clusters}
- calc_description
- assumptions
- uses_cached_result: true/false
- referenced_entities: {product_id: "...", customer_id: "..."}
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

**Revision Tracking & Progressive Feedback:**
- **Revision 1**: General feedback about missing functionality
- **Revision 2**: Specific implementation steps and examples
- **Revision 3**: Detailed code-level instructions with exact SQL/methods
- **Revision 4+**: Consider rejecting if no progress made

**Context Validation:**
- Verify DS used correct entity IDs from shared_context.key_findings
- Check if cached results were properly leveraged to avoid redundant queries
- Validate entity continuity (e.g., "this product" should use specific product_id from context)
- Ensure new queries build logically on previous results

**Special Case - Keyword Extraction:**
- For keyword requests, DS must use multi-step approach: SQL + keyword_extraction action
- Validate that text processing actually occurred, not just data retrieval
- Check if results include actual keywords with frequencies

Return ONLY valid JSON:
{
  "judgment": "approved/needs_revision/rejected",
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
    "entity_continuity_correct": true/false,
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
    zip_file = st.file_uploader("Upload ZIP of CSVs", type=["zip"])
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
        return json.loads(resp.choices[0].message.content or "{}")
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
    
    # Extract key findings and entities
    key_findings = {}
    if "sql" in cached_results:
        # Try to extract top products, customers, etc. from recent SQL results
        for query_key, query_data in recent_sql_results.items():
            sample_data = query_data.get("sample_data", [])
            if sample_data and isinstance(sample_data, list) and len(sample_data) > 0:
                first_row = sample_data[0]
                if "product_id" in first_row:
                    key_findings["top_selling_product_id"] = first_row["product_id"]
                if "customer_id" in first_row:
                    key_findings["top_customer_id"] = first_row["customer_id"]
                if any(col.endswith("_sales") or col.endswith("_revenue") for col in first_row.keys()):
                    key_findings["sales_metrics_available"] = True
    
    return {
        "cached_results": cached_results,
        "recent_sql_results": recent_sql_results,
        "key_findings": key_findings,
        "conversation_context": {
            "central_question": st.session_state.central_question,
            "current_question": st.session_state.current_question,
            "prior_questions": st.session_state.prior_questions
        },
        "available_tables": {k: list(v.columns) for k, v in get_all_tables().items()},
        "context_timestamp": pd.Timestamp.now().isoformat()
    }


def add_msg(role, content, artifacts=None):
    st.session_state.chat.append({"role": role, "content": content, "artifacts": artifacts or {}})


def render_chat(incremental: bool = True):
    msgs = st.session_state.chat
    start = st.session_state.last_rendered_idx if incremental else 0
    for m in msgs[start:]:
        with st.chat_message(m["role"]):
            st.write(m["content"])
            if m.get("artifacts"):
                with st.expander("Artifacts", expanded=False):
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
    shared_context = build_shared_context()
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
    synonym_map = {
        "aggregate": "sql", "aggregate_sales": "sql", "aggregation": "sql",
        "summarize": "explain", "preview": "overview", "analyze": "eda", "interpret": "explain",
        "explanation": "explain"
    }
    return synonym_map.get(a, fallback if fallback in allowed else "eda")


def _normalize_sequence(seq, fallback_action) -> List[dict]:
    out: List[dict] = []
    for raw in (seq or [])[:5]:
        if isinstance(raw, dict):
            a = _coerce_allowed(raw.get("action"), fallback_action)
            plan = raw.get("model_plan")
            if a == "modeling":
                plan = infer_default_model_plan(st.session_state.current_question, plan)
            out.append({
                "action": a,
                "duckdb_sql": raw.get("duckdb_sql"),
                "charts": raw.get("charts"),
                "model_plan": plan,
                "calc_description": raw.get("calc_description"),
            })
        elif isinstance(raw, str):
            a = _coerce_allowed(raw, fallback_action)
            plan = infer_default_model_plan(st.session_state.current_question, {} ) if a == "modeling" else None
            out.append({"action": a, "duckdb_sql": None, "charts": None,
                        "model_plan": plan, "calc_description": None})
    if not out:
        out = [{"action": _coerce_allowed(None, fallback_action),
                "duckdb_sql": None, "charts": None,
                "model_plan": None, "calc_description": None}]
    return out


def run_ds_step(am_json: dict, column_hints: dict, thread_ctx: dict) -> dict:
    shared_context = build_shared_context()
    
    ds_payload = {
        "am_plan": am_json.get("plan_for_ds", ""),
        "am_next_action_type": am_json.get("next_action_type", "eda"),
        "am_action_sequence": am_json.get("action_sequence", []),
        "shared_context": shared_context,
        "column_hints": column_hints,
    }
    ds_json = llm_json(SYSTEM_DS, json.dumps(ds_payload))
    st.session_state.last_ds_json = ds_json

    am_mode = (am_json.get("task_mode") or ("multi" if am_json.get("action_sequence") else "single")).lower()
    if am_mode == "multi":
        seq = ds_json.get("action_sequence") or am_json.get("action_sequence") or []
        norm_seq = _normalize_sequence(seq, (am_json.get("next_action_type") or "eda").lower())
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
        return {
            "judgment": "needs_revision",
            "addresses_user_question": False,
            "user_question_analysis": "DS response has critical errors that prevent execution",
            "revision_analysis": {
                "revision_number": current_revision,
                "progress_made": False,
                "same_issues_repeated": True,
                "escalation_needed": True
            },
            "quality_issues": validation["issues"],
            "revision_notes": f"CRITICAL VALIDATION ERRORS (Revision {current_revision}): " + "; ".join(validation["issues"]) + 
                            f"\n\nYou MUST provide actual SQL queries, not NULL values. Example for product info: SELECT p.product_category_name FROM products p WHERE p.product_id = '{am_response.get('referenced_entities', {}).get('product_id', 'MISSING_ID')}'",
            "implementation_guidance": "Replace ALL NULL duckdb_sql values with actual executable SQL queries. Use table names and column names from the schema. Use the specific product_id from shared context.",
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
    
    shared_context = build_shared_context()
    
    judge_payload = {
        "user_question": user_question,
        "shared_context": shared_context,
        "am_response": am_response,
        "ds_response": ds_response,
        "executed_results": actual_results,
        "revision_history": st.session_state.revision_history,
        "current_revision_number": current_revision,
        "available_tables": tables_schema,
    }
    
    return llm_json(SYSTEM_JUDGE, json.dumps(judge_payload))

def am_review(ceo_prompt: str, ds_json: dict, meta: dict) -> dict:
    shared_context = build_shared_context()
    bundle = {
        "ceo_question": ceo_prompt,
        "shared_context": shared_context,
        "am_plan": st.session_state.last_am_json,
        "ds_json": ds_json,
        "meta": meta,
    }
    return llm_json(SYSTEM_AM_REVIEW, json.dumps(bundle))


def revise_ds(am_json: dict, prev_ds_json: dict, review_json: dict, column_hints: dict, thread_ctx: dict) -> dict:
    shared_context = build_shared_context()
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

    add_msg("system", "Context updated: central question & history considered.", artifacts={
        "intent": intent, "related": related,
        "central_question": st.session_state.central_question,
        "current_question": st.session_state.current_question,
    })
    render_chat()

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
            for step in ds_json_local.get("action_sequence")[:5]:
                # Cache each step result with approval
                action_id = f"{step.get('action', 'unknown')}_{len(st.session_state.executed_results)}"
                cache_result_with_approval(action_id, step, approved=True)
                render_final_for_action(step)
            add_msg("system", "✅ Judge-approved multi-step results rendered.")
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
        
        add_msg("judge", f"Quality Assessment: {judge_result.get('judgment', 'Unknown')}", 
                artifacts=judge_result)
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
        
        if review.get("sufficient_to_answer") and not review.get("must_revise") and judge_approved:
            # Hide judge feedback if approved to keep conversation clean
            if judge_approved:
                # Remove or minimize judge message display
                st.session_state.chat = [msg for msg in st.session_state.chat if msg.get("role") != "judge" or msg.get("content") != f"Quality Assessment: {judge_result.get('judgment', 'Unknown')}"]
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
            add_msg("ds", ds_json.get("ds_summary","(revised)"),
                    artifacts={"mode": "multi" if ds_json.get("action_sequence") else "single"})
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
    if zip_file and st.session_state.tables_raw is None:
        st.session_state.tables_raw = load_zip_tables(zip_file)
        st.session_state.tables = get_all_tables()
        add_msg("system", f"Loaded {len(st.session_state.tables_raw)} raw tables.")
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
