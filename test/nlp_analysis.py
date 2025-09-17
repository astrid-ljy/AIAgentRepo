"""
Natural Language Processing and Text Analysis - Complete version from original app.py
"""
import re
import pandas as pd
import streamlit as st
from typing import Dict, Any, List, Union
from config import _TEXTBLOB_AVAILABLE, _NLTK_AVAILABLE

if _TEXTBLOB_AVAILABLE:
    from textblob import TextBlob

if _NLTK_AVAILABLE:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize

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
            "terms": ["precio", "costo", "valor", "dinero", "pago", "barato", "caro", "descuento",
                     "oferta", "promoción", "económico", "costoso"],
            "found": []
        },
        "quality": {
            "terms": ["calidad", "excelente", "bueno", "malo", "defectuoso", "perfecto", "terrible",
                     "satisfecho", "decepcionado", "recomiendo", "satisfacción"],
            "found": []
        }
    }

    # Find keywords in each category
    for category, data in keywords.items():
        for term in data["terms"]:
            if term in comment_lower:
                data["found"].append(term)

    # Return only the found keywords
    return {category: data["found"] for category, data in keywords.items()}

def translate_spanish_to_english(text: str) -> Dict[str, Any]:
    """
    Translate Spanish text to English using available libraries.
    Fallback to keyword mapping if translation libraries unavailable.
    """
    if not text or not isinstance(text, str):
        return {
            "original": text,
            "translated": text,
            "confidence": 0.0,
            "method": "none",
            "error": "Invalid or empty text"
        }

    if not _TEXTBLOB_AVAILABLE:
        # Fallback: Simple keyword replacement for common Spanish e-commerce terms
        translation_map = {
            "producto": "product", "envío": "shipping", "entrega": "delivery",
            "precio": "price", "calidad": "quality", "servicio": "service",
            "bueno": "good", "malo": "bad", "excelente": "excellent",
            "terrible": "terrible", "rápido": "fast", "lento": "slow",
            "satisfecho": "satisfied", "decepcionado": "disappointed",
            "recomiendo": "recommend", "cliente": "customer",
            "vendedor": "seller", "paquete": "package",
            "tiempo": "time", "días": "days", "semana": "week"
        }

        translated = text.lower()
        for spanish, english in translation_map.items():
            translated = translated.replace(spanish, english)

        return {
            "original": text,
            "translated": translated,
            "confidence": 0.5,
            "method": "keyword_mapping"
        }

    try:
        # Use TextBlob for translation
        blob = TextBlob(text)
        translated = blob.translate(to='en')

        return {
            "original": text,
            "translated": str(translated),
            "confidence": 0.8,
            "method": "textblob"
        }
    except Exception as e:
        return {
            "original": text,
            "translated": text,
            "confidence": 0.0,
            "method": "error",
            "error": str(e)
        }

def extract_keywords_from_reviews(review_df: pd.DataFrame, text_column: str = "review_comment_message", top_n: int = 10) -> Dict[str, Any]:
    """
    Extract keywords from review text using multiple methods.
    """
    if review_df.empty or text_column not in review_df.columns:
        return {
            "top_keywords": [],
            "sentiment_summary": {},
            "keyword_categories": {},
            "error": "No valid review data found"
        }

    # Filter out null/empty comments
    valid_reviews = review_df[review_df[text_column].notna() & (review_df[text_column] != "")]

    if valid_reviews.empty:
        return {
            "top_keywords": [],
            "sentiment_summary": {},
            "keyword_categories": {},
            "error": "No valid review text found"
        }

    all_keywords = {
        "product": [], "shipment": [], "service": [], "price": [], "quality": []
    }
    sentiment_scores = []

    # Process each review
    for _, review in valid_reviews.iterrows():
        comment = str(review[text_column])

        # Extract keywords
        keywords = extract_ecommerce_keywords(comment)
        for category, terms in keywords.items():
            all_keywords[category].extend(terms)

        # Analyze sentiment
        sentiment = analyze_spanish_sentiment(comment)
        sentiment_scores.append(sentiment)

    # Count keyword frequencies
    keyword_counts = {}
    for category, terms in all_keywords.items():
        if terms:
            # Count frequency of each term
            term_counts = {}
            for term in terms:
                term_counts[term] = term_counts.get(term, 0) + 1
            keyword_counts[category] = sorted(term_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]

    # Aggregate sentiment
    if sentiment_scores:
        avg_polarity = sum(s["polarity"] for s in sentiment_scores) / len(sentiment_scores)
        sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
        for s in sentiment_scores:
            sentiment_counts[s["sentiment_label"]] += 1

        sentiment_summary = {
            "average_polarity": avg_polarity,
            "sentiment_distribution": sentiment_counts,
            "total_reviews": len(sentiment_scores)
        }
    else:
        sentiment_summary = {"error": "No sentiment data available"}

    # Get top keywords across all categories
    all_terms = []
    for category, terms in keyword_counts.items():
        all_terms.extend([(term, count, category) for term, count in terms])

    top_keywords = sorted(all_terms, key=lambda x: x[1], reverse=True)[:top_n]

    return {
        "top_keywords": [{"term": term, "count": count, "category": cat} for term, count, cat in top_keywords],
        "sentiment_summary": sentiment_summary,
        "keyword_categories": keyword_counts,
        "total_reviews_processed": len(valid_reviews)
    }

def analyze_spanish_comments(comments: Union[str, List[str]]) -> Dict[str, Any]:
    """
    Comprehensive analysis of Spanish comments including sentiment and keywords.
    """
    if isinstance(comments, str):
        comments = [comments]

    if not comments or not any(comments):
        return {
            "sentiment_analysis": {},
            "keyword_extraction": {},
            "language_detection": {},
            "summary": {},
            "error": "No valid comments provided"
        }

    results = {
        "sentiment_analysis": [],
        "keyword_extraction": [],
        "translations": [],
        "summary": {}
    }

    # Process each comment
    for i, comment in enumerate(comments):
        if not comment or not isinstance(comment, str):
            continue

        # Sentiment analysis
        sentiment = analyze_spanish_sentiment(comment)
        results["sentiment_analysis"].append({
            "comment_index": i,
            "sentiment": sentiment
        })

        # Keyword extraction
        keywords = extract_ecommerce_keywords(comment)
        results["keyword_extraction"].append({
            "comment_index": i,
            "keywords": keywords
        })

        # Translation
        translation = translate_spanish_to_english(comment)
        results["translations"].append({
            "comment_index": i,
            "translation": translation
        })

    # Create summary
    if results["sentiment_analysis"]:
        sentiments = [r["sentiment"] for r in results["sentiment_analysis"]]
        avg_polarity = sum(s["polarity"] for s in sentiments) / len(sentiments)

        sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
        for s in sentiments:
            sentiment_counts[s["sentiment_label"]] += 1

        # Aggregate keywords
        all_keywords = {"product": [], "shipment": [], "service": [], "price": [], "quality": []}
        for r in results["keyword_extraction"]:
            for category, terms in r["keywords"].items():
                all_keywords[category].extend(terms)

        results["summary"] = {
            "total_comments": len([c for c in comments if c]),
            "average_sentiment": avg_polarity,
            "sentiment_distribution": sentiment_counts,
            "most_common_keywords": {
                category: list(set(terms))[:5] for category, terms in all_keywords.items() if terms
            }
        }

    return results

def classify_intent(previous_question: str, central_question: str, prior_questions: List[str], new_text: str) -> dict:
    """Classify user intent for conversation management."""
    new_text_lower = new_text.lower().strip()

    # Check for explicit new thread indicators
    if any(phrase in new_text_lower for phrase in [
        "new question", "different topic", "change subject", "new analysis",
        "forget about", "start over", "new data", "different question"
    ]):
        return {"intent": "new_request", "related": False}

    # Check for continuation patterns
    continuation_patterns = [
        "this product", "this customer", "this order", "this seller",
        "the product", "the customer", "the order", "the seller",
        "it", "its", "that", "those", "these"
    ]

    if any(pattern in new_text_lower for pattern in continuation_patterns):
        return {"intent": "continuation", "related": True}

    # Check for clarification requests
    clarification_patterns = [
        "what does", "explain", "clarify", "what is", "how", "why",
        "tell me more", "can you elaborate", "what about"
    ]

    if any(pattern in new_text_lower for pattern in clarification_patterns):
        return {"intent": "clarification", "related": True}

    # Default to new request if no clear patterns
    return {"intent": "new_request", "related": False}