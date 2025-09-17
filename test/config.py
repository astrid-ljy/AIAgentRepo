"""
Configuration and shared imports for the AI Agent system.
"""
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

# OpenAI integration
try:
    from openai import OpenAI
    api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    _OPENAI_AVAILABLE = bool(api_key)
except ImportError:
    _OPENAI_AVAILABLE = False
    api_key = None
    model = None

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

# === SESSION STATE INITIALIZATION ===
def init_session_state():
    """Initialize all session state variables."""
    if "ds_cache" not in st.session_state:
        st.session_state.ds_cache = {"profiles": {}, "featprops": {}, "clusters": {}, "colmaps": {}}

    if "executed_results" not in st.session_state:
        st.session_state.executed_results = {}
    if "last_results" not in st.session_state:
        st.session_state.last_results = {}
    if "tables_fe" not in st.session_state:
        st.session_state.tables_fe = {}
    if "tables" not in st.session_state:
        st.session_state.tables = None
    if "chat" not in st.session_state:
        st.session_state.chat = []
    if "last_rendered_idx" not in st.session_state:
        st.session_state.last_rendered_idx = 0
    if "last_am_json" not in st.session_state:
        st.session_state.last_am_json = {}
    if "last_ds_json" not in st.session_state:
        st.session_state.last_ds_json = {}
    if "last_user_prompt" not in st.session_state:
        st.session_state.last_user_prompt = ""
    if "current_question" not in st.session_state:
        st.session_state.current_question = ""
    if "threads" not in st.session_state:
        st.session_state.threads = []
    if "central_question" not in st.session_state:
        st.session_state.central_question = ""
    if "prior_questions" not in st.session_state:
        st.session_state.prior_questions = []
