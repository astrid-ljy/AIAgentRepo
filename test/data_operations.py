"""
Complete data operations and file loading - Complete version from original app.py
"""
import os
import io
import json
import zipfile
import hashlib
from typing import Dict, Any, Optional
import pandas as pd
import duckdb
import streamlit as st
from config import api_key, model, _OPENAI_AVAILABLE

if _OPENAI_AVAILABLE:
    from openai import OpenAI

def ensure_openai():
    """Ensure OpenAI client is available."""
    if not _OPENAI_AVAILABLE:
        raise RuntimeError("OpenAI SDK missing.")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY")
    return OpenAI(api_key=api_key)

def llm_json(system_prompt: str, user_payload: str) -> dict:
    """Robust JSON helper with structured-mode first, then free-form parse fallback."""
    client = ensure_openai()

    sys_msg = system_prompt.strip() + "\\n\\nReturn ONLY a single JSON object. This line contains the word json."
    user_msg = (user_payload or "").strip() + "\\n\\nPlease respond with JSON only (a single object)."

    # Preferred structured call
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_msg}
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=2000
        )
        content = resp.choices[0].message.content
        return eval(content) if content else {}
    except Exception:
        pass

    # Fallback to free-form
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.1,
            max_tokens=2000
        )
        content = resp.choices[0].message.content or ""

        # Extract JSON from response
        if "{" in content:
            start = content.find("{")
            end = content.rfind("}") + 1
            json_str = content[start:end]
            return eval(json_str)
        return {}
    except Exception:
        return {}

def load_zip_tables(file) -> Dict[str, pd.DataFrame]:
    """Load CSV files from a ZIP archive."""
    tables = {}
    with zipfile.ZipFile(file) as z:
        for name in z.namelist():
            if not name.lower().endswith(".csv"):
                continue
            with z.open(name) as f:
                df = pd.read_csv(io.BytesIO(f.read()))
            key = os.path.splitext(os.path.basename(name))[0]
            i, base = 1, key
            while key in tables:
                key = f"{base}_{i}"
                i += 1
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
            df = pd.read_csv(file, sep='\\t')
            return {base_name: df}

        elif file_ext == '.txt':
            # Try to detect delimiter
            file.seek(0)
            sample = file.read(1024)
            if isinstance(sample, bytes):
                sample = sample.decode('utf-8')
            file.seek(0)

            # Common delimiters in corporate files
            if '\\t' in sample:
                df = pd.read_csv(file, sep='\\t')
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
    """Get all loaded tables from session state."""
    if hasattr(st.session_state, 'tables_raw') and st.session_state.tables_raw:
        return st.session_state.tables_raw
    return st.session_state.tables or {}

def run_duckdb_sql(sql: str, use_cache: bool = True) -> pd.DataFrame:
    """Execute SQL with optional caching to avoid re-running identical queries."""
    if use_cache:
        sql_hash = hashlib.md5(sql.encode()).hexdigest()
        cache_key = f"sql_cache_{sql_hash}"

        if hasattr(st.session_state, cache_key):
            return getattr(st.session_state, cache_key)

    # Load tables into DuckDB for this query
    tables = get_all_tables()
    if not tables:
        raise Exception("No tables loaded. Please upload data first.")

    try:
        # Register tables with DuckDB
        for table_name, df in tables.items():
            # Clean table name for SQL compatibility
            clean_name = table_name.replace('-', '_').replace(' ', '_')
            duckdb.register(clean_name, df)

        # Execute query
        result = duckdb.query(sql).to_df()

        if use_cache:
            setattr(st.session_state, cache_key, result)

        return result
    except Exception as e:
        raise Exception(f"SQL execution failed: {e}")

def cache_result_with_approval(action_id: str, result_data: Any, approved: bool = False) -> None:
    """Cache a result with approval status."""
    if "executed_results" not in st.session_state:
        st.session_state.executed_results = {}

    st.session_state.executed_results[action_id] = {
        "result": result_data,
        "approved": approved,
        "timestamp": pd.Timestamp.now()
    }

def get_cached_result(action_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve a cached result by action ID."""
    if "executed_results" not in st.session_state:
        return None
    return st.session_state.executed_results.get(action_id)

def get_last_approved_result(action_type: str) -> Optional[Any]:
    """Get the most recent approved result of a specific action type."""
    if "executed_results" not in st.session_state:
        return None

    for action_id, cached_data in reversed(st.session_state.executed_results.items()):
        if cached_data.get("approved", False) and action_id.startswith(action_type):
            return cached_data.get("result")
    return None

def build_column_hints(question: str) -> dict:
    """Build column hints mapping business terms to database columns."""
    question_lower = question.lower()

    # Base mappings
    hints = {
        "revenue": ["price", "payment_value", "total_sales"],
        "sales": ["price", "payment_value"],
        "customer": ["customer_id", "customer_unique_id"],
        "product": ["product_id"],
        "seller": ["seller_id"],
        "order": ["order_id"],
        "category": ["product_category_name"],
        "location": ["customer_city", "customer_state", "seller_city", "seller_state"],
        "geography": ["customer_state", "seller_state"],
        "time": ["order_purchase_timestamp", "order_delivered_customer_date"],
        "rating": ["review_score"],
        "review": ["review_comment_message", "review_comment_title"],
        "payment": ["payment_type", "payment_value"],
        "shipping": ["freight_value", "shipping_limit_date"],
        "quality": ["review_score", "review_comment_message"]
    }

    # Add contextual hints based on question
    if any(word in question_lower for word in ["top", "best", "highest"]):
        hints["ranking"] = ["price", "review_score", "total_sales"]

    if any(word in question_lower for word in ["frequency", "count", "number"]):
        hints["counting"] = ["customer_id", "product_id", "order_id"]

    if any(word in question_lower for word in ["geographic", "region", "location"]):
        hints["geographic"] = ["customer_state", "seller_state", "customer_city", "seller_city"]

    return hints

def _sql_first(maybe_sql):
    """Get first SQL query from various formats."""
    if isinstance(maybe_sql, list):
        return maybe_sql[0] if maybe_sql else None
    return maybe_sql

def setup_file_upload():
    """Setup file upload widget in sidebar."""
    with st.sidebar:
        st.header("⚙️ Data")
        uploaded_file = st.file_uploader(
            "Upload Data File",
            type=["zip", "csv", "xlsx", "xls", "json", "jsonl", "tsv", "txt", "parquet"],
            help="Supports: ZIP (containing CSVs), CSV, Excel (.xlsx/.xls), JSON, JSONL, TSV, TXT, and Parquet files"
        )

        if uploaded_file:
            return uploaded_file

    return None

def load_if_needed():
    """Load data file if uploaded and not already loaded."""
    uploaded_file = setup_file_upload()

    if uploaded_file and st.session_state.get('tables_raw') is None:
        with st.spinner("Loading data..."):
            st.session_state.tables_raw = load_data_file(uploaded_file)
            st.session_state.tables = get_all_tables()

            if st.session_state.tables_raw:
                file_type = os.path.splitext(uploaded_file.name)[1].upper()
                from ui import add_msg
                add_msg("system", f"Loaded {len(st.session_state.tables_raw)} tables from {file_type} file: {uploaded_file.name}")

                # Display table summary
                st.sidebar.write("**Loaded Tables:**")
                for table_name, df in st.session_state.tables_raw.items():
                    st.sidebar.write(f"- {table_name}: {len(df)} rows, {len(df.columns)} columns")
            else:
                st.sidebar.error("Failed to load data from file")
