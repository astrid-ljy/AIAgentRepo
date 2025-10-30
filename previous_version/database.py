"""
Database operations and data loading functionality.
"""
import hashlib
import zipfile
import io
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
    try:
        with zipfile.ZipFile(file, 'r') as z:
            for name in z.namelist():
                if name.endswith('.csv'):
                    with z.open(name) as f:
                        clean_name = name.replace('.csv', '').split('/')[-1]
                        tables[clean_name] = pd.read_csv(f)
        return tables
    except Exception as e:
        st.error(f"Error loading ZIP: {e}")
        return {}

def load_data_file(file) -> Dict[str, pd.DataFrame]:
    """Load data from various file formats."""
    try:
        if file.name.endswith('.zip'):
            return load_zip_tables(file)
        elif file.name.endswith('.csv'):
            table_name = file.name.replace('.csv', '')
            return {table_name: pd.read_csv(file)}
        elif file.name.endswith(('.xlsx', '.xls')):
            table_name = file.name.replace('.xlsx', '').replace('.xls', '')
            return {table_name: pd.read_excel(file)}
        else:
            st.error("Unsupported file format. Please upload CSV, Excel, or ZIP files.")
            return {}
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return {}

def get_all_tables() -> Dict[str, pd.DataFrame]:
    """Get all loaded tables from session state."""
    return st.session_state.tables or {}

def run_duckdb_sql(sql: str, use_cache: bool = True) -> pd.DataFrame:
    """Execute SQL with optional caching to avoid re-running identical queries."""
    if use_cache:
        sql_hash = hashlib.md5(sql.encode()).hexdigest()
        cache_key = f"sql_cache_{sql_hash}"

        if hasattr(st.session_state, cache_key):
            return getattr(st.session_state, cache_key)

    # Execute query
    try:
        result = duckdb.query(sql).to_df()

        if use_cache:
            setattr(st.session_state, cache_key, result)

        return result
    except Exception as e:
        raise Exception(f"SQL execution failed: {e}")

def cache_result_with_approval(action_id: str, result_data: Any, approved: bool = False) -> None:
    """Cache a result with approval status."""
    st.session_state.executed_results[action_id] = {
        "result": result_data,
        "approved": approved,
        "timestamp": pd.Timestamp.now()
    }

def get_cached_result(action_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve a cached result by action ID."""
    return st.session_state.executed_results.get(action_id)

def get_last_approved_result(action_type: str) -> Optional[Any]:
    """Get the most recent approved result of a specific action type."""
    for action_id, cached_data in reversed(st.session_state.executed_results.items()):
        if cached_data.get("approved", False) and action_id.startswith(action_type):
            return cached_data.get("result")
    return None