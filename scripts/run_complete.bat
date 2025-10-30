@echo off
echo Starting AI Agent - Complete Modular Version
echo ============================================
echo.
echo This is the COMPLETE modular version that preserves ALL original functionality:
echo - Advanced SQL generation and fallback systems
echo - Complete NLP and text analysis capabilities
echo - Full file upload support (CSV, Excel, ZIP, JSON, etc.)
echo - All entity detection and context management
echo - Complete agent coordination (AM -> DS -> Judge workflow)
echo - Fixed NULL duckdb_sql bug with multiple fallback layers
echo - Fixed import errors with safe streamlit imports
echo.
echo File Upload: Use the sidebar to upload your data files
echo Supported: ZIP, CSV, Excel, JSON, JSONL, TSV, TXT, Parquet
echo.
echo Testing imports first...
python test_imports.py
echo.
echo If imports test successful, starting Streamlit app...
streamlit run app_complete.py
echo.
pause