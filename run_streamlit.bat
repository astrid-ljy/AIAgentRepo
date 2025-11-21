@echo off
echo ========================================
echo   Tool-Based Agent - Streamlit UI
echo ========================================
echo.
echo Starting Streamlit app...
echo The app will open in your browser at http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo ========================================
echo.

cd /d %~dp0
py -m streamlit run streamlit_tool_agent.py

pause
