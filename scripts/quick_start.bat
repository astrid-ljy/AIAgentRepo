@echo off
echo ============================================================
echo ChatDev Agent System - Quick Start
echo ============================================================
echo.

echo [1/3] Installing dependencies...
pip install sqlglot pydantic opentelemetry-api
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Failed to install dependencies
    echo Please install manually: pip install sqlglot pydantic opentelemetry-api
    pause
    exit /b 1
)
echo.

echo [2/3] Verifying installation...
python verify_installation.py
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Verification failed
    echo Please check the errors above and fix them
    pause
    exit /b 1
)
echo.

echo [3/3] Starting Streamlit app...
echo.
echo ============================================================
echo App is ready! Look for the checkbox in the sidebar:
echo     [x] Use ChatDev-style agents
echo ============================================================
echo.
streamlit run app.py
