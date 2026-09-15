@echo off
setlocal

echo ==============================================
echo   OmniHazard-VLM System Startup Script
echo ==============================================

set "SCRIPT_DIR=%~dp0"
set "PYTHON_EXE=%SCRIPT_DIR%.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=%SCRIPT_DIR%venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"

if not exist "%SCRIPT_DIR%api\app.py" (
    echo Missing project files in %SCRIPT_DIR%
    pause
    exit /b 1
)

if not exist "%SCRIPT_DIR%dashboard\app.py" (
    echo Missing dashboard in %SCRIPT_DIR%dashboard\n
    pause
    exit /b 1
)

echo [1/2] Starting FastAPI Backend on Port 8001...
start "FastAPI Backend" cmd /c ""%PYTHON_EXE%" -m uvicorn api.app:app --host 0.0.0.0 --port 8001"

echo [2/2] Starting Streamlit Dashboard on Port 8501...
start "Streamlit Dashboard" cmd /c ""%PYTHON_EXE%" -m streamlit run dashboard\app.py --server.port 8501"

echo.
echo Both services have been launched in separate windows!
echo API Docs: http://localhost:8001/docs
echo Dashboard: http://localhost:8501
echo.
pause
endlocal
