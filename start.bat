@echo off
setlocal

echo ==============================================
echo   MultiHazard Predictor Startup Script
echo ==============================================

set "INFERENCE_MODE=planetary_computer"
set "ALLOW_SYNTHETIC_INFERENCE=false"
set "APP_ENV=development"
set "MULTIHAZARD_API_URL=http://127.0.0.1:8000/api/v1"

set "SCRIPT_DIR=%~dp0"
set "PYTHON_EXE=%SCRIPT_DIR%.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=%SCRIPT_DIR%.venv-2\Scripts\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=%SCRIPT_DIR%venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"

if not exist "%SCRIPT_DIR%api\app.py" (
    echo Missing project files in %SCRIPT_DIR%
    pause
    exit /b 1
)

if not exist "%SCRIPT_DIR%web\package.json" (
    echo Missing React dashboard in %SCRIPT_DIR%web
    pause
    exit /b 1
)

where node >nul 2>nul
if errorlevel 1 (
    echo Node.js is required to run the React dashboard.
    pause
    exit /b 1
)

if not exist "%SCRIPT_DIR%web\node_modules" (
    echo Frontend dependencies are missing. Run npm install in the web directory first.
    pause
    exit /b 1
)

echo [1/2] Starting FastAPI Backend on Port 8000...
start "FastAPI Backend" cmd /c ""%PYTHON_EXE%" -m uvicorn api.app:app --host 0.0.0.0 --port 8000"

echo Waiting for FastAPI Backend to become ready...
set "API_READY="
for /l %%I in (1,1,30) do (
    powershell -NoProfile -Command "try { Invoke-WebRequest -Uri 'http://127.0.0.1:8000/api/v1/health' -UseBasicParsing -TimeoutSec 2 ^| Out-Null; exit 0 } catch { exit 1 }" >nul 2>nul
    if not errorlevel 1 (
        set "API_READY=1"
        goto :api_ready
    )
    timeout /t 2 /nobreak >nul
)

:api_ready
if not defined API_READY (
    echo FastAPI Backend did not become ready. Check the FastAPI Backend window before using the dashboard.
    pause
    exit /b 1
)

echo [2/2] Starting React Dashboard on Port 8503...
start "React Dashboard" cmd /k "cd /d ""%SCRIPT_DIR%web"" && npm run dev"

echo.
echo Both services have been launched in separate windows!
echo API Docs: http://localhost:8000/docs
echo Dashboard: http://localhost:8503
echo.
pause
endlocal
