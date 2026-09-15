@echo off
REM MultiHazard-VLM Phase 1 Setup Script for Windows
REM This script helps set up PostgreSQL, Redis, and the full stack

echo.
echo ============================================================
echo   MultiHazard-VLM Phase 1 Setup Script
echo ============================================================
echo.

REM Check if Python virtual environment exists
if not exist ".venv\Scripts\activate.bat" (
    echo ❌ Virtual environment not found!
    echo Creating .venv...
    python -m venv .venv
    call .venv\Scripts\activate.bat
) else (
    call .venv\Scripts\activate.bat
)

echo.
echo ============================================================
echo   Step 1: Installing Dependencies
echo ============================================================
echo.
pip install -r requirements.txt --upgrade

echo.
echo ============================================================
echo   Step 2: Check PostgreSQL and Redis
echo ============================================================
echo.

REM Check if PostgreSQL is accessible
echo Checking PostgreSQL connection...
psql -U postgres -c "SELECT version();" >nul 2>&1
if errorlevel 1 (
    echo ⚠️  PostgreSQL might not be running or installed
    echo    Please ensure PostgreSQL is installed and running on localhost:5432
    echo    Download from: https://www.postgresql.org/download/windows/
) else (
    echo ✅ PostgreSQL is accessible
)

REM Check if Redis is accessible
echo Checking Redis connection...
redis-cli ping >nul 2>&1
if errorlevel 1 (
    echo ⚠️  Redis might not be running or installed
    echo    Please ensure Redis is installed and running on localhost:6379
    echo    Download from: https://github.com/microsoftarchive/redis/releases
) else (
    echo ✅ Redis is accessible
)

echo.
echo ============================================================
echo   Step 3: Initialize Database
echo ============================================================
echo.
python init_database.py

echo.
echo ============================================================
echo   Step 4: Launch Menu
echo ============================================================
echo.
echo Choose what to do:
echo   1) Start API server (Uvicorn)
echo   2) Start API server (Gunicorn)
echo   3) Start Streamlit Dashboard
echo   4) Start both API and Dashboard
echo   5) View API Health
echo   6) View Database Status
echo   7) View Redis Status
echo   8) Exit
echo.
set /p choice="Enter your choice (1-8): "

if "%choice%"=="1" (
    echo Starting FastAPI (Uvicorn)...
    python -m uvicorn api.app:app --reload
) else if "%choice%"=="2" (
    echo Starting FastAPI (Gunicorn)...
    gunicorn -c gunicorn_config.py api.app:app
) else if "%choice%"=="3" (
    echo Starting Streamlit Dashboard...
    python -m streamlit run dashboard/app.py --server.port 8503
) else if "%choice%"=="4" (
    echo Starting both API and Dashboard...
    echo Make sure to run these in separate terminals:
    echo   Terminal 1: python -m uvicorn api.app:app --reload
    echo   Terminal 2: python -m streamlit run dashboard/app.py --server.port 8503
    pause
) else if "%choice%"=="5" (
    echo Checking API Health...
    curl http://localhost:8000/api/v1/health
    pause
) else if "%choice%"=="6" (
    echo Checking Database...
    psql -U postgres -d hazard_db -c "SELECT COUNT(*) as predictions_count FROM predictions;"
    pause
) else if "%choice%"=="7" (
    echo Checking Redis...
    redis-cli DBSIZE
    pause
) else if "%choice%"=="8" (
    echo Exiting...
    exit /b 0
) else (
    echo Invalid choice!
    pause
)

pause
