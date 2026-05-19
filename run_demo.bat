@echo off
title Early Fault Diagnosis Demo
cd /d "%~dp0"

echo ==========================================
echo   Early Fault Diagnosis Demo Launcher
echo ==========================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed or not added to PATH.
    echo Please install Python first, then run this file again.
    echo.
    pause
    exit /b
)

echo Installing required packages...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo.
echo Starting demo application...
echo If browser does not open automatically, copy the local address shown below.
echo.
python -m streamlit run fault_diagnosis_demo_app.py

pause
