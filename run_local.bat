@echo off
REM RHEED Classifier - 로컬 실행 스크립트
cd /d "%~dp0"
echo.
echo ======================================
echo   RHEED Classifier - Local Server
echo ======================================
echo.
streamlit run app.py
pause
