@echo off
REM Traffic Guardian AI - starts the live prediction API
REM Usage: run.bat
cd /d "%~dp0"
echo Starting Traffic Guardian AI on http://127.0.0.1:8000 ...
echo Open http://127.0.0.1:8000 in your browser once it says "Application startup complete".
python -m uvicorn app.main:app --reload --port 8000
