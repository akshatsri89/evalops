@echo off
echo Starting EvalOps...
echo.

REM Start the FastAPI server in a new window
start "EvalOps API" cmd /k "cd /d %~dp0 && venv\Scripts\activate && uvicorn app.main:app --reload"

REM Wait a couple seconds so the API has time to boot before the dashboard tries to read from it
timeout /t 3 /nobreak >nul

REM Start the Streamlit dashboard in a new window
start "EvalOps Dashboard" cmd /k "cd /d %~dp0 && venv\Scripts\activate && streamlit run dashboard.py"

echo Both servers are starting in separate windows.
echo API docs:   http://127.0.0.1:8000/docs
echo Dashboard:  http://localhost:8501
echo.
echo Close this window any time - it's not needed once the other two are running.
pause
