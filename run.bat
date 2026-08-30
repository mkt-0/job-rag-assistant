@echo off
setlocal
cd /d "%~dp0"

:: ---- locate python ----
where python >nul 2>nul
if errorlevel 1 (
  where py >nul 2>nul
  if errorlevel 1 ( echo [error] Python not found. Install Python 3.10+ and retry. & pause & exit /b 1 )
  set PY=py
) else ( set PY=python )

:: ---- create local venv + install deps (once) ----
if not exist .venv (
  echo [setup] Creating virtual env .venv ...
  %PY% -m venv .venv
)
call .venv\Scripts\activate.bat
echo [setup] Installing dependencies (first time only) ...
pip install -q -r requirements.txt

:: ---- start server in background, log to server.log ----
echo [info] Starting server ... (first launch builds the index, may take a minute)
start "" /b python app.py > server.log 2>&1

:: ---- wait for /health, then open browser ----
powershell -NoProfile -Command "$n=0; while($n -lt 120){ try{ (Invoke-WebRequest http://localhost:5000/health -UseBasicParsing -TimeoutSec 2).StatusCode; Start-Process 'http://localhost:5000'; Write-Host '[ok] Opened http://localhost:5000'; break }catch{ Start-Sleep 2; $n++ } }"

echo.
echo   Server running. Log: server.log
echo   Stop: close this window, or Ctrl+C.
echo.
pause
