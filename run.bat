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

:: ---- pre-flight: ensure .env exists and key is valid ----
echo [check] 校验配置 (.env / API key) ...
python -c "import sys, rag_core; sys.exit(0 if rag_core.is_key_ready() else 1)"
if errorlevel 1 (
  python -c "import rag_core; rag_core.preflight()"
  echo [error] 请按上面提示在 .env 中填入有效的 SILICONFLOW_API_KEY 后，重新运行 run.bat
  pause
  exit /b 1
)

:: ---- start server in background, log to server.log ----
echo [info] Starting server ... (首次启动会用 jobs.json 自动构建索引，可能耗时一两分钟)
start "" /b python app.py > server.log 2>&1

:: ---- wait for /health, then open browser ----
powershell -NoProfile -Command "$n=0; while($n -lt 120){ try{ (Invoke-WebRequest http://localhost:5000/health -UseBasicParsing -TimeoutSec 2).StatusCode; Start-Process 'http://localhost:5000'; Write-Host '[ok] Opened http://localhost:5000'; break }catch{ Start-Sleep 2; $n++ } }"

echo.
echo   Server running. Log: server.log
echo   Stop: close this window, or Ctrl+C.
echo.
pause
