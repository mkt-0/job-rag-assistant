#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

if ! command -v python3 >/dev/null 2>&1; then
  echo "[error] python3 not found. Install Python 3.10+ and retry."; exit 1
fi
PY=python3

if [ ! -d .venv ]; then
  echo "[setup] Creating virtual env .venv ..."
  $PY -m venv .venv
fi
# shellcheck disable=SC1091
. .venv/bin/activate
echo "[setup] Installing dependencies (first time only) ..."
pip install -q -r requirements.txt

echo "[check] 校验配置 (.env / API key) ..."
if ! python -c "import sys, rag_core; sys.exit(0 if rag_core.is_key_ready() else 1)"; then
  python -c "import rag_core; rag_core.preflight()"
  echo "[error] 请按上面提示在 .env 中填入有效的 SILICONFLOW_API_KEY 后，重新运行 run.sh"
  exit 1
fi

echo "[info] Starting server ... (首次启动会用 jobs.json 自动构建索引，可能耗时一两分钟)"
python app.py > server.log 2>&1 &
SERVER=$!

for i in $(seq 1 120); do
  if curl -s http://localhost:5000/health >/dev/null 2>&1; then
    echo "[ok] Opened http://localhost:5000"
    (command -v xdg-open >/dev/null && xdg-open http://localhost:5000) \
      || (command -v open >/dev/null && open http://localhost:5000) || true
    break
  fi
  sleep 2
done

echo "Server running. Log: server.log  |  Stop: Ctrl+C"
wait "$SERVER"
