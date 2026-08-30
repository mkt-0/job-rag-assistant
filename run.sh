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

echo "[info] Starting server ... (first launch builds the index, may take a minute)"
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
