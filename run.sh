#!/bin/bash
# Traffic Guardian AI - starts the live prediction API
# Usage: ./run.sh
cd "$(dirname "$0")"
PYTHON="python3"
if [ -d ".venv" ]; then
    PYTHON="./.venv/bin/python3"
fi
echo "Starting Traffic Guardian AI on http://127.0.0.1:8000 ..."
echo "Open http://127.0.0.1:8000 in your browser once it says 'Application startup complete'."
$PYTHON -m uvicorn app.main:app --reload --port 8000
