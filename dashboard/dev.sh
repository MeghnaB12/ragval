#!/usr/bin/env bash
# Run the full dashboard locally: FastAPI backend + Vite frontend.
# Usage: ./dev.sh   (from the dashboard/ directory)
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "Installing ragval + backend deps..."
pip install -e "$ROOT" -q
pip install -r backend/requirements.txt -q

echo "Starting backend on :8000..."
( cd backend && uvicorn main:app --port 8000 --reload ) &
BACK=$!

echo "Starting frontend on :5173..."
( cd frontend && npm install && npm run dev )

kill $BACK 2>/dev/null || true
