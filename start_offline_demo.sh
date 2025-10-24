#!/usr/bin/env bash
# start_offline_demo.sh
# Minimal start script: activate venv and run uvicorn (development).
# Edit PYTHONPATH or Uvicorn options as needed.

set -e

# path to your virtualenv (adjust if different)
VENV="./venv"

if [ -f "$VENV/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
else
  echo "Virtualenv not found at $VENV. Make sure you created the venv and installed requirements."
fi

export OLLAMA_HOST="${OLLAMA_HOST:-http://127.0.0.1:11434}"
export OLLAMA_MODEL="${OLLAMA_MODEL:-mistral:7b}"

echo "Activating venv..."
echo "Starting server..."

# run uvicorn with reload for development
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
