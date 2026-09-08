#!/bin/sh
set -e

if [ ! -d /app/chroma_data ] || [ -z "$(ls -A /app/chroma_data 2>/dev/null)" ]; then
    echo "Knowledge base not found. Building it from data/ ..."
    python ingest.py
else
    echo "Knowledge base found. Skipping ingest."
fi

exec uvicorn main:app --host 0.0.0.0 --port 8000