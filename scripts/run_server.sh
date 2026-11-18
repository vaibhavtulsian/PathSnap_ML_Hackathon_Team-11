#!/usr/bin/env bash
# ========================================================
# PathSnap - Launch Server (Laptop 1)
# Hosts FastAPI inference backend & server-side evaluation
# ========================================================

echo "Starting PathSnap API Server on Laptop 1..."
echo "Host: 0.0.0.0 | Port: 8000"
echo "Evaluators: Terminal logs will display live incoming requests below."
echo "--------------------------------------------------------"

uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
