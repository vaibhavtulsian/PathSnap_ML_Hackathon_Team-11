#!/usr/bin/env python3
"""
PathSnap - Laptop 1 Server Entry Point.
Run this file directly to start the FastAPI backend:
    python server.py
"""

import uvicorn

if __name__ == "__main__":
    print("=======================================================")
    print("🚀 Starting PathSnap Model & Evaluation Server (Laptop 1)")
    print("Host: 0.0.0.0 (Accessible to Laptop 2 across local network)")
    print("Port: 8000")
    print("Docs: http://localhost:8000/docs")
    print("Evaluators: Terminal logs will stream below on incoming requests.")
    print("=======================================================")
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
