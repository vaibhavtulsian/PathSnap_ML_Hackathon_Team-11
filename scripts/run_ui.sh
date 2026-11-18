#!/usr/bin/env bash
# ========================================================
# PathSnap - Launch Evaluator UI (Laptop 2)
# Hosts Streamlit dashboard connecting to Laptop 1
# ========================================================

echo "Starting PathSnap Evaluator Dashboard on Laptop 2..."
echo "Ensure Laptop 1 is running and accessible on local network."
echo "--------------------------------------------------------"

streamlit run src/ui/app.py
