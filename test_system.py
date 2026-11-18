#!/usr/bin/env python3
"""
PathSnap - 1-Click Verification Test for All 10 Evaluation Trips.
Run this file to verify that data loading, route inference, metrics, and maps work:
    python test_system.py
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from scripts.test_local import run_tests

if __name__ == "__main__":
    run_tests()
