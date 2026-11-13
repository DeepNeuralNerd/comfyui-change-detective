#!/usr/bin/env python3
"""Run the test-suite with no PYTHONPATH setup:  python run_tests.py"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, os.path.join(HERE, "src"))
sys.path.insert(1, os.path.join(HERE, "tests"))

if __name__ == "__main__":
    unittest.main(module=None,
                  argv=[sys.argv[0], "discover", "-s", "tests", "-t", "tests"])
