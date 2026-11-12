import os
import sys

# Ensure repository root is on Python module search path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from app import app

# Vercel Serverless Function entry point
app = app
