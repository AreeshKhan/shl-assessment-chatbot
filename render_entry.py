"""
============================================================
Render Deployment Entry Point
============================================================
This file is what Render runs. It imports the FastAPI app
from our app package. Render's start command points here:
    uvicorn app.main:app --host 0.0.0.0 --port $PORT
============================================================
"""
# This file exists so we can reference app.main:app in Render's config
# The actual app is in app/main.py
