"""
MeetMind — FastAPI application entry point.

Routes:
    GET  /          serves the frontend (static/index.html)
    GET  /health    health check
    WS   /ws/{user_id}/{session_id}   Gemini Live bidirectional session
"""

import logging

from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from core.config import GOOGLE_API_KEY
from core.pipeline import run_session_pipeline

logging.basicConfig(level=logging.INFO)

if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY is not set. Copy .env.example to .env and add your key.")

app = FastAPI(title="MeetMind", version="1.0.0")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.websocket("/ws/{user_id}/{session_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str, session_id: str):
    await websocket.accept()
    await run_session_pipeline(websocket, user_id, session_id)
