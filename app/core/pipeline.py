"""
Async pipeline: two concurrent tasks bridging the browser WebSocket
and the Gemini Live API via Google ADK's LiveRequestQueue.

Architecture:
    Browser  <--WebSocket-->  upstream_task  -->  LiveRequestQueue  -->  Gemini Live
    Browser  <--WebSocket-->  downstream_task  <--  runner.run_live()  <--  Gemini Live

The two tasks run concurrently via asyncio.gather(). Each task has a
single responsibility:
    - upstream_task:   read browser messages, forward to Gemini
    - downstream_task: read Gemini events, forward to browser
"""

import asyncio
import base64
import json
import logging

from fastapi import WebSocket
from fastapi.websockets import WebSocketDisconnect
from google.adk.agents.live_request_queue import LiveRequestQueue
from google.genai import types

from core.config import INPUT_SAMPLE_RATE
from core.session import runner, build_run_config, get_or_create_session

logger = logging.getLogger(__name__)


async def upstream_task(
    websocket: WebSocket,
    live_request_queue: LiveRequestQueue,
) -> None:
    """
    Reads messages from the browser WebSocket and forwards them to
    Gemini Live via the LiveRequestQueue.

    Message types handled:
        audio  - raw PCM at 16kHz, base64 encoded
        screen - JPEG frame, base64 encoded
        text   - plain text question from the user
    """
    try:
        while True:
            raw = await websocket.receive_text()
            message = json.loads(raw)
            msg_type = message.get("type")

            if msg_type == "audio":
                audio_bytes = base64.b64decode(message["data"])
                blob = types.Blob(
                    mime_type=f"audio/pcm;rate={INPUT_SAMPLE_RATE}",
                    data=audio_bytes
                )
                live_request_queue.send_realtime(blob)

            elif msg_type == "screen":
                image_bytes = base64.b64decode(message["data"])
                blob = types.Blob(
                    mime_type="image/jpeg",
                    data=image_bytes
                )
                live_request_queue.send_realtime(blob)

            elif msg_type == "text":
                content = types.Content(
                    parts=[types.Part(text=message["data"])]
                )
                live_request_queue.send_content(content)

    except WebSocketDisconnect:
        logger.info("[upstream] client disconnected")
    except Exception as e:
        logger.error(f"[upstream] error: {e}")


async def downstream_task(
    websocket: WebSocket,
    live_request_queue: LiveRequestQueue,
    user_id: str,
    session_id: str,
) -> None:
    """
    Iterates over events from runner.run_live() and forwards them to
    the browser over the WebSocket.

    Events handled:
        audio parts          - PCM at 24kHz, forwarded as base64
        text parts           - text responses, forwarded as-is
        input_transcription  - user speech transcript (longest chunk wins)
        output_transcription - agent speech transcript (longest chunk wins)

    Deduplication strategy:
        Gemini streams transcriptions incrementally. Each chunk is longer
        than the previous as words are added. We track the last sent
        transcript length and only forward when the new text is longer,
        ensuring the browser always shows the most complete version without
        displaying duplicate or partial repeats.
    """
    last_agent_transcript = ""
    last_user_transcript = ""

    run_config = build_run_config()

    try:
        async for event in runner.run_live(
            user_id=user_id,
            session_id=session_id,
            live_request_queue=live_request_queue,
            run_config=run_config,
        ):
            # Audio and text content parts
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.inline_data and part.inline_data.mime_type.startswith("audio"):
                        audio_b64 = base64.b64encode(part.inline_data.data).decode()
                        await websocket.send_text(
                            json.dumps({"type": "audio", "data": audio_b64})
                        )
                    elif part.text:
                        await websocket.send_text(
                            json.dumps({"type": "text", "data": part.text})
                        )

            # User speech transcription
            input_t = getattr(event, "input_transcription", None)
            if input_t:
                txt = (getattr(input_t, "text", "") or "").strip()
                if txt and len(txt) > len(last_user_transcript):
                    last_user_transcript = txt
                    last_agent_transcript = ""  # new user turn resets agent buffer
                    await websocket.send_text(
                        json.dumps({"type": "transcript_user", "data": txt})
                    )

            # Agent speech transcription
            output_t = getattr(event, "output_transcription", None)
            if output_t:
                txt = (getattr(output_t, "text", "") or "").strip()
                if txt and len(txt) > len(last_agent_transcript):
                    last_agent_transcript = txt
                    await websocket.send_text(
                        json.dumps({"type": "transcript_agent", "data": txt})
                    )

    except Exception as e:
        logger.error(f"[downstream] error: {e}")
        try:
            await websocket.send_text(
                json.dumps({"type": "error", "data": str(e)})
            )
        except Exception:
            pass


async def run_session_pipeline(
    websocket: WebSocket,
    user_id: str,
    session_id: str,
) -> None:
    """
    Entry point for a full session. Creates the LiveRequestQueue,
    ensures the session exists, then runs upstream and downstream
    tasks concurrently until either completes or the client disconnects.
    """
    await get_or_create_session(user_id, session_id)
    live_request_queue = LiveRequestQueue()

    try:
        await asyncio.gather(
            upstream_task(websocket, live_request_queue),
            downstream_task(websocket, live_request_queue, user_id, session_id),
            return_exceptions=True,
        )
    finally:
        live_request_queue.close()
