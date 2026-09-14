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

from core.config import INPUT_SAMPLE_RATE, TIMING_ENABLED
from core.session import runner, build_run_config, get_or_create_session
from core.timing import TurnClock

logger = logging.getLogger(__name__)


def parse_client_message(raw: str) -> tuple[str, object]:
    """
    Turn a raw browser frame into something LiveRequestQueue accepts.

    Returns ("realtime", Blob) for audio and screen frames, or
    ("content", Content) for typed questions. Raises ValueError if the
    frame cannot be forwarded.
    """
    try:
        message = json.loads(raw)
    except json.JSONDecodeError:
        raise ValueError("not valid JSON") from None

    if not isinstance(message, dict):
        raise ValueError("expected a JSON object")

    msg_type = message.get("type")
    data = message.get("data")

    if msg_type == "text":
        if not isinstance(data, str) or not data.strip():
            raise ValueError("text needs a non-empty data string")
        return "content", types.Content(parts=[types.Part(text=data)])

    if msg_type in ("audio", "screen"):
        if not isinstance(data, str):
            raise ValueError(f"{msg_type} needs a base64 data string")
        try:
            decoded = base64.b64decode(data, validate=True)
        except ValueError:
            raise ValueError(f"{msg_type} data is not valid base64") from None
        mime_type = (
            f"audio/pcm;rate={INPUT_SAMPLE_RATE}" if msg_type == "audio" else "image/jpeg"
        )
        return "realtime", types.Blob(mime_type=mime_type, data=decoded)

    raise ValueError(f"unknown message type {msg_type!r}")


async def upstream_task(
    websocket: WebSocket,
    live_request_queue: LiveRequestQueue,
    clock: TurnClock,
) -> None:
    """
    Reads messages from the browser WebSocket and forwards them to
    Gemini Live via the LiveRequestQueue.

    A frame we cannot parse is reported back to the client and skipped,
    so one bad message does not take the connection down with it.
    """
    try:
        while True:
            raw = await websocket.receive_text()

            try:
                kind, payload = parse_client_message(raw)
            except ValueError as e:
                logger.warning("[upstream] rejected frame: %s", e)
                await websocket.send_text(
                    json.dumps({"type": "error", "data": f"ignored message: {e}"})
                )
                continue

            if kind == "realtime":
                live_request_queue.send_realtime(payload)
            else:
                live_request_queue.send_content(payload)
            clock.input_forwarded()

    except WebSocketDisconnect:
        logger.info("[upstream] client disconnected")


async def downstream_task(
    websocket: WebSocket,
    live_request_queue: LiveRequestQueue,
    user_id: str,
    session_id: str,
    clock: TurnClock,
) -> None:
    """
    Iterates over events from runner.run_live() and forwards them to
    the browser over the WebSocket.

    Events handled:
        audio parts          - PCM at 24kHz, forwarded as base64
        text parts           - text responses, forwarded as-is
        input_transcription  - user speech transcript (longest chunk wins)
        output_transcription - agent speech transcript (longest chunk wins)
        state_delta          - meeting state the tools just changed

    Meeting state:
        The tools reassign a whole list at a time, so each delta carries the
        complete value for every key it mentions. The browser can merge it
        key by key without needing to replay earlier deltas.

    Deduplication strategy:
        Gemini streams transcriptions incrementally, each chunk longer than
        the last as words are added, so within a turn we forward only when
        the new text is longer than what we already sent.

        Both buffers reset on a turn boundary. Without that, a short answer
        following a long one never reaches the browser, because it loses the
        length comparison against the previous turn.
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
                        answered_in = clock.first_audio_ms()
                        if answered_in is not None:
                            await websocket.send_text(
                                json.dumps({"type": "timing",
                                            "data": {"server_ms": answered_in}})
                            )
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

            # Meeting state the tools changed on this event
            if event.actions and event.actions.state_delta:
                await websocket.send_text(
                    json.dumps({
                        "type": "meeting_state",
                        "data": event.actions.state_delta,
                    })
                )

            if event.turn_complete or event.interrupted:
                clock.end_turn()
                last_user_transcript = ""
                last_agent_transcript = ""
                # The reason matters to the caller. A turn that ended because
                # somebody spoke over the agent is the thing bench/latency.py
                # times in interrupt mode, and it is not the same event as an
                # answer finishing on its own.
                await websocket.send_text(json.dumps({
                    "type": "turn_complete",
                    "data": {"interrupted": bool(event.interrupted)},
                }))

    except Exception as e:
        logger.error(f"[downstream] error: {e}")
        try:
            await websocket.send_text(
                json.dumps({"type": "error", "data": str(e)})
            )
        except Exception:
            pass


def report_failures(tasks) -> None:
    """
    Log whichever side fell over.

    Task.exception() raises CancelledError for a cancelled task rather than
    returning it, so a task the server cancelled from outside would otherwise
    take the whole teardown with it.
    """
    for task in tasks:
        if task.cancelled():
            continue
        error = task.exception()
        if error:
            logger.error("[%s] failed: %s", task.get_name(), error)


async def run_session_pipeline(
    websocket: WebSocket,
    user_id: str,
    session_id: str,
) -> None:
    """
    Entry point for a full session. Creates the LiveRequestQueue, ensures
    the session exists, then runs both tasks until one of them finishes.

    Whichever side ends first cancels the other. Waiting for both instead
    leaves downstream parked on run_live() after the browser is gone,
    holding the Gemini session open until the model happens to emit again.
    """
    await get_or_create_session(user_id, session_id)
    live_request_queue = LiveRequestQueue()
    clock = TurnClock(TIMING_ENABLED)

    tasks = [
        asyncio.create_task(
            upstream_task(websocket, live_request_queue, clock),
            name="upstream",
        ),
        asyncio.create_task(
            downstream_task(websocket, live_request_queue, user_id, session_id, clock),
            name="downstream",
        ),
    ]

    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        report_failures(done)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        live_request_queue.close()
