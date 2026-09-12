"""
MeetMind test suite.

Run with:
    pytest tests/ -v
"""

import asyncio
import base64
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.websockets import WebSocketDisconnect
from google.genai import types


# ---------------------------------------------------------------------------
# core/config.py
# ---------------------------------------------------------------------------

class TestConfig:
    def test_app_name(self):
        from core.config import APP_NAME
        assert APP_NAME == "meetmind"

    def test_input_sample_rate(self):
        from core.config import INPUT_SAMPLE_RATE
        assert INPUT_SAMPLE_RATE == 16000


# ---------------------------------------------------------------------------
# core/session.py
# ---------------------------------------------------------------------------

class TestSession:
    def test_build_run_config(self):
        from core.session import build_run_config
        from google.adk.agents.run_config import StreamingMode
        config = build_run_config()
        assert config.streaming_mode == StreamingMode.BIDI
        assert "AUDIO" in config.response_modalities

    @pytest.mark.asyncio
    async def test_get_or_create_session_creates_when_missing(self):
        from core.session import get_or_create_session, session_service
        from core.config import APP_NAME

        await get_or_create_session("test_user_001", "test_session_001")
        session = await session_service.get_session(
            app_name=APP_NAME,
            user_id="test_user_001",
            session_id="test_session_001"
        )
        assert session is not None

    @pytest.mark.asyncio
    async def test_get_or_create_session_idempotent(self):
        """Calling twice should not raise and should return the same session."""
        from core.session import get_or_create_session, session_service
        from core.config import APP_NAME

        await get_or_create_session("test_user_002", "test_session_002")
        await get_or_create_session("test_user_002", "test_session_002")

        session = await session_service.get_session(
            app_name=APP_NAME,
            user_id="test_user_002",
            session_id="test_session_002"
        )
        assert session is not None


# ---------------------------------------------------------------------------
# core/pipeline.py: parse_client_message
# ---------------------------------------------------------------------------

class TestParseClientMessage:
    def test_audio_frame_becomes_a_pcm_blob(self):
        from core.pipeline import parse_client_message
        from core.config import INPUT_SAMPLE_RATE

        pcm = b"\x00\x01" * 160
        kind, blob = parse_client_message(
            json.dumps({"type": "audio", "data": base64.b64encode(pcm).decode()})
        )
        assert kind == "realtime"
        assert blob.mime_type == f"audio/pcm;rate={INPUT_SAMPLE_RATE}"
        assert blob.data == pcm

    def test_screen_frame_becomes_a_jpeg_blob(self):
        from core.pipeline import parse_client_message

        jpeg = b"\xff\xd8\xff" + b"\x00" * 100
        kind, blob = parse_client_message(
            json.dumps({"type": "screen", "data": base64.b64encode(jpeg).decode()})
        )
        assert kind == "realtime"
        assert blob.mime_type == "image/jpeg"
        assert blob.data == jpeg

    def test_text_frame_becomes_content(self):
        from core.pipeline import parse_client_message

        kind, content = parse_client_message(
            json.dumps({"type": "text", "data": "What is the ROI formula?"})
        )
        assert kind == "content"
        assert content.parts[0].text == "What is the ROI formula?"

    @pytest.mark.parametrize("raw, reason", [
        ("{ not json", "malformed json"),
        (json.dumps(["not", "an", "object"]), "json array rather than object"),
        (json.dumps({"type": "audio"}), "missing data key"),
        (json.dumps({"type": "audio", "data": "!!!not base64!!!"}), "undecodable payload"),
        (json.dumps({"type": "audio", "data": 12345}), "non-string payload"),
        (json.dumps({"type": "text", "data": "   "}), "blank question"),
        (json.dumps({"type": "mystery", "data": "x"}), "unknown type"),
    ])
    def test_unforwardable_frames_are_rejected(self, raw, reason):
        from core.pipeline import parse_client_message
        with pytest.raises(ValueError):
            parse_client_message(raw)


# ---------------------------------------------------------------------------
# core/pipeline.py: upstream_task
# ---------------------------------------------------------------------------

class TestUpstreamTask:
    @pytest.mark.asyncio
    async def test_audio_message_forwarded(self):
        from core.pipeline import upstream_task

        pcm = b"\x00\x01" * 160
        websocket = AsyncMock()
        websocket.receive_text = AsyncMock(side_effect=[
            json.dumps({"type": "audio", "data": base64.b64encode(pcm).decode()}),
            WebSocketDisconnect(),
        ])

        queue = MagicMock()
        await upstream_task(websocket, queue)

        assert queue.send_realtime.call_args[0][0].data == pcm

    @pytest.mark.asyncio
    async def test_text_message_forwarded(self):
        from core.pipeline import upstream_task

        websocket = AsyncMock()
        websocket.receive_text = AsyncMock(side_effect=[
            json.dumps({"type": "text", "data": "What is the ROI formula?"}),
            WebSocketDisconnect(),
        ])

        queue = MagicMock()
        await upstream_task(websocket, queue)

        assert queue.send_content.call_args[0][0].parts[0].text == "What is the ROI formula?"

    @pytest.mark.asyncio
    async def test_screen_message_forwarded(self):
        from core.pipeline import upstream_task

        jpeg = b"\xff\xd8\xff" + b"\x00" * 100
        websocket = AsyncMock()
        websocket.receive_text = AsyncMock(side_effect=[
            json.dumps({"type": "screen", "data": base64.b64encode(jpeg).decode()}),
            WebSocketDisconnect(),
        ])

        queue = MagicMock()
        await upstream_task(websocket, queue)

        assert queue.send_realtime.call_args[0][0].mime_type == "image/jpeg"

    @pytest.mark.asyncio
    async def test_unknown_message_type_never_reaches_gemini(self):
        from core.pipeline import upstream_task

        websocket = AsyncMock()
        websocket.receive_text = AsyncMock(side_effect=[
            json.dumps({"type": "unknown", "data": "whatever"}),
            WebSocketDisconnect(),
        ])

        queue = MagicMock()
        await upstream_task(websocket, queue)

        queue.send_realtime.assert_not_called()
        queue.send_content.assert_not_called()

    @pytest.mark.asyncio
    async def test_bad_frames_do_not_end_the_session(self):
        """
        Regression: the loop used to sit under one try/except, so the first
        malformed frame killed upstream while the client still showed
        Connected.
        """
        from core.pipeline import upstream_task

        websocket = AsyncMock()
        websocket.receive_text = AsyncMock(side_effect=[
            "{ not json",
            json.dumps({"type": "audio", "data": "!!!not base64!!!"}),
            json.dumps({"type": "text", "data": "did this survive?"}),
            WebSocketDisconnect(),
        ])

        queue = MagicMock()
        await upstream_task(websocket, queue)

        assert queue.send_content.call_count == 1
        assert queue.send_content.call_args[0][0].parts[0].text == "did this survive?"

        replies = [json.loads(c[0][0]) for c in websocket.send_text.call_args_list]
        assert [m["type"] for m in replies] == ["error", "error"]


# ---------------------------------------------------------------------------
# core/pipeline.py: downstream_task
# ---------------------------------------------------------------------------

def _event(agent_text=None, user_text=None, audio=None,
           turn_complete=False, interrupted=False):
    """Minimal stand-in for one ADK live event."""
    content = None
    if audio is not None:
        part = SimpleNamespace(
            inline_data=SimpleNamespace(mime_type="audio/pcm", data=audio),
            text=None,
        )
        content = SimpleNamespace(parts=[part])

    return SimpleNamespace(
        content=content,
        input_transcription=types.Transcription(text=user_text) if user_text else None,
        output_transcription=types.Transcription(text=agent_text) if agent_text else None,
        turn_complete=turn_complete,
        interrupted=interrupted,
    )


async def _browser_sees(events):
    """Run downstream_task over `events` and return the frames it sent."""
    import core.pipeline as pipeline

    async def stream(**kwargs):
        for event in events:
            yield event

    websocket = AsyncMock()
    with patch.object(pipeline.runner, "run_live", stream):
        await pipeline.downstream_task(websocket, MagicMock(), "u", "s")

    return [json.loads(call[0][0]) for call in websocket.send_text.call_args_list]


def _agent_lines(frames):
    return [f["data"] for f in frames if f["type"] == "transcript_agent"]


class TestDownstreamTask:
    @pytest.mark.asyncio
    async def test_audio_forwarded_as_base64(self):
        frames = await _browser_sees([_event(audio=b"\x01\x02\x03")])
        assert frames == [
            {"type": "audio", "data": base64.b64encode(b"\x01\x02\x03").decode()}
        ]

    @pytest.mark.asyncio
    async def test_user_speech_forwarded(self):
        frames = await _browser_sees([_event(user_text="what is our runway")])
        assert frames == [
            {"type": "transcript_user", "data": "what is our runway"}
        ]

    @pytest.mark.asyncio
    async def test_only_the_longest_chunk_of_a_turn_is_sent(self):
        frames = await _browser_sees([
            _event(agent_text="The ROI formula"),
            _event(agent_text="The ROI"),          # shorter revision
            _event(agent_text="The ROI formula"),  # same length as the first
        ])
        assert _agent_lines(frames) == ["The ROI formula"]

    @pytest.mark.asyncio
    async def test_blank_transcript_ignored(self):
        frames = await _browser_sees([_event(agent_text="   ")])
        assert _agent_lines(frames) == []

    @pytest.mark.asyncio
    async def test_turn_complete_reaches_the_browser(self):
        frames = await _browser_sees([_event(turn_complete=True)])
        assert {"type": "turn_complete"} in frames

    @pytest.mark.asyncio
    async def test_short_answer_after_a_long_one_still_arrives(self):
        """
        Regression: buffers only reset on user speech, so a typed follow up
        answered more briefly than the previous turn lost the length
        comparison and never reached the browser.
        """
        long_answer = "The ROI formula is net profit divided by cost."
        short_answer = "About twelve percent."

        frames = await _browser_sees([
            _event(agent_text=long_answer),
            _event(turn_complete=True),
            _event(agent_text=short_answer),
        ])
        assert _agent_lines(frames) == [long_answer, short_answer]

    @pytest.mark.asyncio
    async def test_interruption_also_resets_the_buffers(self):
        frames = await _browser_sees([
            _event(agent_text="A long sentence that the user cut off midway"),
            _event(interrupted=True),
            _event(agent_text="Short."),
        ])
        assert "Short." in _agent_lines(frames)


# ---------------------------------------------------------------------------
# core/pipeline.py: run_session_pipeline
# ---------------------------------------------------------------------------

class TestRunSessionPipeline:
    @pytest.mark.asyncio
    async def test_disconnect_tears_the_session_down(self):
        """
        Regression: gather() waited for both tasks, so after the browser left
        downstream stayed parked on run_live() holding the Gemini session
        open until the model happened to emit again.
        """
        import core.pipeline as pipeline

        async def never_emits(**kwargs):
            await asyncio.Event().wait()
            yield  # pragma: no cover

        websocket = AsyncMock()
        websocket.receive_text = AsyncMock(side_effect=[WebSocketDisconnect()])
        queue = MagicMock()

        with patch.object(pipeline, "get_or_create_session", AsyncMock()), \
             patch.object(pipeline, "LiveRequestQueue", return_value=queue), \
             patch.object(pipeline.runner, "run_live", never_emits):
            await asyncio.wait_for(
                pipeline.run_session_pipeline(websocket, "u", "s"), timeout=5
            )

        assert queue.close.called
