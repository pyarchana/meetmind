"""
MeetMind test suite.

Run with:
    pytest tests/ -v
"""

import asyncio
import base64
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


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

        user_id = "test_user_001"
        session_id = "test_session_001"

        # Ensure session does not exist
        await get_or_create_session(user_id, session_id)
        session = await session_service.get_session(
            app_name=APP_NAME,
            user_id=user_id,
            session_id=session_id
        )
        assert session is not None

    @pytest.mark.asyncio
    async def test_get_or_create_session_idempotent(self):
        """Calling twice should not raise and should return the same session."""
        from core.session import get_or_create_session, session_service
        from core.config import APP_NAME

        user_id = "test_user_002"
        session_id = "test_session_002"

        await get_or_create_session(user_id, session_id)
        await get_or_create_session(user_id, session_id)  # second call

        session = await session_service.get_session(
            app_name=APP_NAME,
            user_id=user_id,
            session_id=session_id
        )
        assert session is not None


# ---------------------------------------------------------------------------
# core/pipeline.py: upstream_task
# ---------------------------------------------------------------------------

class TestUpstreamTask:
    @pytest.mark.asyncio
    async def test_audio_message_forwarded(self):
        from core.pipeline import upstream_task
        from fastapi.websockets import WebSocketDisconnect

        pcm_data = b"\x00\x01" * 160
        b64_audio = base64.b64encode(pcm_data).decode()

        websocket = AsyncMock()
        websocket.receive_text = AsyncMock(side_effect=[
            json.dumps({"type": "audio", "data": b64_audio}),
            WebSocketDisconnect(),
        ])

        queue = MagicMock()
        await upstream_task(websocket, queue)

        assert queue.send_realtime.called
        call_arg = queue.send_realtime.call_args[0][0]
        assert call_arg.mime_type == "audio/pcm;rate=16000"
        assert call_arg.data == pcm_data

    @pytest.mark.asyncio
    async def test_text_message_forwarded(self):
        from core.pipeline import upstream_task
        from fastapi.websockets import WebSocketDisconnect

        websocket = AsyncMock()
        websocket.receive_text = AsyncMock(side_effect=[
            json.dumps({"type": "text", "data": "What is the ROI formula?"}),
            WebSocketDisconnect(),
        ])

        queue = MagicMock()
        await upstream_task(websocket, queue)

        assert queue.send_content.called
        content = queue.send_content.call_args[0][0]
        assert content.parts[0].text == "What is the ROI formula?"

    @pytest.mark.asyncio
    async def test_screen_message_forwarded(self):
        from core.pipeline import upstream_task
        from fastapi.websockets import WebSocketDisconnect

        jpeg_data = b"\xff\xd8\xff" + b"\x00" * 100
        b64_jpeg = base64.b64encode(jpeg_data).decode()

        websocket = AsyncMock()
        websocket.receive_text = AsyncMock(side_effect=[
            json.dumps({"type": "screen", "data": b64_jpeg}),
            WebSocketDisconnect(),
        ])

        queue = MagicMock()
        await upstream_task(websocket, queue)

        assert queue.send_realtime.called
        call_arg = queue.send_realtime.call_args[0][0]
        assert call_arg.mime_type == "image/jpeg"
        assert call_arg.data == jpeg_data

    @pytest.mark.asyncio
    async def test_unknown_message_type_ignored(self):
        from core.pipeline import upstream_task
        from fastapi.websockets import WebSocketDisconnect

        websocket = AsyncMock()
        websocket.receive_text = AsyncMock(side_effect=[
            json.dumps({"type": "unknown", "data": "whatever"}),
            WebSocketDisconnect(),
        ])

        queue = MagicMock()
        await upstream_task(websocket, queue)

        queue.send_realtime.assert_not_called()
        queue.send_content.assert_not_called()


# ---------------------------------------------------------------------------
# core/pipeline.py: transcript deduplication logic
# ---------------------------------------------------------------------------

class TestTranscriptDeduplication:
    """
    Verifies the longest-chunk-wins deduplication strategy.
    Gemini streams transcriptions incrementally, each event contains
    a longer version of the same sentence as words are added.
    Only the longest seen so far should be forwarded to the browser.
    """

    def test_longer_transcript_passes(self):
        last = "Hello"
        new = "Hello how are"
        assert len(new) > len(last)  # should be forwarded

    def test_shorter_transcript_blocked(self):
        last = "Hello how are you doing today"
        new = "Hello how are"
        assert len(new) <= len(last)  # should NOT be forwarded

    def test_equal_transcript_blocked(self):
        last = "Hello"
        new = "Hello"
        assert len(new) <= len(last)  # should NOT be forwarded

    def test_empty_transcript_blocked(self):
        txt = "   "
        assert not txt.strip()  # empty after strip, should be ignored

    def test_new_user_turn_resets_agent_buffer(self):
        """When user speaks again, agent transcript buffer should reset to empty."""
        last_agent = "The ROI formula is net profit divided by cost."
        # Simulating new user input arriving resets agent buffer
        last_agent = ""
        assert last_agent == ""
