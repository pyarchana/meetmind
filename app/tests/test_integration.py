"""
Integration checks that drive the real ASGI app.

The unit tests call upstream_task and downstream_task directly. These go in
through the actual websocket route instead, so they catch the wiring those
cannot: route registration, the accept handshake, both tasks starting, and
the connection closing cleanly afterwards.

Gemini is stubbed everywhere except the live test at the bottom, which is
skipped unless a real key is in the environment.
"""

import asyncio
import base64
import json
import os
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from google.genai import types

from conftest import PLACEHOLDER_KEY


def _event(**overrides):
    event = {
        "content": None,
        "input_transcription": None,
        "output_transcription": None,
        "actions": SimpleNamespace(state_delta=None),
        "turn_complete": False,
        "interrupted": False,
    }
    event.update(overrides)
    return SimpleNamespace(**event)


def _audio_event(pcm):
    part = SimpleNamespace(
        inline_data=SimpleNamespace(mime_type="audio/pcm", data=pcm),
        text=None,
    )
    return _event(content=SimpleNamespace(parts=[part]))


def _client_with(monkeypatch, events):
    import core.pipeline as pipeline
    import main

    async def fake_run_live(**kwargs):
        for event in events:
            yield event
        # A real live stream stays open for the whole session. Returning
        # here instead would tear the pipeline down before the client had
        # a chance to send anything.
        await asyncio.Event().wait()

    monkeypatch.setattr(pipeline.runner, "run_live", fake_run_live)
    return TestClient(main.app)


@pytest.fixture
def answering_client(monkeypatch):
    return _client_with(monkeypatch, [
        _event(output_transcription=types.Transcription(text="Paris.")),
        _audio_event(b"\x01\x02\x03\x04"),
        _event(actions=SimpleNamespace(
            state_delta={"decisions": [{"id": 1, "text": "ship on friday"}]}
        )),
        _event(turn_complete=True),
    ])


@pytest.fixture
def quiet_client(monkeypatch):
    """A model that never says anything, so only our own frames come back."""
    return _client_with(monkeypatch, [])


class TestRoutes:
    def test_health(self, quiet_client):
        response = quiet_client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_root_serves_the_frontend(self, quiet_client):
        response = quiet_client.get("/")
        assert response.status_code == 200
        assert "html" in response.headers["content-type"]


class TestWebSocketRoute:
    """
    Every send here is followed by a read of its reply. Leaving a message in
    flight when the `with` block closes the socket races the server teardown,
    which passes locally and fails on a slower CI runner.

    The stub emits its events on connect, independent of anything sent, so the
    downstream cases do not send at all. Sending there would prove nothing and
    reintroduce the race.
    """

    def test_model_events_reach_the_client(self, answering_client):
        with answering_client.websocket_connect("/ws/u1/s1") as socket:
            frames = [json.loads(socket.receive_text()) for _ in range(4)]

        by_type = {frame["type"]: frame for frame in frames}
        assert by_type["transcript_agent"]["data"] == "Paris."
        assert by_type["audio"]["data"]
        assert by_type["meeting_state"]["data"]["decisions"][0]["text"] == "ship on friday"
        assert "turn_complete" in by_type

    def test_a_good_audio_frame_draws_no_complaint(self, quiet_client):
        """
        The valid frame is answered with silence, so a bad frame is sent behind
        it. Messages are handled in order, so reaching the second error means
        the first was accepted.
        """
        with quiet_client.websocket_connect("/ws/u2/s2") as socket:
            socket.send_text(json.dumps({
                "type": "audio",
                "data": base64.b64encode(bytes([0, 1]) * 160).decode(),
            }))
            socket.send_text(json.dumps({"type": "audio", "data": "!!!not base64!!!"}))
            reply = json.loads(socket.receive_text())

        assert reply["type"] == "error"
        assert "base64" in reply["data"]

    def test_bad_frames_do_not_end_the_session(self, quiet_client):
        with quiet_client.websocket_connect("/ws/u3/s3") as socket:
            socket.send_text("{ not json")
            first = json.loads(socket.receive_text())

            # A second one proves the loop is still running after the first.
            socket.send_text(json.dumps({"type": "text", "data": "   "}))
            second = json.loads(socket.receive_text())

        assert first["type"] == "error" and "not valid JSON" in first["data"]
        assert second["type"] == "error" and "non-empty" in second["data"]

    def test_each_session_id_gets_its_own_board(self, answering_client):
        """Meeting state must not bleed between two different sessions."""
        boards = []
        for room in ("room-a", "room-b"):
            with answering_client.websocket_connect(f"/ws/u4/{room}") as socket:
                frames = [json.loads(socket.receive_text()) for _ in range(4)]
            state = next(f for f in frames if f["type"] == "meeting_state")
            boards.append(state["data"]["decisions"])

        assert all(len(board) == 1 for board in boards)


@pytest.mark.skipif(
    os.environ.get("GOOGLE_API_KEY", PLACEHOLDER_KEY) == PLACEHOLDER_KEY,
    reason="needs a real GOOGLE_API_KEY",
)
class TestAgainstRealGemini:
    """
    Opt in. Run with a real key to check the whole path for real:

        GOOGLE_API_KEY=... pytest tests/test_integration.py -v -k Real

    Left out of CI on purpose. It costs money, needs network, and a flaky
    model response should not turn the build red.
    """

    def test_a_question_gets_a_spoken_answer(self):
        import main

        with TestClient(main.app).websocket_connect("/ws/live/smoke") as socket:
            socket.send_text(json.dumps({
                "type": "text",
                "data": "In one short sentence, what is the capital of France?",
            }))

            heard_audio = False
            transcript = ""
            for _ in range(60):
                frame = json.loads(socket.receive_text())
                if frame["type"] == "audio":
                    heard_audio = True
                elif frame["type"] == "transcript_agent":
                    transcript = frame["data"]
                elif frame["type"] == "turn_complete":
                    break
                elif frame["type"] == "error":
                    pytest.fail(f"server reported: {frame['data']}")

        assert heard_audio, "no audio came back"
        assert "paris" in transcript.lower(), transcript
