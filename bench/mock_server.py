"""
Fake MeetMind server for checking bench/latency.py without Gemini or a key.

It answers a fixed number of milliseconds after the last input it received,
so the harness can be measured against a delay we already know. If the
harness reports roughly 400ms against a 400ms mock, the measurement is sound.

The reply is scheduled from the timestamp of the last message received, not
from the moment the server notices the client went quiet. Otherwise the gap
detection would be baked into every number.

Usage:
    python bench/mock_server.py --port 8765 --delay 0.4
"""

import argparse
import asyncio
import base64
import json
import time

import websockets

# How long without a message counts as the client having stopped. Audio
# mode sends a frame every 256ms so this has to clear that, but it also
# puts a floor on the delay the mock can simulate: anything shorter is
# swallowed by the detection window.
QUIET_GAP_SECONDS = 0.5
REPLY_PCM = b"\x00\x01" * 1200


async def answer_after_quiet(socket, delay, quiet_gap=QUIET_GAP_SECONDS):
    last_input_at = None

    while True:
        try:
            await asyncio.wait_for(socket.recv(), timeout=quiet_gap)
            last_input_at = time.perf_counter()
        except asyncio.TimeoutError:
            if last_input_at is None:
                continue
            break
        except websockets.ConnectionClosed:
            return

    due_at = last_input_at + delay
    await asyncio.sleep(max(0.0, due_at - time.perf_counter()))

    server_ms = round((time.perf_counter() - last_input_at) * 1000, 2)
    await socket.send(json.dumps({"type": "timing", "data": {"server_ms": server_ms}}))
    await socket.send(json.dumps({
        "type": "audio",
        "data": base64.b64encode(REPLY_PCM).decode(),
    }))
    await socket.send(json.dumps({"type": "turn_complete"}))


def serve(port, delay, quiet_gap=QUIET_GAP_SECONDS):
    async def handler(socket):
        try:
            await answer_after_quiet(socket, delay, quiet_gap)
        except websockets.ConnectionClosed:
            pass

    return websockets.serve(handler, "127.0.0.1", port)


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--delay", type=float, default=0.4, help="seconds before replying")
    args = parser.parse_args()

    async with serve(args.port, args.delay):
        print(f"mock server on ws://127.0.0.1:{args.port}, replying after {args.delay}s")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
