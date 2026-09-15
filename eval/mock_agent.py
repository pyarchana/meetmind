"""
Fake MeetMind for checking the eval runner without a key.

Replies come from a lookup you supply, so a run can be forced to answer
perfectly, answer wrongly, or hallucinate, and the scoring can then be checked
against an outcome that is already known. An eval harness that reports a good
score for deliberately wrong answers is worse than no harness.
"""

import argparse
import asyncio
import json

import websockets

DEFAULT_REPLY = "I do not have that."


def serve(port, answers=None, default=DEFAULT_REPLY):
    answers = answers or {}

    async def handler(socket):
        try:
            async for raw in socket:
                message = json.loads(raw)
                if message.get("type") != "text":
                    continue

                reply = answers.get(message["data"], default)
                await socket.send(json.dumps({
                    "type": "transcript_agent", "data": reply,
                }))
                await socket.send(json.dumps({
                    "type": "turn_complete", "data": {"interrupted": False},
                }))
        except websockets.ConnectionClosed:
            pass

    return websockets.serve(handler, "127.0.0.1", port)


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8811)
    args = parser.parse_args()

    async with serve(args.port):
        print(f"mock agent on ws://127.0.0.1:{args.port}")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
