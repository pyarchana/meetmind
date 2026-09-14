"""
End to end latency benchmark for MeetMind.

Measures the gap between the last input sent and the first byte of audio that
comes back, over N runs, and reports p50 and p95.

Two modes:
    text    Sends a typed question. Fully deterministic, no audio file needed,
            and it isolates the model round trip from anything speech related.
    audio   Streams a 16kHz mono WAV at real time pace, then trailing silence.

Why the harness stops sending after the question:
    A live browser streams continuously, silence included, because Gemini runs
    its own voice activity detection and uses trailing silence to close a turn.
    That makes "the last frame we sent" meaningless as a start point, since it
    is always a few hundred ms old. The benchmark sends the question plus a
    fixed tail of silence and then stops, which gives one well defined end of
    input to measure from.

Usage:
    python bench/latency.py --mode text --runs 10
    python bench/latency.py --mode audio --wav bench/samples/question.wav --runs 10

Point it at a server started with MEETMIND_TIMING=1 to also collect the
server side number.
"""

import argparse
import asyncio
import base64
import json
import math
import statistics
import time
import uuid
import wave
from pathlib import Path

import websockets

SAMPLE_RATE = 16000
FRAME_SAMPLES = 4096                      # 256ms, same as the browser
FRAME_SECONDS = FRAME_SAMPLES / SAMPLE_RATE
TRAILING_SILENCE_SECONDS = 1.0
DEFAULT_QUESTION = "In one sentence, what is the capital of France?"


def percentile(values, p):
    """Nearest rank percentile, p in 0..100."""
    if not values:
        raise ValueError("percentile of an empty sample")
    ordered = sorted(values)
    rank = math.ceil(p / 100 * len(ordered))
    return ordered[max(rank, 1) - 1]


def summarise(runs):
    """Turn a list of trial dicts into reportable statistics."""
    ok = [r for r in runs if r.get("first_audio_ms") is not None]
    failed = len(runs) - len(ok)

    if not ok:
        return {"runs": len(runs), "failed": failed, "note": "no run produced audio"}

    client = [r["first_audio_ms"] for r in ok]
    server = [r["server_ms"] for r in ok if r.get("server_ms") is not None]

    summary = {
        "runs": len(runs),
        "failed": failed,
        "client_ms": {
            "mean": round(statistics.fmean(client), 1),
            "p50": round(percentile(client, 50), 1),
            "p95": round(percentile(client, 95), 1),
            "min": round(min(client), 1),
            "max": round(max(client), 1),
        },
    }

    if server:
        summary["server_ms"] = {
            "mean": round(statistics.fmean(server), 1),
            "p50": round(percentile(server, 50), 1),
            "p95": round(percentile(server, 95), 1),
        }
        # Both are durations, so this subtraction is safe across machines in a
        # way that subtracting two wall clock timestamps would not be.
        summary["transport_ms"] = {
            "p50": round(percentile(client, 50) - percentile(server, 50), 1),
        }

    return summary


def read_wav(path):
    """Read a 16kHz mono 16 bit WAV into raw PCM bytes."""
    with wave.open(str(path), "rb") as handle:
        if handle.getnchannels() != 1:
            raise SystemExit(f"{path}: expected mono, got {handle.getnchannels()} channels")
        if handle.getsampwidth() != 2:
            raise SystemExit(f"{path}: expected 16 bit samples")
        if handle.getframerate() != SAMPLE_RATE:
            raise SystemExit(
                f"{path}: expected {SAMPLE_RATE}Hz, got {handle.getframerate()}Hz. "
                "Resample it rather than letting the benchmark do it silently."
            )
        return handle.readframes(handle.getnframes())


def frames(pcm):
    """Split raw PCM into browser sized frames, padding the last one."""
    step = FRAME_SAMPLES * 2
    for start in range(0, len(pcm), step):
        chunk = pcm[start:start + step]
        yield chunk.ljust(step, b"\x00")


def silence_frames(seconds):
    count = max(1, round(seconds / FRAME_SECONDS))
    return [b"\x00" * (FRAME_SAMPLES * 2)] * count


async def run_trial(base_url, mode, pcm, question, timeout):
    """One question, one answer. Returns the measured milliseconds."""
    url = f"{base_url}/ws/bench/{uuid.uuid4().hex[:10]}"
    result = {"first_audio_ms": None, "first_transcript_ms": None, "server_ms": None}

    async with websockets.connect(url, max_size=None) as socket:
        if mode == "text":
            await socket.send(json.dumps({"type": "text", "data": question}))
        else:
            for frame in list(frames(pcm)) + silence_frames(TRAILING_SILENCE_SECONDS):
                await socket.send(json.dumps({
                    "type": "audio",
                    "data": base64.b64encode(frame).decode(),
                }))
                await asyncio.sleep(FRAME_SECONDS)   # real time pace

        sent_at = time.perf_counter()
        deadline = sent_at + timeout

        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                return result
            try:
                raw = await asyncio.wait_for(socket.recv(), timeout=remaining)
            except asyncio.TimeoutError:
                return result

            message = json.loads(raw)
            elapsed_ms = (time.perf_counter() - sent_at) * 1000

            if message["type"] == "timing":
                result["server_ms"] = message["data"]["server_ms"]
            elif message["type"] == "transcript_agent":
                result["first_transcript_ms"] = result["first_transcript_ms"] or elapsed_ms
            elif message["type"] == "audio":
                result["first_audio_ms"] = elapsed_ms
                return result
            elif message["type"] == "error":
                result["error"] = message["data"]
                return result


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://127.0.0.1:8000")
    parser.add_argument("--mode", choices=("text", "audio"), default="text")
    parser.add_argument("--wav", type=Path, help="16kHz mono WAV, required for audio mode")
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--out", type=Path, help="write the full result as JSON")
    parser.add_argument("--label", default="", help="network conditions, host, anything")
    args = parser.parse_args()

    pcm = None
    if args.mode == "audio":
        if not args.wav:
            raise SystemExit("audio mode needs --wav")
        pcm = read_wav(args.wav)

    runs = []
    for index in range(args.runs):
        trial = await run_trial(args.url, args.mode, pcm, args.question, args.timeout)
        runs.append(trial)

        shown = trial["first_audio_ms"]
        print(f"run {index + 1:>3}/{args.runs}  "
              + (f"{shown:8.1f} ms" if shown is not None else "   no audio")
              + (f"   server {trial['server_ms']:.1f} ms" if trial["server_ms"] else "")
              + (f"   error: {trial['error']}" if trial.get("error") else ""))

    report = {
        "label": args.label,
        "mode": args.mode,
        "url": args.url,
        "summary": summarise(runs),
        "runs": runs,
    }

    print()
    print(json.dumps(report["summary"], indent=2))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
