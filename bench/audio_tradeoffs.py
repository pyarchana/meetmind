"""
Offline measurements of the audio transport choices.

Answers the parts of issue #4 that do not need a live model. Bytes on the wire
and CPU per second of audio are properties of the encoding, not of Gemini, so
they can be measured here and the numbers committed.

What this cannot answer: whether any of it changes transcription accuracy.
That needs a real run, and packet_loss.py covers the harness for it.

Usage:
    python bench/audio_tradeoffs.py --out bench/results/audio_tradeoffs.json
"""

import argparse
import array
import base64
import json
import math
import time
from pathlib import Path

SAMPLE_RATE = 16000
FRAME_SAMPLES = 4096
BYTES_PER_SAMPLE = 2


def voice_like(samples, rate=SAMPLE_RATE):
    """
    A deterministic speech shaped signal.

    A pure tone would understate quantisation error, because real speech
    spends most of its time well below peak amplitude where the error is
    proportionally louder. This stacks a 120Hz fundamental with harmonics and
    modulates the envelope at syllable rate to get a similar distribution.
    """
    out = array.array("h", bytes(samples * 2))
    for i in range(samples):
        t = i / rate
        envelope = 0.35 * (1 + math.sin(2 * math.pi * 3.5 * t)) / 2 + 0.05
        tone = (
            math.sin(2 * math.pi * 120 * t)
            + 0.5 * math.sin(2 * math.pi * 240 * t)
            + 0.25 * math.sin(2 * math.pi * 480 * t)
            + 0.12 * math.sin(2 * math.pi * 960 * t)
        ) / 1.87
        out[i] = max(-32768, min(32767, int(tone * envelope * 32767)))
    return out


def snr_db(original, restored):
    """Signal to noise ratio in dB between two equal length sample arrays."""
    if len(original) != len(restored):
        raise ValueError("arrays must be the same length")

    signal = sum(float(s) * s for s in original)
    noise = sum((float(a) - b) ** 2 for a, b in zip(original, restored))

    if noise == 0:
        return math.inf
    if signal == 0:
        return -math.inf
    return round(10 * math.log10(signal / noise), 2)


def to_8bit(samples):
    """
    Drop to 8 bit, then back, so the loss can be measured.

    Rounds rather than truncating. Truncation is a shift and looks tidier in
    code, but it biases every sample toward zero and costs roughly another
    6dB, which would make 8 bit look worse than it is.

    This is linear 8 bit. Telephony uses u-law, which spends its levels where
    speech actually sits and does considerably better at the same bit depth.
    """
    restored = array.array("h", bytes(len(samples) * 2))
    for i, value in enumerate(samples):
        restored[i] = max(-128, min(127, round(value / 256))) * 256
    return restored


def json_frame(pcm: bytes) -> bytes:
    """What the browser sends today: base64 inside a JSON text frame."""
    return json.dumps({
        "type": "audio",
        "data": base64.b64encode(pcm).decode(),
    }).encode()


def binary_frame(pcm: bytes) -> bytes:
    """What a websocket binary frame would carry: the bytes themselves."""
    return pcm


def _time(fn, repeats):
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - start) / repeats


def measure_encoding(frame_samples=FRAME_SAMPLES, repeats=2000):
    """Bytes and CPU for one frame, each way, plus the per second cost."""
    pcm = voice_like(frame_samples).tobytes()
    frames_per_second = SAMPLE_RATE / frame_samples

    as_json = json_frame(pcm)
    as_binary = binary_frame(pcm)

    encoded = base64.b64encode(pcm).decode()

    return {
        "frame_samples": frame_samples,
        "frame_ms": round(frame_samples / SAMPLE_RATE * 1000, 1),
        "raw_bytes": len(pcm),
        "json_base64_bytes": len(as_json),
        "binary_bytes": len(as_binary),
        "overhead_pct": round((len(as_json) / len(as_binary) - 1) * 100, 1),
        "json_kb_per_sec": round(len(as_json) * frames_per_second / 1024, 1),
        "binary_kb_per_sec": round(len(as_binary) * frames_per_second / 1024, 1),
        "encode_us": round(_time(lambda: json_frame(pcm), repeats) * 1e6, 1),
        "decode_us": round(_time(lambda: base64.b64decode(encoded), repeats) * 1e6, 1),
    }


def measure_frame_sizes(sizes=(1024, 2048, 4096, 8192), repeats=500):
    """
    What each buffer size costs.

    Smaller buffers cut the latency floor, because nothing can be detected
    faster than one frame. They also multiply the number of JSON envelopes per
    second, so the wire cost moves the other way.
    """
    rows = []
    for size in sizes:
        pcm = voice_like(size).tobytes()
        per_second = SAMPLE_RATE / size
        frame = json_frame(pcm)

        rows.append({
            "frame_samples": size,
            "frame_ms": round(size / SAMPLE_RATE * 1000, 1),
            "frames_per_sec": round(per_second, 1),
            "vad_floor_ms": round(size / SAMPLE_RATE * 1000 * 3, 1),
            "json_kb_per_sec": round(len(frame) * per_second / 1024, 1),
            "envelope_bytes_per_sec": round(
                (len(frame) - len(base64.b64encode(pcm))) * per_second, 1
            ),
            "encode_us_per_sec_audio": round(
                _time(lambda: json_frame(pcm), repeats) * per_second * 1e6, 1
            ),
        })
    return rows


def measure_quantisation(seconds=2.0):
    """
    Halving the bit depth halves the bytes. The question is what it costs in
    signal quality, which is measurable here. Whether that loss changes what
    Gemini transcribes is not, and needs a real run.
    """
    samples = voice_like(int(SAMPLE_RATE * seconds))
    restored = to_8bit(samples)

    return {
        "seconds": seconds,
        "bytes_16bit": len(samples) * 2,
        "bytes_8bit": len(samples),
        "bytes_saved_pct": 50.0,
        "snr_db": snr_db(samples, restored),
        "note": (
            "SNR is measured on a synthetic speech shaped signal. The effect "
            "on transcription accuracy needs a real run and real speech."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    report = {
        "encoding": measure_encoding(),
        "frame_sizes": measure_frame_sizes(),
        "quantisation": measure_quantisation(),
    }

    enc = report["encoding"]
    print(f"Encoding, {enc['frame_ms']}ms frames")
    print(f"  json+base64   {enc['json_kb_per_sec']} KB/s   {enc['json_base64_bytes']} B/frame")
    print(f"  binary        {enc['binary_kb_per_sec']} KB/s   {enc['binary_bytes']} B/frame")
    print(f"  overhead      {enc['overhead_pct']} percent")
    print(f"  cpu           {enc['encode_us']}us encode, {enc['decode_us']}us decode")

    print("\nFrame sizes")
    print(f"  {'samples':>8} {'ms':>7} {'vad floor':>10} {'KB/s':>8} {'cpu us/s':>9}")
    for row in report["frame_sizes"]:
        print(f"  {row['frame_samples']:>8} {row['frame_ms']:>7} "
              f"{row['vad_floor_ms']:>10} {row['json_kb_per_sec']:>8} "
              f"{row['encode_us_per_sec_audio']:>9}")

    q = report["quantisation"]
    print(f"\nQuantisation 16 to 8 bit: {q['bytes_saved_pct']} percent smaller, "
          f"SNR {q['snr_db']} dB")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
