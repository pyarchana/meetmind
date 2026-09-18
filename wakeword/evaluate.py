"""
Scoring a wake word detector.

Reports the two numbers issue #3 asks for, and they are not symmetric in cost:

    false reject   the user said the wake word and nothing happened. Annoying,
                   and they will say it again.
    false accept   the agent woke up during a sentence that was not addressed
                   to it, in a meeting with other people in the room. That is
                   the one that makes the product embarrassing to use, so it
                   is the number to optimise against.

Reporting a single accuracy figure would hide that asymmetry, which is why
nothing here returns one.
"""

import time

import numpy as np

from detectors import frames_of


def run_clip(detector, samples):
    """Feed one clip through frame by frame. Returns (fired, frames_seen)."""
    detector.reset()
    fired = False
    seen = 0

    for frame in frames_of(samples):
        seen += 1
        if detector.push(frame) and not fired:
            fired = True

    return fired, seen


def evaluate(detector, clips):
    """
    Score a detector over labelled clips.

    `clips` is an iterable of (samples, is_wake_word).
    """
    positives = negatives = 0
    hits = false_alarms = 0
    frames_total = 0

    started = time.perf_counter()
    for samples, is_wake in clips:
        fired, seen = run_clip(detector, samples)
        frames_total += seen

        if is_wake:
            positives += 1
            hits += bool(fired)
        else:
            negatives += 1
            false_alarms += bool(fired)
    elapsed = time.perf_counter() - started

    return {
        "detector": getattr(detector, "name", type(detector).__name__),
        "positives": positives,
        "negatives": negatives,
        "false_reject_rate": round(1 - hits / positives, 4) if positives else None,
        "false_accept_rate": round(false_alarms / negatives, 4) if negatives else None,
        "recall": round(hits / positives, 4) if positives else None,
        "frames": frames_total,
        "ms_per_frame": round(elapsed / frames_total * 1000, 3) if frames_total else None,
    }


def latency(detector, frame, repeats=200):
    """
    Milliseconds for one push, which is the number that has to stay under 10.

    Measured after a warm up, because the first call through numpy pays for
    buffer allocation and would flatter or ruin the figure depending on luck.
    """
    detector.reset()
    for _ in range(20):
        detector.push(frame)

    started = time.perf_counter()
    for _ in range(repeats):
        detector.push(frame)
    elapsed = time.perf_counter() - started

    return round(elapsed / repeats * 1000, 3)


def compare(detectors, clips, frame=None):
    """Run several detectors over the same clips so the numbers line up."""
    clips = list(clips)
    rows = [evaluate(detector, clips) for detector in detectors]

    if frame is not None:
        for detector, row in zip(detectors, rows):
            row["push_ms"] = latency(detector, frame)
            row["within_budget"] = row["push_ms"] < 10.0

    return rows


def render(rows):
    """A table for the terminal and the write up."""
    header = f"{'detector':<14} {'FA':>8} {'FR':>8} {'recall':>8} {'push ms':>9}"
    lines = [header, "-" * len(header)]

    for row in rows:
        push = row.get("push_ms")
        lines.append(
            f"{row['detector']:<14} "
            f"{_pct(row['false_accept_rate']):>8} "
            f"{_pct(row['false_reject_rate']):>8} "
            f"{_pct(row['recall']):>8} "
            f"{('-' if push is None else f'{push:.3f}'):>9}"
        )
    return "\n".join(lines)


def _pct(value):
    return "-" if value is None else f"{value * 100:.1f}%"
