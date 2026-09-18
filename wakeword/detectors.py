"""
Detectors that decide whether the wake word just occurred.

Everything here takes one mic frame at a time and answers "did it fire", so a
trained model and the current heuristic can be measured through the same
harness on the same clips.

About the baseline. The RMS gate is not a wake word detector at all, it is a
speech detector, and that is exactly why it is the honest thing to compare
against. It is what ships today. On clips containing any speech it fires
regardless of the words, so its false accept rate on non wake word speech
should come out near total. A model that cannot beat that is not earning the
complexity it costs, and a model compared against nothing at all would look
impressive without meaning anything.
"""

import numpy as np

from features import MIC_FRAME, RollingWindow

# Ported from web/src/lib/audio.js. Kept in step by test_wakeword.py, which
# fails if the two drift apart.
SPEECH_THRESHOLD = 0.012
SPEECH_FRAMES = 3


def rms(frame):
    frame = np.asarray(frame, dtype=np.float32)
    if len(frame) == 0:
        return 0.0
    return float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))


class RmsGate:
    """
    The heuristic in audio.js today: fire once energy stays above a threshold
    for SPEECH_FRAMES consecutive callbacks.

    Fires on the transition only, not on every loud frame, matching the
    browser where onSpeechStart runs once per utterance.
    """

    name = "rms gate"

    def __init__(self, threshold=SPEECH_THRESHOLD, frames=SPEECH_FRAMES):
        self.threshold = threshold
        self.frames = frames
        self.reset()

    def reset(self):
        self.loud_run = 0
        self.firing = False

    def push(self, frame):
        if rms(frame) > self.threshold:
            self.loud_run += 1
            if self.loud_run >= self.frames and not self.firing:
                self.firing = True
                return True
            return False

        self.loud_run = 0
        self.firing = False
        return False


class ModelDetector:
    """
    Wraps a trained scorer behind the same interface.

    `score` takes a log mel window of shape (frames, mels) and returns a
    probability. Kept separate from any training framework so the harness does
    not depend on one, and so a model can be swapped without touching the
    evaluation.

    `cooldown_frames` stops one utterance registering as many detections. The
    window is a rolling second of audio, so without it the wake word sits in
    the buffer and scores high on every callback for about fifteen frames in a
    row, which would read as fifteen separate wake ups.
    """

    name = "model"

    def __init__(self, score, threshold=0.5, cooldown_frames=16):
        self.score = score
        self.threshold = threshold
        self.cooldown_frames = cooldown_frames
        self.reset()

    def reset(self):
        self.window = RollingWindow()
        self.cooldown = 0

    def push(self, frame):
        self.window.push(frame)

        if self.cooldown > 0:
            self.cooldown -= 1
            return False

        if self.score(self.window.features()) >= self.threshold:
            self.cooldown = self.cooldown_frames
            return True
        return False


def frames_of(samples, size=MIC_FRAME):
    """Chop a clip into mic sized frames, the way the browser would deliver it."""
    samples = np.asarray(samples, dtype=np.float32)
    for start in range(0, len(samples) - size + 1, size):
        yield samples[start:start + size]
