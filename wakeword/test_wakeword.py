"""
Checks for the wake word feature and evaluation layers.

No trained model yet, so nothing here asserts detection quality. What it does
assert is that the machinery around a model is correct: that the features are
what they claim to be, that the baseline matches the browser, and that the
scoring maths produces the right numbers for detectors whose behaviour is
known in advance. Getting that wrong would make every later result a lie.
"""

import re
from pathlib import Path

import numpy as np
import pytest

from detectors import (
    SPEECH_FRAMES,
    SPEECH_THRESHOLD,
    ModelDetector,
    RmsGate,
    frames_of,
    rms,
)
from evaluate import compare, evaluate, latency, run_clip
from features import (
    MIC_FRAME,
    N_MELS,
    SAMPLE_RATE,
    WINDOW_SAMPLES,
    RollingWindow,
    hz_to_mel,
    log_mel,
    mel_filterbank,
    mel_to_hz,
)

AUDIO_JS = Path(__file__).resolve().parents[1] / "web" / "src" / "lib" / "audio.js"


def tone(hz, seconds=1.0, amplitude=0.5, rate=SAMPLE_RATE):
    t = np.arange(int(rate * seconds)) / rate
    return (amplitude * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def silence(seconds=1.0, rate=SAMPLE_RATE):
    return np.zeros(int(rate * seconds), dtype=np.float32)


class TestMelScale:
    def test_round_trip(self):
        for hz in (20, 440, 1000, 4000, 8000):
            assert mel_to_hz(hz_to_mel(hz)) == pytest.approx(hz, rel=1e-6)

    def test_is_monotonic(self):
        mels = hz_to_mel(np.array([100, 500, 1000, 4000, 8000]))
        assert list(mels) == sorted(mels)

    def test_resolution_is_finer_at_the_bottom(self):
        """
        The point of the mel scale, and the reason speech features use it.
        A fixed step in mel covers a widening range of Hz as you climb, so
        evenly spaced filters end up narrow and numerous where the formants
        are, and broad up where they are not.
        """
        widths = [mel_to_hz(m + 100) - mel_to_hz(m) for m in (200, 800, 1600, 2400)]
        assert widths == sorted(widths), widths
        assert widths[-1] > widths[0] * 4


class TestFilterbank:
    def test_shape(self):
        assert mel_filterbank().shape == (N_MELS, 257)

    def test_every_filter_has_energy(self):
        """An empty filter is a dead feature the model can never use."""
        assert (mel_filterbank().sum(axis=1) > 0).all()

    def test_filters_are_non_negative(self):
        assert (mel_filterbank() >= 0).all()

    def test_centres_climb(self):
        bank = mel_filterbank()
        peaks = bank.argmax(axis=1)
        assert list(peaks) == sorted(peaks)


class TestLogMel:
    def test_shape_is_frames_by_mels(self):
        assert log_mel(tone(440)).shape == (98, N_MELS)

    def test_silence_is_a_finite_floor(self):
        """
        Unfloored, log(0) is negative infinity and every downstream mean or
        gradient becomes nan.
        """
        quiet = log_mel(silence())
        assert np.isfinite(quiet).all()
        assert np.allclose(quiet, quiet[0, 0])

    def test_a_low_tone_lands_in_a_low_bin(self):
        assert log_mel(tone(440)).mean(axis=0).argmax() < N_MELS // 3

    def test_a_high_tone_lands_in_a_high_bin(self):
        assert log_mel(tone(6000)).mean(axis=0).argmax() > N_MELS // 2

    def test_louder_is_larger(self):
        quiet = log_mel(tone(440, amplitude=0.05)).mean()
        loud = log_mel(tone(440, amplitude=0.5)).mean()
        assert loud > quiet

    def test_too_short_an_input_gives_no_frames(self):
        assert log_mel(np.zeros(100, dtype=np.float32)).shape == (0, N_MELS)


class TestRollingWindow:
    def test_first_push_already_produces_a_full_window(self):
        """
        A detector whose input shape changes for the first second of a session
        fails exactly when somebody is most likely to speak.
        """
        window = RollingWindow()
        assert len(window.push(np.ones(MIC_FRAME, dtype=np.float32))) == WINDOW_SAMPLES
        assert window.features().shape == (98, N_MELS)

    def test_old_audio_falls_off_the_back(self):
        window = RollingWindow()
        for _ in range(WINDOW_SAMPLES // MIC_FRAME + 4):
            window.push(np.ones(MIC_FRAME, dtype=np.float32))
        assert np.allclose(window.buffer, 1.0)

    def test_a_frame_longer_than_the_window_keeps_the_tail(self):
        window = RollingWindow()
        long = np.arange(WINDOW_SAMPLES * 2, dtype=np.float32)
        window.push(long)
        assert window.buffer[-1] == long[-1]

    def test_reset_clears_it(self):
        window = RollingWindow()
        window.push(np.ones(MIC_FRAME, dtype=np.float32))
        window.reset()
        assert not window.buffer.any()


class TestRmsGate:
    def test_constants_match_the_browser(self):
        """
        detectors.py claims to port audio.js. If the two drift, every number
        this harness reports about "the current baseline" is about something
        that is not shipping.
        """
        source = AUDIO_JS.read_text(encoding="utf-8")
        js_threshold = float(re.search(r"SPEECH_THRESHOLD = ([\d.]+)", source).group(1))
        js_frames = int(re.search(r"SPEECH_FRAMES = (\d+)", source).group(1))

        assert js_threshold == SPEECH_THRESHOLD
        assert js_frames == SPEECH_FRAMES

    def test_waits_for_the_run_before_firing(self):
        gate = RmsGate()
        loud = np.full(MIC_FRAME, 0.5, dtype=np.float32)

        assert [gate.push(loud) for _ in range(SPEECH_FRAMES)] == \
            [False] * (SPEECH_FRAMES - 1) + [True]

    def test_fires_once_per_utterance(self):
        gate = RmsGate()
        loud = np.full(MIC_FRAME, 0.5, dtype=np.float32)
        assert sum(gate.push(loud) for _ in range(30)) == 1

    def test_silence_never_fires(self):
        gate = RmsGate()
        quiet = np.zeros(MIC_FRAME, dtype=np.float32)
        assert not any(gate.push(quiet) for _ in range(50))

    def test_a_broken_run_starts_again(self):
        gate = RmsGate()
        loud = np.full(MIC_FRAME, 0.5, dtype=np.float32)
        quiet = np.zeros(MIC_FRAME, dtype=np.float32)

        gate.push(loud)
        gate.push(loud)
        gate.push(quiet)
        assert gate.push(loud) is False

    def test_rms_of_silence_is_zero(self):
        assert rms(np.zeros(MIC_FRAME, dtype=np.float32)) == 0.0

    def test_rms_of_empty_is_zero_not_a_crash(self):
        assert rms(np.array([], dtype=np.float32)) == 0.0


class TestModelDetector:
    def test_fires_when_the_score_clears_the_threshold(self):
        detector = ModelDetector(score=lambda _: 0.9, threshold=0.5)
        assert detector.push(np.zeros(MIC_FRAME, dtype=np.float32)) is True

    def test_stays_quiet_below_the_threshold(self):
        detector = ModelDetector(score=lambda _: 0.1, threshold=0.5)
        assert not any(
            detector.push(np.zeros(MIC_FRAME, dtype=np.float32)) for _ in range(20)
        )

    def test_cooldown_stops_one_utterance_reading_as_many(self):
        """
        The window holds a rolling second, so without a cooldown the wake word
        scores high on every callback while it sits in the buffer, and one
        wake up is reported about fifteen times.
        """
        detector = ModelDetector(score=lambda _: 1.0, threshold=0.5, cooldown_frames=16)
        fires = sum(detector.push(np.zeros(MIC_FRAME, dtype=np.float32)) for _ in range(17))
        assert fires == 1

    def test_it_can_fire_again_after_the_cooldown(self):
        detector = ModelDetector(score=lambda _: 1.0, threshold=0.5, cooldown_frames=4)
        fires = sum(detector.push(np.zeros(MIC_FRAME, dtype=np.float32)) for _ in range(11))
        assert fires == 3

    def test_the_model_sees_log_mel_of_the_right_shape(self):
        seen = []
        detector = ModelDetector(score=lambda window: seen.append(window.shape) or 0.0)
        detector.push(np.zeros(MIC_FRAME, dtype=np.float32))
        assert seen == [(98, N_MELS)]


class _Always:
    name = "always"

    def reset(self):
        pass

    def push(self, frame):
        return True


class _Never:
    name = "never"

    def reset(self):
        pass

    def push(self, frame):
        return False


class _Oracle:
    """Fires only on clips that were loud, standing in for a perfect model."""

    name = "oracle"

    def reset(self):
        pass

    def push(self, frame):
        return rms(frame) > 0.1


class TestFraming:
    def test_chops_into_mic_sized_frames(self):
        assert len(list(frames_of(np.zeros(MIC_FRAME * 5, dtype=np.float32)))) == 5

    def test_drops_a_ragged_tail(self):
        produced = list(frames_of(np.zeros(MIC_FRAME * 2 + 7, dtype=np.float32)))
        assert len(produced) == 2
        assert all(len(f) == MIC_FRAME for f in produced)


class TestEvaluate:
    def clips(self):
        return [
            (tone(440, 0.5), True),
            (tone(880, 0.5), True),
            (silence(0.5), False),
            (silence(0.5), False),
            (silence(0.5), False),
        ]

    def test_a_detector_that_always_fires_has_no_misses_and_total_false_alarms(self):
        result = evaluate(_Always(), self.clips())
        assert result["false_reject_rate"] == 0.0
        assert result["false_accept_rate"] == 1.0
        assert result["recall"] == 1.0

    def test_a_detector_that_never_fires_is_the_mirror_image(self):
        result = evaluate(_Never(), self.clips())
        assert result["false_reject_rate"] == 1.0
        assert result["false_accept_rate"] == 0.0

    def test_a_perfect_detector_scores_zero_on_both(self):
        result = evaluate(_Oracle(), self.clips())
        assert result["false_accept_rate"] == 0.0
        assert result["false_reject_rate"] == 0.0

    def test_counts_are_reported(self):
        result = evaluate(_Oracle(), self.clips())
        assert result["positives"] == 2
        assert result["negatives"] == 3

    def test_no_single_accuracy_number_is_offered(self):
        """
        False accepts and false rejects cost very different amounts here, so a
        blended figure would hide the one that matters.
        """
        assert "accuracy" not in evaluate(_Oracle(), self.clips())

    def test_empty_clip_set_does_not_divide_by_zero(self):
        result = evaluate(_Never(), [])
        assert result["false_accept_rate"] is None
        assert result["false_reject_rate"] is None

    def test_run_clip_reports_how_many_frames_it_saw(self):
        fired, seen = run_clip(_Never(), silence(1.0))
        assert fired is False
        assert seen == SAMPLE_RATE // MIC_FRAME


class TestLatency:
    def test_the_rms_gate_is_far_inside_the_budget(self):
        frame = np.full(MIC_FRAME, 0.2, dtype=np.float32)
        assert latency(RmsGate(), frame, repeats=50) < 10.0

    def test_feature_extraction_alone_is_measured(self):
        """
        The real cost of any model detector is the log mel on a rolling second
        of audio, before the model is even called. If that alone blows the
        10ms budget, no architecture choice rescues it.
        """
        frame = np.full(MIC_FRAME, 0.2, dtype=np.float32)
        cost = latency(ModelDetector(score=lambda _: 0.0), frame, repeats=50)
        assert cost < 10.0, f"features alone cost {cost}ms"


class TestCompare:
    def test_puts_detectors_on_the_same_clips(self):
        clips = [(tone(440, 0.5), True), (silence(0.5), False)]
        frame = np.full(MIC_FRAME, 0.2, dtype=np.float32)

        rows = compare([RmsGate(), _Never()], clips, frame=frame)
        assert [row["detector"] for row in rows] == ["rms gate", "never"]
        assert all("within_budget" in row for row in rows)
