"""
Checks for the offline audio measurements.

These numbers get quoted, so the things producing them are pinned against
theory where theory exists, rather than against whatever they happened to
print the first time.
"""

import array
import math

import pytest

from audio_tradeoffs import (
    SAMPLE_RATE,
    json_frame,
    binary_frame,
    measure_encoding,
    measure_frame_sizes,
    measure_quantisation,
    snr_db,
    to_8bit,
    voice_like,
)


def _truncating_8bit(samples):
    """The tempting shift version, kept here only to show it is worse."""
    out = array.array("h", bytes(len(samples) * 2))
    for i, value in enumerate(samples):
        out[i] = max(-128, min(127, value >> 8)) * 256
    return out


class TestVoiceLike:
    def test_is_deterministic(self):
        assert voice_like(500).tobytes() == voice_like(500).tobytes()

    def test_stays_inside_int16(self):
        assert all(-32768 <= s <= 32767 for s in voice_like(4000))

    def test_is_not_silence(self):
        samples = voice_like(4000)
        assert max(abs(s) for s in samples) > 1000

    def test_spends_most_of_its_time_well_below_peak(self):
        """
        The whole reason for not using a pure tone. Quantisation error hurts
        quiet passages most, so a signal pinned near full scale would report a
        flattering SNR.
        """
        samples = voice_like(SAMPLE_RATE)
        peak = max(abs(s) for s in samples)
        rms = math.sqrt(sum(float(s) * s for s in samples) / len(samples))
        assert rms < peak * 0.5


class TestSnr:
    def test_identical_signals_have_no_noise(self):
        samples = voice_like(1000)
        assert snr_db(samples, samples) == math.inf

    def test_a_known_ratio(self):
        """Signal power 100x noise power is 20dB by definition."""
        signal = array.array("h", [100] * 1000)
        noisy = array.array("h", [110] * 1000)   # error of 10, so 100:1 in power
        assert snr_db(signal, noisy) == pytest.approx(20.0, abs=0.01)

    def test_length_mismatch_is_an_error(self):
        with pytest.raises(ValueError):
            snr_db(array.array("h", [1, 2]), array.array("h", [1]))


class TestQuantisation:
    def test_error_never_exceeds_half_a_step(self):
        samples = voice_like(4000)
        restored = to_8bit(samples)
        assert max(abs(a - b) for a, b in zip(samples, restored)) <= 128

    def test_rounding_beats_truncating_by_about_six_db(self):
        """
        Truncation biases every sample toward zero. The gap is close to 6dB
        and that is why to_8bit rounds.
        """
        samples = voice_like(SAMPLE_RATE)
        rounded = snr_db(samples, to_8bit(samples))
        truncated = snr_db(samples, _truncating_8bit(samples))

        assert 4 < rounded - truncated < 8, (rounded, truncated)

    def test_reported_snr_is_in_a_sane_range(self):
        """
        Full scale linear 8 bit tops out near 50dB. This signal sits well
        below full scale, so anything near or above 50 would mean the
        measurement is wrong, not that 8 bit is good.
        """
        snr = measure_quantisation(seconds=0.5)["snr_db"]
        assert 20 < snr < 45, snr

    def test_halving_the_depth_halves_the_bytes(self):
        result = measure_quantisation(seconds=0.5)
        assert result["bytes_8bit"] * 2 == result["bytes_16bit"]


class TestEncoding:
    def test_base64_costs_about_a_third(self):
        pcm = voice_like(4096).tobytes()
        ratio = len(json_frame(pcm)) / len(binary_frame(pcm))
        assert 1.30 < ratio < 1.40, ratio

    def test_binary_frame_is_the_bytes_themselves(self):
        pcm = voice_like(256).tobytes()
        assert binary_frame(pcm) == pcm

    def test_reported_throughput_matches_the_frame_rate(self):
        result = measure_encoding(frame_samples=4096, repeats=20)
        expected = result["json_base64_bytes"] * (SAMPLE_RATE / 4096) / 1024
        assert result["json_kb_per_sec"] == pytest.approx(expected, abs=0.1)

    def test_raw_audio_is_two_bytes_a_sample(self):
        result = measure_encoding(frame_samples=1024, repeats=20)
        assert result["raw_bytes"] == 1024 * 2


class TestFrameSizes:
    def test_smaller_frames_lower_the_barge_in_floor(self):
        rows = {r["frame_samples"]: r for r in measure_frame_sizes(repeats=20)}
        assert rows[1024]["vad_floor_ms"] < rows[4096]["vad_floor_ms"]
        assert rows[1024]["vad_floor_ms"] == 192.0
        assert rows[4096]["vad_floor_ms"] == 768.0

    def test_smaller_frames_cost_more_envelope_overhead(self):
        """
        Four times the frames means four times the JSON wrappers. This is the
        price of the lower latency floor, and it is small.
        """
        rows = {r["frame_samples"]: r for r in measure_frame_sizes(repeats=20)}
        assert rows[1024]["envelope_bytes_per_sec"] > rows[4096]["envelope_bytes_per_sec"]

    def test_the_bandwidth_difference_is_marginal(self):
        """
        The headline: dropping to 1024 samples costs well under a kilobyte a
        second, which is what makes it the cheap latency win.
        """
        rows = {r["frame_samples"]: r for r in measure_frame_sizes(repeats=20)}
        assert rows[1024]["json_kb_per_sec"] - rows[4096]["json_kb_per_sec"] < 1.0
