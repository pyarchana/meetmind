"""
Checks for the latency harness.

The end to end cases run against bench/mock_server.py, which answers after a
delay we choose. A harness that cannot recover a known delay cannot be trusted
to report an unknown one.
"""

import asyncio
import wave

import pytest

from latency import (
    FRAME_SAMPLES,
    SAMPLE_RATE,
    run_interrupt_trial,
    percentile,
    read_wav,
    run_trial,
    silence_frames,
    summarise,
    frames,
)
from mock_server import serve


class TestPercentile:
    def test_median_of_a_known_sample(self):
        assert percentile(list(range(1, 11)), 50) == 5

    def test_p95_reaches_the_tail(self):
        assert percentile(list(range(1, 11)), 95) == 10

    def test_single_value(self):
        assert percentile([42], 95) == 42

    def test_ignores_input_order(self):
        assert percentile([9, 1, 5, 3, 7], 50) == 5

    def test_empty_sample_is_an_error(self):
        with pytest.raises(ValueError):
            percentile([], 50)


class TestSummarise:
    def test_reports_percentiles_and_counts(self):
        runs = [{"first_audio_ms": float(v), "server_ms": float(v) - 20} for v in range(100, 200, 10)]
        summary = summarise(runs)

        assert summary["runs"] == 10
        assert summary["failed"] == 0
        assert summary["client_ms"]["min"] == 100
        assert summary["client_ms"]["max"] == 190

    def test_counts_runs_that_produced_no_audio(self):
        runs = [{"first_audio_ms": 100.0}, {"first_audio_ms": None}, {"first_audio_ms": 200.0}]
        summary = summarise(runs)

        assert summary["runs"] == 3
        assert summary["failed"] == 1
        assert summary["client_ms"]["mean"] == 150.0

    def test_survives_a_run_where_nothing_came_back(self):
        summary = summarise([{"first_audio_ms": None}])
        assert summary["failed"] == 1
        assert "client_ms" not in summary

    def test_transport_is_the_gap_between_client_and_server(self):
        runs = [{"first_audio_ms": 500.0, "server_ms": 430.0}] * 5
        assert summarise(runs)["transport_ms"]["p50"] == 70.0

    def test_omits_server_stats_when_timing_was_off(self):
        summary = summarise([{"first_audio_ms": 500.0, "server_ms": None}] * 3)
        assert "server_ms" not in summary
        assert "transport_ms" not in summary


class TestFraming:
    def test_pads_a_short_final_frame(self):
        produced = list(frames(b"\x01\x02" * 10))
        assert len(produced) == 1
        assert len(produced[0]) == FRAME_SAMPLES * 2

    def test_splits_on_browser_frame_size(self):
        pcm = b"\x00" * (FRAME_SAMPLES * 2 * 3)
        assert len(list(frames(pcm))) == 3

    def test_one_second_of_silence_is_about_four_frames(self):
        assert len(silence_frames(1.0)) == 4


class TestAgainstMockServer:
    @pytest.mark.asyncio
    async def test_recovers_a_known_delay(self):
        """A 400ms mock should be measured as roughly 400ms."""
        delay = 0.4
        # quiet_gap well under the delay, otherwise the mock's own detection
        # window is what gets measured and the test passes for free.
        async with serve(8781, delay, quiet_gap=0.1):
            trial = await run_trial(
                "ws://127.0.0.1:8781", "text", None, "how long did that take?", 10.0
            )

        measured = trial["first_audio_ms"]
        assert measured is not None, "harness saw no audio at all"
        assert 380 < measured < 480, measured

    @pytest.mark.asyncio
    async def test_server_and_client_numbers_agree(self):
        async with serve(8782, 0.3, quiet_gap=0.1):
            trial = await run_trial("ws://127.0.0.1:8782", "text", None, "q", 10.0)

        assert trial["server_ms"] is not None
        # The client number includes the websocket hop, so it can only be larger.
        assert trial["first_audio_ms"] >= trial["server_ms"] - 5

    @pytest.mark.asyncio
    async def test_distinguishes_two_different_delays(self):
        """The harness has to actually track the delay, not report a constant."""
        async with serve(8783, 0.2, quiet_gap=0.05):
            fast = await run_trial("ws://127.0.0.1:8783", "text", None, "q", 10.0)
        async with serve(8784, 0.8, quiet_gap=0.05):
            slow = await run_trial("ws://127.0.0.1:8784", "text", None, "q", 10.0)

        gap = slow["first_audio_ms"] - fast["first_audio_ms"]
        assert 500 < gap < 700, gap

    @pytest.mark.asyncio
    async def test_reports_no_audio_when_the_server_never_answers(self):
        async def silent(socket):
            await asyncio.sleep(5)

        import websockets
        async with websockets.serve(silent, "127.0.0.1", 8785):
            trial = await run_trial("ws://127.0.0.1:8785", "text", None, "q", 1.0)

        assert trial["first_audio_ms"] is None


def _write_wav(path, seconds=0.5, rate=SAMPLE_RATE, channels=1, width=2):
    """A WAV of loud-ish alternating samples, enough to stand in for speech."""
    count = int(rate * seconds)
    pcm = bytes(bytearray([0x00, 0x40] * count))
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(width)
        handle.setframerate(rate)
        handle.writeframes(pcm)
    return path


class TestReadWav:
    def test_reads_a_well_formed_file(self, tmp_path):
        pcm = read_wav(_write_wav(tmp_path / "ok.wav", seconds=0.25))
        assert len(pcm) == int(SAMPLE_RATE * 0.25) * 2

    def test_refuses_the_wrong_sample_rate(self, tmp_path):
        path = _write_wav(tmp_path / "wrong.wav", rate=44100)
        with pytest.raises(SystemExit, match="44100"):
            read_wav(path)

    def test_refuses_stereo(self, tmp_path):
        path = _write_wav(tmp_path / "stereo.wav", channels=2)
        with pytest.raises(SystemExit, match="mono"):
            read_wav(path)


class TestInterruptMode:
    @pytest.mark.asyncio
    async def test_times_how_long_the_server_keeps_talking(self, tmp_path):
        """The mock gives up 300ms after being talked over, so we should read that."""
        pcm = read_wav(_write_wav(tmp_path / "speech.wav", seconds=0.4))

        async with serve(8791, delay=0.2, interrupt_delay=0.3):
            trial = await run_interrupt_trial(
                "ws://127.0.0.1:8791", pcm, "say something long", 15.0, speak_after=0.5
            )

        assert trial["first_audio_ms"] is not None, "agent never started talking"
        ack = trial["interrupt_ack_ms"]
        assert ack is not None, "never saw the interrupted signal"
        assert 250 < ack < 550, ack

    @pytest.mark.asyncio
    async def test_separates_a_slow_reaction_from_a_fast_one(self, tmp_path):
        pcm = read_wav(_write_wav(tmp_path / "speech.wav", seconds=0.4))

        async with serve(8792, delay=0.2, interrupt_delay=0.2):
            fast = await run_interrupt_trial(
                "ws://127.0.0.1:8792", pcm, "q", 15.0, speak_after=0.5)
        async with serve(8793, delay=0.2, interrupt_delay=1.2):
            slow = await run_interrupt_trial(
                "ws://127.0.0.1:8793", pcm, "q", 15.0, speak_after=0.5)

        gap = slow["interrupt_ack_ms"] - fast["interrupt_ack_ms"]
        assert 800 < gap < 1200, gap

    def test_summary_reports_the_interrupt_percentiles(self):
        runs = [{"first_audio_ms": 400.0, "interrupt_ack_ms": float(v)}
                for v in range(200, 300, 10)]
        summary = summarise(runs)
        assert summary["interrupt_ack_ms"]["p50"] == 240
        assert summary["interrupt_ack_ms"]["p95"] == 290
