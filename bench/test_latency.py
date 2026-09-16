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
    FRAME_SECONDS,
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

    def test_silence_is_measured_in_seconds_not_frames(self):
        # Derived from the frame size rather than hardcoded, so changing the
        # buffer moves this with it instead of breaking it.
        # The property that matters is the duration, not the count. Frame
        # counts do not scale linearly with seconds, because 2.0 / 0.064 is
        # 31.25 and rounds to 31 rather than to twice 16.
        for seconds in (0.5, 1.0, 2.0, 3.0):
            produced = len(silence_frames(seconds)) * FRAME_SECONDS
            assert abs(produced - seconds) <= FRAME_SECONDS / 2, (seconds, produced)

    def test_a_short_request_still_yields_a_frame(self):
        assert len(silence_frames(0.001)) == 1


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


class TestPacketLoss:
    def test_zero_percent_drops_nothing(self):
        from latency import lossy
        outgoing = [b"x"] * 200
        assert sum(1 for _, lost in lossy(outgoing, 0.0) if lost) == 0

    def test_one_hundred_percent_drops_everything(self):
        from latency import lossy
        outgoing = [b"x"] * 200
        assert all(lost for _, lost in lossy(outgoing, 100.0))

    def test_the_rate_is_roughly_what_was_asked_for(self):
        from latency import lossy
        outgoing = [b"x"] * 4000
        lost = sum(1 for _, dropped in lossy(outgoing, 20.0) if dropped)
        assert 0.17 < lost / len(outgoing) < 0.23, lost / len(outgoing)

    def test_the_same_seed_drops_the_same_frames(self):
        """
        Otherwise a loss experiment measures the seed, not the loss rate, and
        two runs at 20 percent are not comparable.
        """
        from latency import lossy
        outgoing = [bytes([i % 256]) for i in range(500)]
        first = [lost for _, lost in lossy(outgoing, 30.0, seed=7)]
        second = [lost for _, lost in lossy(outgoing, 30.0, seed=7)]
        assert first == second

    def test_a_different_seed_drops_a_different_set(self):
        from latency import lossy
        outgoing = [bytes([i % 256]) for i in range(500)]
        first = [lost for _, lost in lossy(outgoing, 30.0, seed=1)]
        second = [lost for _, lost in lossy(outgoing, 30.0, seed=2)]
        assert first != second

    def test_frames_pass_through_unchanged(self):
        from latency import lossy
        outgoing = [b"one", b"two", b"three"]
        assert [frame for frame, _ in lossy(outgoing, 50.0)] == outgoing

    @pytest.mark.asyncio
    async def test_a_lossy_run_reports_what_it_dropped(self, tmp_path):
        from latency import read_wav, run_trial

        pcm = read_wav(_write_wav(tmp_path / "speech.wav", seconds=1.5))
        # A generous quiet gap, because dropped frames stretch the spacing
        # between the ones that survive.
        async with serve(8795, delay=0.2, quiet_gap=1.5):
            trial = await run_trial("ws://127.0.0.1:8795", "audio", pcm, "", 20.0,
                                    drop_pct=50.0, seed=3)

        assert trial["frames_dropped"] > 0
        assert trial["frames_sent"] + trial["frames_dropped"] > 5
