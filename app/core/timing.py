"""
Stage timings for the latency benchmark.

Off unless MEETMIND_TIMING=1, so nothing is measured or sent in normal use.

Everything here is a duration on a monotonic clock, never a wall clock
timestamp. The benchmark client subtracts its own durations from these, and
two durations can be compared across machines while two timestamps cannot.
"""

import time


class TurnClock:
    """
    Measures the gap between the last input the server forwarded and the
    first audio Gemini sent back, once per turn.

    This only means "time to answer the question" when the client stops
    sending after it finishes speaking. A live browser streams silence
    continuously, so the last frame is always a few hundred ms old and the
    number collapses. bench/latency.py stops on purpose for that reason.
    """

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.last_input = None
        self.reported = False

    def input_forwarded(self) -> None:
        if self.enabled:
            self.last_input = time.monotonic()

    def end_turn(self) -> None:
        self.reported = False

    def first_audio_ms(self) -> float | None:
        """Milliseconds since the last forwarded input, once per turn."""
        if not self.enabled or self.reported or self.last_input is None:
            return None

        self.reported = True
        return round((time.monotonic() - self.last_input) * 1000, 2)
