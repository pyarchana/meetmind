# Latency benchmark

Answers issue #1. Measures how long MeetMind takes to respond, so a change can
be shown to have helped rather than assumed to have.

## Running it

Start the server with timings on:

```bash
cd app
MEETMIND_TIMING=1 uvicorn main:app --port 8000
```

Then pick a mode:

```bash
# Typed question to first audio byte. Deterministic, needs no audio file.
python bench/latency.py --runs 20 --label "local, wifi" --out bench/results/local.json

# Same measurement driven by real speech.
python bench/latency.py --mode audio --wav question.wav --runs 20

# Talk over the agent and time how long it keeps going.
python bench/latency.py --mode interrupt --wav speech.wav --runs 10
```

`--wav` takes 16kHz mono 16 bit. The harness refuses anything else rather than
resampling behind your back. Record a real question: a tone gets a confused
answer and a latency number that means nothing.

## What the numbers mean

| Field | Meaning |
|---|---|
| `client_ms` | Last input sent to first audio byte received. The headline. |
| `server_ms` | Same gap measured inside the server. Needs `MEETMIND_TIMING=1`. |
| `transport_ms` | `client_ms` minus `server_ms`, so websocket plus scheduling. |
| `interrupt_ack_ms` | Talking over the agent to the server admitting it. |

Both halves of `transport_ms` are durations, not timestamps, so subtracting
them is safe even when the client and server are on different machines with
unsynchronised clocks.

## Things that will bite you

**The benchmark stops sending, a browser does not.** Gemini runs its own voice
activity detection and uses trailing silence to close a turn, so the browser
streams continuously. That makes "the last frame we sent" useless as a start
point, because it is always a few hundred ms old. The harness sends the
question plus a fixed tail of silence and then stops, which gives one well
defined moment to measure from. It is not what a real client does, and that is
the point.

**256ms of quantisation.** The mic uses a 4096 sample buffer at 16kHz, so
frames arrive every 256ms and no client side measurement is finer than that.
Report it as measurement error or shrink the buffer for the experiment and
note what it costs in CPU.

**Interrupt mode measures the server half only.** The browser cuts its own
playback as soon as local VAD fires, which never touches the network. That
half is fixed by the constants in `web/src/lib/audio.js` and pinned by the web
tests: three frames of speech to trigger, so up to 768ms before playback stops.
On most networks that will dominate the server round trip, so measure both
before optimising either.

## Checking the harness itself

`mock_server.py` answers after a delay you choose, so the harness can be
checked against a number that is already known.

```bash
cd bench && pytest . -v
```

Those tests recover a known 400ms delay, separate 200ms from 800ms, and
recover a known interruption reaction. A harness that cannot do that cannot be
trusted to report an unknown number.
