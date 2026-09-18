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

**64ms of quantisation.** The mic uses a 1024 sample buffer at 16kHz, so
frames arrive every 64ms and no client side measurement is finer than that.
Report it as measurement error or shrink the buffer for the experiment and
note what it costs in CPU.

**Interrupt mode measures the server half only.** The browser cuts its own
playback as soon as local VAD fires, which never touches the network. That
half is fixed by the constants in `web/src/lib/audio.js` and pinned by the web
tests: three frames of speech to trigger, so up to 192ms before playback stops.
On most networks that will dominate the server round trip, so measure both
before optimising either.


## Audio transport tradeoffs

Answers issue #4. Byte counts and CPU are properties of the encoding rather
than of Gemini, so they are measured offline and the results are committed in
`results/audio_tradeoffs.json`.

```bash
python bench/audio_tradeoffs.py --out bench/results/audio_tradeoffs.json
```

### Encoding, measured at a 256ms frame

| | Per frame | Per second |
|---|---|---|
| JSON plus base64 (today) | 10953 B | 41.8 KB/s |
| Binary websocket frame | 8192 B | 31.2 KB/s |

Base64 costs 33.7 percent, and encoding costs 68 microseconds per 256ms frame.

That is 0.03 percent of one core. Binary frames are tidier, but 10 KB/s and
rounding error on CPU will not show up in any measurement a user can feel, so
this is not worth a protocol change on its own.

### Frame size

| Samples | Frame | Barge in floor | Wire | CPU per second |
|---|---|---|---|---|
| 1024 (today) | 64 ms | **192 ms** | 42.1 KB/s | 330 us |
| 2048 | 128 ms | 384 ms | 41.9 KB/s | 284 us |
| 4096 (was) | 256 ms | 768 ms | 41.8 KB/s | 271 us |
| 8192 | 512 ms | 1536 ms | 41.7 KB/s | 263 us |

This was the finding worth acting on, and it has been applied. Issue #1
pinned barge in at up to 768 ms, three frames of confirmation at 256 ms each,
and showed it is probably larger than the server round trip. The mic now uses
1024 sample frames, so that floor is 192 ms.

What it costs is sensitivity. A transient only has to survive 192 ms to cut the
agent off now, where it used to need 768 ms, so coughs and door slams are more
likely to register as speech. If that shows up in real meetings, raise
SPEECH_FRAMES rather than putting the buffer back: five frames gives a 320 ms
window and still beats the old number by more than two to one.

It costs 0.3 KB/s and about 60 extra microseconds of CPU per second of audio.
For a four fold improvement in the slowest part of the interaction, that is
close to free.

The wire cost barely moves because the payload is the same audio either way.
Only the JSON envelope multiplies, and the envelope is tiny next to 8 KB of
base64.

### Quantisation

Sixteen bit to eight bit halves the bytes and costs 33.9 dB SNR on a speech
shaped signal.

Two caveats. This is linear eight bit; u-law spends its levels where speech
actually sits and does considerably better at the same depth. And SNR is not
accuracy: whether Gemini transcribes any worse needs a real run, which is what
`--drop-pct` and a live key are for.

`to_8bit` rounds rather than truncating. Truncating is a shift and looks
neater, but it biases every sample toward zero and costs about another 6 dB,
which would have made eight bit look worse than it is. There is a test pinning
that gap.

### Packet loss

`--drop-pct` drops a seeded percentage of upstream frames.

```bash
python bench/latency.py --mode audio --wav speech.wav --drop-pct 20 --runs 10
```

Seeded on purpose: without it, two runs at 20 percent drop different frames
and are not comparable, so the experiment measures the seed. Dropped frames
still occupy their slot in real time, otherwise the stream quietly speeds up
in proportion to the loss rate.

What this cannot tell you offline is the thing worth knowing, which is what
loss does to transcription accuracy. That needs a live key.

## Checking the harness itself

`mock_server.py` answers after a delay you choose, so the harness can be
checked against a number that is already known.

```bash
cd bench && pytest . -v
```

Those tests recover a known 400ms delay, separate 200ms from 800ms, and
recover a known interruption reaction. A harness that cannot do that cannot be
trusted to report an unknown number.
