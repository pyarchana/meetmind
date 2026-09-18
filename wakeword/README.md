# Wake word detector

Issue #3. Nothing is trained yet. What exists is the machinery a trained model
needs: features, a detector interface, the current heuristic as a baseline,
and scoring that reports the right numbers.

```bash
cd wakeword && pytest . -v
```

## Why the harness came first

A model is easy to produce and hard to trust. If the scoring is wrong, every
number that follows is wrong too and nothing about the model will reveal it.
So the evaluation is built and tested first, against detectors whose behaviour
is known in advance: one that always fires, one that never does, and one that
is perfect. Their false accept and false reject rates come out at exactly 1, 0
and 0 respectively, which is what makes later results worth reading.

## The two numbers, and why there is no third

| | Meaning | Cost when it happens |
|---|---|---|
| False reject | The wake word was said, nothing woke up | Annoying. They say it again |
| False accept | It woke up during a sentence not addressed to it | Embarrassing, in front of other people |

`evaluate()` deliberately does not return a blended accuracy. These two cost
very different amounts, and a single figure would let a bad false accept rate
hide behind a good false reject rate. There is a test asserting no accuracy
key is present.

## What the baseline is, and why it is the right one

The issue says compare against the RMS threshold, not against zero. Worth
being exact about what that comparison means.

The RMS gate is not a wake word detector. It is a speech detector, ported from
`web/src/lib/audio.js`, and it fires on any speech regardless of the words. So
on clips of speech that are not the wake word, its false accept rate should
come out near total.

That is the point. It is what ships today, so it is the bar. A model that
cannot clear it is not earning its complexity, and a model measured against
nothing at all would look impressive while meaning nothing.

`test_constants_match_the_browser` parses `audio.js` and fails if the
threshold or frame count drift apart, so the baseline cannot quietly stop
being what actually ships.

## Latency, measured

The issue asks for under 10ms per frame. Per mic callback, on this machine:

| | Cost |
|---|---|
| RMS gate (today) | 0.011 ms |
| Log mel over a rolling 1s window | 1.91 ms |
| Headroom left for the model | 8.16 ms |

The mic fires every 64ms, so there is plenty of wall clock. The 10ms target is
the tighter constraint and the features eat a fifth of it before any model
runs.

**These are Python numbers and the model will not run in Python.** See below.

## Two things to settle before training

**Where inference runs.** The reason for a wake word is to stop streaming
every frame of a meeting. That saving only exists if the gate runs in the
browser, before the audio is sent. Running it server side means the audio has
already been uploaded and nothing is saved. So the model has to run in JS or
WASM, and the 1.91ms measured here is a Python number that says nothing about
the browser. Training in Python is fine; the latency claim has to be remade in
the browser before it means anything.

**Training data.** There is none. A wake word model needs recordings of the
wake word and a lot of speech that is not it. Options, in rough order of
effort:

- Google Speech Commands, free, about 2GB, the standard dataset for this task.
  Has short spoken words suitable for both classes.
- Record the wake word yourself, a few hundred times, in varying conditions.
  Tedious but it matches your actual microphone and voice.
- Both. Pretrain on the public set, fine tune on your own recordings.

Nothing in this directory downloads anything or picks a framework, because
both choices belong to whoever is paying for the disk and the time.

## The constraint that bit issue #1 still applies

Gemini runs its own voice activity detection server side and uses trailing
silence to decide a turn has ended. A gate that cuts the upstream the instant
local speech stops will break turn taking. Whatever wakes the stream up also
needs a hangover window that keeps sending for roughly a second afterwards,
and that window should be measured rather than guessed.
