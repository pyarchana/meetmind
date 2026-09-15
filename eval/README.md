# Answer quality eval

Answers issue #2. 104 questions across 11 meeting scenarios, so a change to the
agent instruction or the tool set can be shown to have helped rather than
assumed to have.

## Running it

```bash
cd app && uvicorn main:app --port 8000
```

```bash
python eval/run_eval.py --out eval/results/baseline.json --label "baseline"
python eval/run_eval.py --scenario pricing-and-launch     # one scenario
```

Each scenario opens its own session, replays the transcript one line at a
time, then asks its questions. Replaying line by line is the point: the thing
under test is whether the agent records at the right moments, not whether it
can summarise a transcript handed to it in one block.

```bash
cd eval && pytest . -v      # checks the harness, needs no key
```

## Why a scenario and not a flat list of QA pairs

"What did we decide about pricing" has no correct answer without a meeting
that decided something. So the unit is a scenario: a short transcript plus the
questions asked against it. Eleven transcripts carry 104 questions, which is
both less to write and a better test, because the same meeting state gets
probed from several angles.

## How questions are labelled

Every question carries a category, a `requires_state` flag, and two layers of
expectation.

| Category | What it tests |
|---|---|
| `decision_recall` | What the meeting actually agreed |
| `action_owner` | Who owns what, and by when |
| `open_questions` | What was deliberately left unresolved |
| `deferral` | Never discussed. The agent must say so |
| `general` | Answerable without the meeting at all |

`requires_state: true` means the answer is only reachable through
`get_meeting_state`. 87 of the 104 are. The remaining 17 are a control group:
if a change breaks ordinary answering, those catch it while the meeting
questions stay green.

**Rules for writing one:**

- The answer must be decidable from the transcript alone. If two readers could
  disagree about what the meeting decided, fix the transcript, not the rubric.
- `expect.mentions` is every phrase that must appear. A nested list means any
  one of those options counts, which is how synonyms and number formats are
  handled (`["twelfth", "12th"]`).
- `expect.avoids` is for the specific wrong answer, not for everything wrong.
  It exists to catch inversions: naming Whisper when the meeting chose
  Deepgram.
- The rubric states only what is specific to this question. Grading stance
  lives on the category in `schema.py`, so it stays consistent across all 104
  and cannot drift row by row.
- Every scenario needs at least one `deferral`. There are 18. They are the
  most valuable rows in the set, because the failure they catch is a confident
  invented answer, which is the one that gets acted on.

`schema.py` refuses to load a `requires_state` question with no
`expect.mentions`, because such a question can only be graded by the judge,
which makes it expensive and impossible to check offline.

## Two layers of scoring

**`checks.py`** is deterministic and free. It catches the answer that ignored
the meeting and replied from training data, and the answer that names the
wrong owner. Run the whole set through it for nothing, as often as you like.

**`judge.py`** is the rubric layer, for what a keyword cannot see: an answer
containing the right words while hedging so hard it commits to nothing.

Both report accuracy overall, per category, and split by `requires_state`.
That split is the interesting one. A high general score with a low meeting
score means the agent is a Gemini wrapper that is not using its tools.

## Auditing the judge

A judge nobody has checked is a second opinion with a confident voice.

```python
from judge import sample_for_review, agreement
```

`sample_for_review` pulls a seeded random 20 out for a person to label by
hand. The seed is fixed so the same sample comes back after somebody has
already spent an hour on it. `agreement` then reports raw agreement and
Cohen's kappa.

Read the kappa, not the raw number. When 90 percent of answers pass anyway, a
judge that says pass to everything scores 0.9 raw agreement and is worthless.
Kappa corrects for that, and there is a test pinning exactly that case.

## Trusting the harness

The runner is exercised against `mock_agent.py`, whose replies we choose, so
the scoring is checked against outcomes already known:

- answers built from the expectations score 100 percent
- a model that knows nothing scores 0 on everything needing meeting state
- a fluent invented overage rate is marked wrong

A harness that scores deliberately wrong answers well is worse than no
harness, so those three run in CI on every push.
