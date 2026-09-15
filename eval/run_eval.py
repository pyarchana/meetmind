"""
Answer quality eval for MeetMind.

For each scenario the transcript is replayed one line at a time, the way a
meeting actually arrives, then the questions are asked against whatever state
the agent built up. Pasting the whole transcript in one message would be
faster and cheaper, and would test nothing: the thing under test is whether
the agent records at the right moments, not whether it can summarise a wall
of text it was handed.

Scoring runs in two layers:
    checks.py   deterministic, free, catches the answer that ignored the
                meeting and answered from training data
    judge.py    rubric scoring for everything a keyword cannot see

Usage:
    python eval/run_eval.py --url ws://127.0.0.1:8000 --out eval/results/run.json
    python eval/run_eval.py --scenario pricing-and-launch --no-judge
"""

import argparse
import asyncio
import json
import sys
import uuid
from pathlib import Path

import websockets

sys.path.insert(0, str(Path(__file__).parent))

from checks import check, tally           # noqa: E402
from schema import load_dataset           # noqa: E402

SETTLE_SECONDS = 0.4


async def _turn(socket, text, timeout):
    """
    Send one message and collect the agent's reply for that turn.

    Returns the last transcript seen before the turn closed. Transcripts
    stream incrementally and the server already forwards only the longest, so
    the last one is the complete answer.
    """
    await socket.send(json.dumps({"type": "text", "data": text}))

    answer = ""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout

    while True:
        remaining = deadline - loop.time()
        if remaining <= 0:
            return answer

        try:
            raw = await asyncio.wait_for(socket.recv(), timeout=remaining)
        except (asyncio.TimeoutError, websockets.ConnectionClosed):
            return answer

        message = json.loads(raw)
        if message["type"] == "transcript_agent":
            answer = message["data"]
        elif message["type"] == "text":
            answer += message["data"]
        elif message["type"] == "turn_complete":
            return answer
        elif message["type"] == "error":
            return f"[server error] {message['data']}"


async def run_scenario(url, scenario, timeout):
    """Replay the meeting, then ask the questions. Returns a list of answers."""
    session = f"eval-{scenario.id}-{uuid.uuid4().hex[:6]}"
    answers = []

    async with websockets.connect(f"{url}/ws/eval/{session}", max_size=None) as socket:
        for line in scenario.transcript:
            await _turn(socket, f"{line['speaker']}: {line['text']}", timeout)
            await asyncio.sleep(SETTLE_SECONDS)

        for question in scenario.questions:
            answers.append({
                "scenario": scenario.id,
                "question": question.id,
                "category": question.category,
                "requires_state": question.requires_state,
                "ask": question.ask,
                "answer": await _turn(socket, question.ask, timeout),
            })

    return answers


def score(scenarios, answers):
    """Attach deterministic pass or fail to each answer and aggregate."""
    lookup = {
        (s.id, q.id): (s, q) for s in scenarios for q in s.questions
    }

    results = []
    for record in answers:
        scenario, question = lookup[(record["scenario"], record["question"])]
        passed, reasons = check(question, record["answer"])
        results.append((scenario, question, passed))
        record["passed"] = passed
        record["reasons"] = reasons

    return tally(results)


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://127.0.0.1:8000")
    parser.add_argument("--scenario", action="append",
                        help="run only these scenario ids, repeatable")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    scenarios = load_dataset()
    if args.scenario:
        wanted = set(args.scenario)
        scenarios = [s for s in scenarios if s.id in wanted]
        missing = wanted - {s.id for s in scenarios}
        if missing:
            raise SystemExit(f"no such scenario: {sorted(missing)}")

    answers = []
    for scenario in scenarios:
        print(f"{scenario.id} ... ", end="", flush=True)
        produced = await run_scenario(args.url, scenario, args.timeout)
        answers.extend(produced)
        hit = sum(1 for a in produced if check(
            next(q for q in scenario.questions if q.id == a["question"]), a["answer"]
        )[0])
        print(f"{hit}/{len(produced)}")

    summary = score(scenarios, answers)

    print()
    print(json.dumps(summary, indent=2))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps({"label": args.label, "summary": summary, "answers": answers},
                       indent=2),
            encoding="utf-8",
        )
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
