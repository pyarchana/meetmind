"""
Rubric judge, plus the machinery for checking the judge against a human.

checks.py already catches answers that missed the facts. This layer is for
what a keyword cannot see: an answer that contains the right words while
hedging so hard it says nothing, or one that names the decision and then
undercuts it.

A judge nobody has audited is just a second opinion with a confident voice, so
`sample_for_review` pulls a random slice out for a person to label by hand and
`agreement` reports how often the two agreed. Raw agreement flatters a judge
that says pass to everything when most answers are passes, so Cohen's kappa is
reported next to it.
"""

import json
import random

from schema import CATEGORY_RUBRICS

MODEL = "gemini-2.5-flash"

PROMPT = """You are grading one answer from a meeting assistant.

The assistant sat in a meeting, tracked what was said, and was then asked a
question about it. Grade only whether the answer is correct and appropriate
for the rubric. Ignore style, length and tone.

How to grade this category:
{category_rubric}

What this specific question needs:
{question_rubric}

Question asked: {ask}

The assistant answered: {answer}

Reply with JSON only, no other text:
{{"verdict": "pass" or "fail", "reason": "one short sentence"}}"""


def build_prompt(question, answer: str) -> str:
    """Compose the category stance and the question specifics into one prompt."""
    return PROMPT.format(
        category_rubric=CATEGORY_RUBRICS[question.category],
        question_rubric=question.rubric,
        ask=question.ask,
        answer=answer or "(the assistant said nothing)",
    )


def parse_verdict(raw: str) -> dict:
    """
    Pull the verdict out of a model reply.

    Models wrap JSON in code fences often enough that failing on it would make
    the judge look worse than it is.
    """
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]

    try:
        parsed = json.loads(text.strip())
    except json.JSONDecodeError:
        return {"verdict": "error", "reason": f"unparseable judge reply: {raw[:80]!r}"}

    verdict = str(parsed.get("verdict", "")).lower()
    if verdict not in {"pass", "fail"}:
        return {"verdict": "error", "reason": f"unexpected verdict {verdict!r}"}

    return {"verdict": verdict, "reason": str(parsed.get("reason", ""))}


def judge_answer(question, answer, client, model=MODEL) -> dict:
    """Ask the judge about one answer. `client` is a google.genai Client."""
    response = client.models.generate_content(
        model=model,
        contents=build_prompt(question, answer),
    )
    return parse_verdict(response.text)


def sample_for_review(records, size=20, seed=0):
    """
    Pull a reproducible random slice out for a person to label.

    Seeded so the same run always produces the same sample, which matters when
    somebody has already spent an hour labelling it.
    """
    pool = list(records)
    if size >= len(pool):
        return pool
    return random.Random(seed).sample(pool, size)


def agreement(pairs):
    """
    Compare judge labels against human labels.

    `pairs` is an iterable of (judge_label, human_label). Returns raw
    agreement and Cohen's kappa.

    Kappa is the number to read. Raw agreement of 0.9 sounds good and means
    very little when 90 percent of answers pass anyway, because a judge that
    says pass to everything scores the same.
    """
    pairs = [(str(j).lower(), str(h).lower()) for j, h in pairs]
    total = len(pairs)
    if not total:
        return {"n": 0, "raw": None, "kappa": None}

    agreed = sum(1 for j, h in pairs if j == h)
    raw = agreed / total

    labels = {label for pair in pairs for label in pair}
    expected = sum(
        (sum(1 for j, _ in pairs if j == label) / total)
        * (sum(1 for _, h in pairs if h == label) / total)
        for label in labels
    )

    if expected == 1:
        # Both sides used a single label, so chance already explains all of it
        # and kappa is undefined rather than perfect.
        kappa = None
    else:
        kappa = (raw - expected) / (1 - expected)

    return {
        "n": total,
        "agreed": agreed,
        "raw": round(raw, 3),
        "kappa": None if kappa is None else round(kappa, 3),
    }
