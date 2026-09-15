"""
Scoring that needs no model.

This layer catches the failures that are both most important and cheapest to
detect: the agent answering from general knowledge instead of from the
meeting, or naming the wrong owner. The rubric judge handles tone, hedging and
everything else a keyword cannot see.

Running this alone gives a usable signal with no API spend, which matters when
the dataset is a few hundred questions and you want to check a prompt change
before paying for a full judged run.
"""

import re

_NOISE = re.compile(r"[^a-z0-9]+")


def normalise(text: str) -> str:
    """
    Lowercase and flatten punctuation, so "usage-based" and "Usage based"
    compare equal. Without this, half the misses are hyphens.
    """
    return _NOISE.sub(" ", (text or "").lower()).strip()


def _present(phrase, haystack) -> bool:
    return normalise(phrase) in haystack


def check(question, answer):
    """
    Score one answer. Returns (passed, reasons).

    A `mentions` entry that is a list passes if any of its options appear,
    which is how synonyms are expressed. A plain string must appear.
    """
    haystack = normalise(answer)
    reasons = []

    if not haystack:
        return False, ["empty answer"]

    for wanted in question.mentions:
        if isinstance(wanted, (list, tuple)):
            if not any(_present(option, haystack) for option in wanted):
                reasons.append(f"none of {list(wanted)}")
        elif not _present(wanted, haystack):
            reasons.append(f"missing {wanted!r}")

    for banned in question.avoids:
        if _present(banned, haystack):
            reasons.append(f"said {banned!r}")

    return not reasons, reasons


def tally(results):
    """
    Aggregate scored results into overall and per category accuracy.

    `results` is an iterable of (scenario, question, passed) triples.
    """
    overall = {"total": 0, "passed": 0}
    by_category = {}
    by_state = {True: {"total": 0, "passed": 0}, False: {"total": 0, "passed": 0}}

    for _, question, passed in results:
        overall["total"] += 1
        overall["passed"] += bool(passed)

        bucket = by_category.setdefault(question.category, {"total": 0, "passed": 0})
        bucket["total"] += 1
        bucket["passed"] += bool(passed)

        state = by_state[question.requires_state]
        state["total"] += 1
        state["passed"] += bool(passed)

    def rate(bucket):
        return round(bucket["passed"] / bucket["total"], 3) if bucket["total"] else None

    return {
        "total": overall["total"],
        "passed": overall["passed"],
        "accuracy": rate(overall),
        "by_category": {
            name: {**bucket, "accuracy": rate(bucket)}
            for name, bucket in sorted(by_category.items())
        },
        "needs_meeting_state": {**by_state[True], "accuracy": rate(by_state[True])},
        "general_knowledge": {**by_state[False], "accuracy": rate(by_state[False])},
    }
