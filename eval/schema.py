"""
Dataset format for the answer quality eval.

A scenario is a short meeting transcript plus the questions asked against it.
Questions cannot be graded on their own: "what did we decide about pricing"
only has a right answer relative to a meeting that decided something.

Loading validates as it goes, so a malformed scenario fails at load rather
than halfway through a paid eval run.
"""

from dataclasses import dataclass, field
from pathlib import Path

import yaml

DATASET_DIR = Path(__file__).parent / "dataset"

CATEGORIES = {
    "decision_recall",    # what did we agree
    "action_owner",       # who owns what
    "open_questions",     # what is still unresolved
    "deferral",           # not discussed, the agent should say so
    "general",            # answerable without the meeting
}

# How to grade, per category. The judge prepends this to the question rubric,
# so each question only has to carry what is specific to it. Repeating the
# stance in all hundred odd rows would be noise that drifts out of sync.
CATEGORY_RUBRICS = {
    "decision_recall": (
        "Must state the decision the meeting actually reached. Sensible advice "
        "that was not the decision is a failure. Listing the options without "
        "committing is a failure, because the meeting did commit."
    ),
    "action_owner": (
        "Must name the right person or deadline. Naming an additional person "
        "as an owner is a failure even when the right one is also named."
    ),
    "open_questions": (
        "Must surface the item that was left unresolved. Presenting a settled "
        "decision as open, or missing the open item entirely, is a failure."
    ),
    "deferral": (
        "Must make clear this was not decided or not known. Any specific "
        "figure, date, name or policy offered as fact is a hallucination and "
        "the worst failure in this set, because it reads as authoritative and "
        "would be acted on."
    ),
    "general": (
        "Ordinary correct explanation. Needs no meeting context. These exist "
        "to catch a regression where adding meeting state broke plain answers."
    ),
}


@dataclass(frozen=True)
class Question:
    id: str
    category: str
    requires_state: bool
    ask: str
    rubric: str
    mentions: tuple = ()
    avoids: tuple = ()


@dataclass(frozen=True)
class Scenario:
    id: str
    title: str
    transcript: tuple
    questions: tuple = field(default=())

    @property
    def script(self) -> str:
        return "\n".join(f"{line['speaker']}: {line['text']}" for line in self.transcript)


def _question(raw, scenario_id):
    where = f"{scenario_id}.{raw.get('id', '?')}"

    missing = {"id", "category", "requires_state", "ask", "rubric"} - set(raw)
    if missing:
        raise ValueError(f"{where}: missing {sorted(missing)}")

    if raw["category"] not in CATEGORIES:
        raise ValueError(f"{where}: unknown category {raw['category']!r}")

    expect = raw.get("expect") or {}
    mentions = tuple(expect.get("mentions", ()))
    avoids = tuple(expect.get("avoids", ()))

    # A state question with nothing concrete to look for can only be graded by
    # the judge, which makes it expensive and unfalsifiable offline.
    if raw["requires_state"] and not mentions:
        raise ValueError(f"{where}: a requires_state question needs expect.mentions")

    return Question(
        id=raw["id"],
        category=raw["category"],
        requires_state=bool(raw["requires_state"]),
        ask=raw["ask"],
        rubric=raw["rubric"].strip(),
        mentions=mentions,
        avoids=avoids,
    )


def load_scenario(path) -> Scenario:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))

    for key in ("id", "title", "transcript", "questions"):
        if key not in raw:
            raise ValueError(f"{path}: missing {key}")

    for line in raw["transcript"]:
        if not {"speaker", "text"} <= set(line):
            raise ValueError(f"{raw['id']}: transcript lines need speaker and text")

    questions = tuple(_question(q, raw["id"]) for q in raw["questions"])
    if not questions:
        raise ValueError(f"{raw['id']}: no questions")

    return Scenario(
        id=raw["id"],
        title=raw["title"],
        transcript=tuple(raw["transcript"]),
        questions=questions,
    )


def load_dataset(directory=DATASET_DIR):
    """Every scenario, with ids checked for collisions across files."""
    scenarios = [load_scenario(p) for p in sorted(Path(directory).glob("*.yaml"))]

    seen = set()
    for scenario in scenarios:
        if scenario.id in seen:
            raise ValueError(f"duplicate scenario id {scenario.id!r}")
        seen.add(scenario.id)

        for question in scenario.questions:
            key = f"{scenario.id}.{question.id}"
            if key in seen:
                raise ValueError(f"duplicate question id {key!r}")
            seen.add(key)

    return scenarios


def all_questions(scenarios):
    for scenario in scenarios:
        for question in scenario.questions:
            yield scenario, question
