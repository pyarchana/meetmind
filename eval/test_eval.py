"""
Checks for the eval harness and the dataset.

The dataset tests enforce the shape issue #2 asked for, so the file cannot
quietly drift below it. The runner tests drive real answers through the real
scoring path against a mock whose replies we chose, because a harness that
scores deliberately wrong answers well is worse than no harness at all.
"""

import asyncio

import pytest

from checks import check, normalise, tally
from mock_agent import serve
from run_eval import run_scenario, score
from schema import CATEGORIES, CATEGORY_RUBRICS, Question, all_questions, load_dataset


@pytest.fixture(scope="module")
def dataset():
    return load_dataset()


class TestDataset:
    def test_has_enough_questions(self, dataset):
        """Issue #2 asked for 100 to 200."""
        count = sum(1 for _ in all_questions(dataset))
        assert 100 <= count <= 200, count

    def test_at_least_half_need_meeting_state(self, dataset):
        pairs = list(all_questions(dataset))
        stateful = sum(1 for _, q in pairs if q.requires_state)
        assert stateful / len(pairs) >= 0.5, f"{stateful}/{len(pairs)}"

    def test_there_is_a_real_general_knowledge_control_group(self, dataset):
        """
        Without questions that do not need the meeting, a regression that
        broke ordinary answering would not show up anywhere.
        """
        general = [q for _, q in all_questions(dataset) if not q.requires_state]
        assert len(general) >= 10, len(general)

    def test_every_category_is_covered(self, dataset):
        used = {q.category for _, q in all_questions(dataset)}
        assert used == CATEGORIES, CATEGORIES - used

    def test_deferral_questions_exist_in_numbers(self, dataset):
        """
        The expensive failure is a confident invented answer, so the set that
        catches it cannot be a token handful.
        """
        deferrals = [q for _, q in all_questions(dataset) if q.category == "deferral"]
        assert len(deferrals) >= 15, len(deferrals)

    def test_every_category_carries_grading_guidance(self):
        """
        The stance lives on the category so each question only states what is
        specific to it. A category without guidance would be graded on vibes.
        """
        assert set(CATEGORY_RUBRICS) == CATEGORIES
        for category, guidance in CATEGORY_RUBRICS.items():
            assert len(guidance) > 80, category

    def test_every_question_says_something_specific(self, dataset):
        for scenario, question in all_questions(dataset):
            assert len(question.rubric) >= 6, f"{scenario.id}.{question.id}"

    def test_transcripts_are_substantial(self, dataset):
        for scenario in dataset:
            assert len(scenario.transcript) >= 5, scenario.id


class TestNormalise:
    def test_flattens_case_and_punctuation(self):
        assert normalise("Usage-Based!") == normalise("usage based")

    def test_collapses_runs_of_separators(self):
        assert normalise("a  --  b") == "a b"

    def test_handles_empty(self):
        assert normalise("") == ""
        assert normalise(None) == ""


def _question(**kwargs):
    base = dict(id="q", category="decision_recall", requires_state=True,
                ask="?", rubric="r", mentions=(), avoids=())
    base.update(kwargs)
    return Question(**base)


class TestCheck:
    def test_passes_when_the_phrase_is_there(self):
        passed, reasons = check(_question(mentions=("usage based",)),
                                "We went with usage based pricing.")
        assert passed and reasons == []

    def test_hyphens_do_not_count_as_a_miss(self):
        passed, _ = check(_question(mentions=("usage based",)), "Usage-based, three tiers.")
        assert passed

    def test_a_list_means_any_of(self):
        question = _question(mentions=(["twelfth", "12th"],))
        assert check(question, "Launching on the 12th.")[0]
        assert check(question, "Launching on the twelfth.")[0]
        assert not check(question, "Launching next month.")[0]

    def test_avoids_catches_the_wrong_answer(self):
        passed, reasons = check(_question(mentions=("deepgram",), avoids=("whisper",)),
                                "We picked Deepgram over Whisper.")
        assert not passed
        assert "whisper" in reasons[0].lower()

    def test_empty_answer_fails(self):
        passed, reasons = check(_question(mentions=("anything",)), "")
        assert not passed and reasons == ["empty answer"]

    def test_reports_every_miss_not_just_the_first(self):
        question = _question(mentions=("alpha", "beta"), avoids=("gamma",))
        _, reasons = check(question, "only gamma here")
        assert len(reasons) == 3


class TestTally:
    def test_splits_by_category_and_by_state(self):
        rows = [
            (None, _question(category="decision_recall", requires_state=True), True),
            (None, _question(category="decision_recall", requires_state=True), False),
            (None, _question(category="general", requires_state=False), True),
        ]
        summary = tally(rows)

        assert summary["accuracy"] == round(2 / 3, 3)
        assert summary["by_category"]["decision_recall"]["accuracy"] == 0.5
        assert summary["needs_meeting_state"]["accuracy"] == 0.5
        assert summary["general_knowledge"]["accuracy"] == 1.0

    def test_empty_is_not_a_crash(self):
        assert tally([])["accuracy"] is None


class TestRunnerAgainstMock:
    @pytest.mark.asyncio
    async def test_perfect_answers_score_full_marks(self, dataset):
        scenario = next(s for s in dataset if s.id == "pricing-and-launch")

        # Feed back exactly what each question is looking for.
        replies = {}
        for question in scenario.questions:
            wanted = []
            for entry in question.mentions:
                wanted.append(entry[0] if isinstance(entry, (list, tuple)) else entry)
            replies[question.ask] = " and ".join(wanted) or "fine"

        async with serve(8821, replies):
            answers = await run_scenario("ws://127.0.0.1:8821", scenario, 5.0)

        summary = score([scenario], answers)
        assert summary["accuracy"] == 1.0, [a for a in answers if not a["passed"]]

    @pytest.mark.asyncio
    async def test_a_model_that_knows_nothing_scores_zero_on_state(self, dataset):
        scenario = next(s for s in dataset if s.id == "vendor-choice")

        async with serve(8822, {}, default="Sorry, could you repeat that?"):
            answers = await run_scenario("ws://127.0.0.1:8822", scenario, 5.0)

        summary = score([scenario], answers)
        assert summary["needs_meeting_state"]["accuracy"] == 0.0

    @pytest.mark.asyncio
    async def test_a_plausible_hallucination_is_caught(self, dataset):
        """
        The failure this whole set exists for: a confident, well written,
        completely invented answer. It must not score.
        """
        scenario = next(s for s in dataset if s.id == "vendor-choice")
        overage = next(q for q in scenario.questions if q.id == "overage-number")

        async with serve(8823, {overage.ask: "The overage rate is $0.004 per minute."}):
            answers = await run_scenario("ws://127.0.0.1:8823", scenario, 5.0)

        score([scenario], answers)
        graded = next(a for a in answers if a["question"] == "overage-number")
        assert graded["passed"] is False, graded

    @pytest.mark.asyncio
    async def test_the_runner_asks_every_question(self, dataset):
        scenario = next(s for s in dataset if s.id == "roadmap-cut")

        async with serve(8824, {}):
            answers = await run_scenario("ws://127.0.0.1:8824", scenario, 5.0)

        assert [a["question"] for a in answers] == [q.id for q in scenario.questions]


class TestJudgePrompt:
    def test_prompt_carries_both_layers_of_rubric(self, dataset):
        from judge import build_prompt
        from schema import CATEGORY_RUBRICS

        scenario = next(s for s in dataset if s.id == "vendor-choice")
        question = next(q for q in scenario.questions if q.category == "deferral")

        prompt = build_prompt(question, "It is $0.004 per minute.")
        assert CATEGORY_RUBRICS["deferral"] in prompt
        assert question.rubric in prompt
        assert question.ask in prompt
        assert "0.004" in prompt

    def test_an_empty_answer_is_described_not_blank(self, dataset):
        from judge import build_prompt

        question = next(q for _, q in all_questions(dataset))
        assert "said nothing" in build_prompt(question, "")


class TestParseVerdict:
    def test_plain_json(self):
        from judge import parse_verdict
        assert parse_verdict('{"verdict": "pass", "reason": "named it"}')["verdict"] == "pass"

    def test_json_wrapped_in_a_code_fence(self):
        from judge import parse_verdict
        fenced = '```json\n{"verdict": "fail", "reason": "invented"}\n```'
        assert parse_verdict(fenced)["verdict"] == "fail"

    def test_unparseable_reply_is_an_error_not_a_pass(self):
        """A judge that breaks must not silently mark things correct."""
        from judge import parse_verdict
        assert parse_verdict("I think it was fine actually")["verdict"] == "error"

    def test_unexpected_verdict_is_an_error(self):
        from judge import parse_verdict
        assert parse_verdict('{"verdict": "maybe"}')["verdict"] == "error"


class TestSampleForReview:
    def test_same_seed_gives_the_same_sample(self):
        from judge import sample_for_review
        pool = [{"i": i} for i in range(100)]
        assert sample_for_review(pool, 20) == sample_for_review(pool, 20)

    def test_asking_for_more_than_exists_returns_everything(self):
        from judge import sample_for_review
        pool = [{"i": i} for i in range(5)]
        assert len(sample_for_review(pool, 20)) == 5


class TestAgreement:
    def test_perfect_agreement(self):
        from judge import agreement
        result = agreement([("pass", "pass"), ("fail", "fail")] * 5)
        assert result["raw"] == 1.0
        assert result["kappa"] == 1.0

    def test_kappa_sees_through_a_judge_that_always_passes(self):
        """
        Nine of ten answers really are passes, and the judge says pass to
        everything. Raw agreement looks strong, kappa does not.
        """
        from judge import agreement
        pairs = [("pass", "pass")] * 9 + [("pass", "fail")]
        result = agreement(pairs)

        assert result["raw"] == 0.9
        assert result["kappa"] < 0.2, result

    def test_kappa_is_undefined_when_only_one_label_was_ever_used(self):
        from judge import agreement
        assert agreement([("pass", "pass")] * 10)["kappa"] is None

    def test_empty_sample(self):
        from judge import agreement
        assert agreement([]) == {"n": 0, "raw": None, "kappa": None}
