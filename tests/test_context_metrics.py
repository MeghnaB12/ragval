"""Tests for context-side metrics: ContextRecall and ContextPrecision."""

import tempfile

from ragval.judges import MockJudge
from ragval.metrics import METRIC_REGISTRY, ContextPrecision, ContextRecall
from ragval.types import EvalSample, RagOutput


def _sample_with_supporting(*titles: str) -> EvalSample:
    return EvalSample(
        id="s1",
        question="When did the war end?",
        ground_truth_answer="1945",
        ground_truth_contexts=list(titles),
        metadata={"supporting_titles": list(titles)},
    )


# ---------- ContextRecall (deterministic) ----------


def test_context_recall_in_registry():
    assert "context_recall" in METRIC_REGISTRY


def test_context_recall_perfect_when_all_supporting_titles_present():
    sample = _sample_with_supporting("World War II", "VE Day")
    output = RagOutput(
        answer="1945",
        retrieved_contexts=[
            "World War II: A global war that ended in 1945.",
            "VE Day: Victory in Europe Day, May 8, 1945.",
            "Some unrelated paragraph about cooking.",
        ],
    )
    judge = MockJudge(cache_dir=tempfile.mkdtemp())
    result = ContextRecall().score(sample, output, judge)
    assert result.score == 1.0
    assert result.cost_usd == 0.0
    assert result.judge_model == "none"


def test_context_recall_zero_when_none_retrieved():
    sample = _sample_with_supporting("World War II", "VE Day")
    output = RagOutput(
        answer="?",
        retrieved_contexts=["Cooking with garlic.", "How to train a puppy."],
    )
    judge = MockJudge(cache_dir=tempfile.mkdtemp())
    result = ContextRecall().score(sample, output, judge)
    assert result.score == 0.0


def test_context_recall_partial_credit():
    sample = _sample_with_supporting("World War II", "VE Day")
    output = RagOutput(
        answer="1945",
        retrieved_contexts=[
            "World War II: A global war that ended in 1945.",
            "Garlic recipes for beginners.",
        ],
    )
    judge = MockJudge(cache_dir=tempfile.mkdtemp())
    result = ContextRecall().score(sample, output, judge)
    assert result.score == 0.5  # 1 of 2 supporting titles found


def test_context_recall_case_insensitive():
    """A title in lowercase context should still be detected."""
    sample = _sample_with_supporting("World War II")
    output = RagOutput(
        answer="1945",
        retrieved_contexts=["world war ii was a major conflict."],
    )
    judge = MockJudge(cache_dir=tempfile.mkdtemp())
    result = ContextRecall().score(sample, output, judge)
    assert result.score == 1.0


def test_context_recall_handles_duplicate_supporting_titles():
    """HotpotQA sometimes lists the same title twice (one per supporting fact).
    The denominator should be unique titles, not raw count."""
    sample = EvalSample(
        id="s1",
        question="?",
        ground_truth_answer="x",
        metadata={"supporting_titles": ["Foo", "Foo", "Bar"]},
    )
    output = RagOutput(answer="?", retrieved_contexts=["foo paragraph", "bar paragraph"])
    judge = MockJudge(cache_dir=tempfile.mkdtemp())
    result = ContextRecall().score(sample, output, judge)
    assert result.score == 1.0  # 2 unique titles, both found


def test_context_recall_no_supporting_titles_is_vacuous():
    """When there's no ground truth, recall is undefined; report as 1.0."""
    sample = EvalSample(id="s1", question="?", ground_truth_answer="x", metadata={})
    output = RagOutput(answer="?", retrieved_contexts=["anything"])
    judge = MockJudge(cache_dir=tempfile.mkdtemp())
    result = ContextRecall().score(sample, output, judge)
    assert result.score == 1.0


def test_context_recall_no_judge_calls():
    """ContextRecall is deterministic — must not call the judge."""
    sample = _sample_with_supporting("Foo")
    output = RagOutput(answer="?", retrieved_contexts=["foo paragraph"])
    judge = MockJudge(cache_dir=tempfile.mkdtemp())
    ContextRecall().score(sample, output, judge)
    assert judge.call_count == 0


# ---------- ContextPrecision (judged) ----------


def test_context_precision_in_registry():
    assert "context_precision" in METRIC_REGISTRY


def test_context_precision_all_relevant():
    judge = MockJudge(
        cache_dir=tempfile.mkdtemp(),
        response_text='{"score": 5, "reasoning": "highly relevant"}',
    )
    sample = _sample_with_supporting("Foo")
    output = RagOutput(answer="?", retrieved_contexts=["chunk a", "chunk b", "chunk c"])
    result = ContextPrecision().score(sample, output, judge)
    assert result.score == 1.0
    assert result.raw_score == 5.0
    assert judge.call_count == 3  # one call per chunk


def test_context_precision_all_irrelevant():
    judge = MockJudge(
        cache_dir=tempfile.mkdtemp(),
        response_text='{"score": 1, "reasoning": "irrelevant"}',
    )
    sample = _sample_with_supporting("Foo")
    output = RagOutput(answer="?", retrieved_contexts=["a", "b"])
    result = ContextPrecision().score(sample, output, judge)
    assert result.score == 0.0


def test_context_precision_empty_retrieval_scores_zero():
    """No retrieval = no precision. Don't reward the retriever for retrieving nothing."""
    judge = MockJudge(cache_dir=tempfile.mkdtemp())
    sample = _sample_with_supporting("Foo")
    output = RagOutput(answer="?", retrieved_contexts=[])
    result = ContextPrecision().score(sample, output, judge)
    assert result.score == 0.0
    assert judge.call_count == 0


def test_context_precision_handles_parse_failure():
    judge = MockJudge(
        cache_dir=tempfile.mkdtemp(),
        response_text="i refuse to respond in json haha",
    )
    sample = _sample_with_supporting("Foo")
    output = RagOutput(answer="?", retrieved_contexts=["a"])
    result = ContextPrecision().score(sample, output, judge)
    assert result.raw_score == 1.0
    assert "PARSE_FAILURE" in result.reasoning
