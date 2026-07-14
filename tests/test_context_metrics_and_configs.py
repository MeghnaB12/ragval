"""Tests for context metrics and the 8 benchmark configs."""

import pytest

from ragval.configs import CONFIG_NAMES, build_rag_for_sample, extract_final_answer
from ragval.judges import MockJudge
from ragval.metrics import (
    ContextPrecision,
    ContextRecall,
    RetrievalPrecision,
    RetrievalRecall,
)
from ragval.types import EvalSample, RagOutput


def _hotpot_sample() -> EvalSample:
    paragraphs = [
        {"title": "George Orwell", "text": "George Orwell wrote 1984, published in 1949."},
        {"title": "Aldous Huxley", "text": "Aldous Huxley wrote Brave New World."},
        {"title": "Canberra", "text": "Canberra is the capital of Australia."},
        {"title": "Sydney", "text": "Sydney is the largest city in Australia."},
    ]
    return EvalSample(
        id="hp-1",
        question="Who wrote 1984?",
        ground_truth_answer="George Orwell",
        ground_truth_contexts=["George Orwell"],
        metadata={"paragraphs": paragraphs, "supporting_titles": ["George Orwell"]},
    )


# ---------- deterministic retrieval metrics ----------


def test_retrieval_recall_full_hit():
    sample = _hotpot_sample()
    output = RagOutput(answer="x", retrieved_contexts=["George Orwell: George Orwell wrote 1984."])
    r = RetrievalRecall().score(sample, output, MockJudge())
    assert r.score == 1.0


def test_retrieval_recall_miss():
    sample = _hotpot_sample()
    output = RagOutput(answer="x", retrieved_contexts=["Sydney: Sydney is the largest city."])
    r = RetrievalRecall().score(sample, output, MockJudge())
    assert r.score == 0.0


def test_retrieval_precision_partial():
    sample = _hotpot_sample()
    output = RagOutput(
        answer="x",
        retrieved_contexts=[
            "George Orwell: wrote 1984.",
            "Sydney: largest city.",
        ],
    )
    r = RetrievalPrecision().score(sample, output, MockJudge())
    assert r.score == 0.5


def test_retrieval_precision_empty_contexts():
    r = RetrievalPrecision().score(
        _hotpot_sample(), RagOutput(answer="x", retrieved_contexts=[]), MockJudge()
    )
    assert r.score == 0.0


# ---------- judge-based context metrics ----------


def test_context_precision_parses_bool_list():
    judge = MockJudge(response_text='{"relevant": [true, false], "reasoning": "r"}')
    output = RagOutput(answer="x", retrieved_contexts=["c1", "c2"])
    r = ContextPrecision().score(_hotpot_sample(), output, judge)
    assert r.score == 0.5
    assert r.metadata["relevant_flags"] == [True, False]


def test_context_precision_length_mismatch_is_parse_failure():
    judge = MockJudge(response_text='{"relevant": [true], "reasoning": "r"}')
    output = RagOutput(answer="x", retrieved_contexts=["c1", "c2"])
    r = ContextPrecision().score(_hotpot_sample(), output, judge)
    assert r.raw_score is None
    assert "PARSE_FAILURE" in r.reasoning


def test_context_recall_all_covered():
    judge = MockJudge(response_text='{"covered": [true, true], "reasoning": "r"}')
    output = RagOutput(answer="x", retrieved_contexts=["c1"])
    r = ContextRecall().score(_hotpot_sample(), output, judge)
    assert r.score == 1.0


def test_context_recall_handles_code_fences():
    judge = MockJudge(response_text='```json\n{"covered": [true, false], "reasoning": "r"}\n```')
    r = ContextRecall().score(
        _hotpot_sample(), RagOutput(answer="x", retrieved_contexts=["c"]), judge
    )
    assert r.score == 0.5


# ---------- configs ----------


def test_all_eight_configs_run():
    sample = _hotpot_sample()
    calls: list[str] = []

    def generate(prompt: str) -> str:
        calls.append(prompt)
        return "FINAL ANSWER: George Orwell" if "FINAL ANSWER" in prompt else "George Orwell"

    assert len(CONFIG_NAMES) == 8
    for config in CONFIG_NAMES:
        out = build_rag_for_sample(config, sample, generate)(sample.question)
        assert out.answer == "George Orwell", config


def test_closed_book_has_no_contexts():
    out = build_rag_for_sample("closed_book", _hotpot_sample(), lambda p: "a")("q")
    assert out.retrieved_contexts == []


def test_oracle_retrieves_only_gold():
    out = build_rag_for_sample("oracle", _hotpot_sample(), lambda p: "a")("q")
    assert len(out.retrieved_contexts) == 1
    assert out.retrieved_contexts[0].startswith("George Orwell:")


def test_bm25_k_controls_context_count():
    sample = _hotpot_sample()
    out1 = build_rag_for_sample("bm25_k1", sample, lambda p: "a")("q")
    out3 = build_rag_for_sample("bm25_k3", sample, lambda p: "a")("q")
    assert len(out1.retrieved_contexts) == 1
    assert len(out3.retrieved_contexts) == 3


def test_full_context_uses_all_paragraphs():
    out = build_rag_for_sample("full_context", _hotpot_sample(), lambda p: "a")("q")
    assert len(out.retrieved_contexts) == 4


def test_cot_answer_extraction():
    assert extract_final_answer("thinking...\nFINAL ANSWER: 42") == "42"
    assert extract_final_answer("no marker here") == "no marker here"


def test_unknown_config_raises():
    with pytest.raises(ValueError, match="Unknown config"):
        build_rag_for_sample("nope", _hotpot_sample(), lambda p: "a")("q")


def test_cot_keeps_raw_generation():
    def generate(prompt: str) -> str:
        return "Step 1... Step 2...\nFINAL ANSWER: George Orwell"

    out = build_rag_for_sample("bm25_k3_cot", _hotpot_sample(), generate)("q")
    assert out.answer == "George Orwell"
    assert "Step 1" in out.metadata["raw_generation"]


def test_retrieval_metrics_handle_colons_in_titles():
    """Wikipedia titles can contain colons; prefix matching must survive them."""
    sample = EvalSample(
        id="hp-2",
        question="q",
        ground_truth_answer="a",
        metadata={
            "paragraphs": [],
            "supporting_titles": ["Star Trek: First Contact"],
        },
    )
    output = RagOutput(
        answer="x",
        retrieved_contexts=["Star Trek: First Contact: A 1996 film."],
    )
    assert RetrievalRecall().score(sample, output, MockJudge()).score == 1.0
    assert RetrievalPrecision().score(sample, output, MockJudge()).score == 1.0
