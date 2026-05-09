import tempfile

from ragval.judges import MockJudge
from ragval.metrics import AnswerRelevance, Faithfulness
from ragval.runner import run_eval
from ragval.types import EvalSample, RagOutput


def _make_dataset(n: int) -> list[EvalSample]:
    return [
        EvalSample(
            id=f"q{i}",
            question=f"Question {i}?",
            ground_truth_answer=f"Answer {i}",
            ground_truth_contexts=[f"context {i}"],
        )
        for i in range(n)
    ]


def _fake_rag(question: str) -> RagOutput:
    return RagOutput(answer=f"Stub answer for: {question}", retrieved_contexts=["stub"])


def test_run_eval_basic():
    dataset = _make_dataset(3)
    judge = MockJudge(
        cache_dir=tempfile.mkdtemp(),
        response_text='{"score": 4, "reasoning": "ok"}',
    )
    result = run_eval(
        rag_system=_fake_rag,
        dataset=dataset,
        metrics=[Faithfulness(), AnswerRelevance()],
        judge=judge,
        config_name="test",
        dataset_name="test-3",
        show_progress=False,
    )
    assert len(result.samples) == 3
    assert set(result.metric_names()) == {"faithfulness", "answer_relevance"}
    for m in result.metric_names():
        scores = result.metric_scores(m)
        assert all(s == 0.75 for s in scores)  # raw 4 -> normalized 0.75


def test_run_eval_handles_rag_error():
    def broken_rag(q):
        raise RuntimeError("boom")

    dataset = _make_dataset(2)
    judge = MockJudge(
        cache_dir=tempfile.mkdtemp(),
        response_text='{"score": 1, "reasoning": "broken"}',
    )
    result = run_eval(
        rag_system=broken_rag,
        dataset=dataset,
        metrics=[Faithfulness()],
        judge=judge,
        show_progress=False,
    )
    assert len(result.samples) == 2
    for s in result.samples:
        assert "RAG_ERROR" in s.rag_output.answer
