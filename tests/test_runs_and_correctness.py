import tempfile
from datetime import datetime, timezone

from ragval.judges import MockJudge
from ragval.metrics import METRIC_REGISTRY, AnswerCorrectness
from ragval.runs import list_runs, load_run, save_run
from ragval.types import EvalSample, MetricResult, RagOutput, RunResult, SampleResult


def _sample():
    return EvalSample(
        id="t1",
        question="Who painted the Mona Lisa?",
        ground_truth_answer="Leonardo da Vinci",
        ground_truth_contexts=[],
    )


def test_answer_correctness_in_registry():
    assert "answer_correctness" in METRIC_REGISTRY


def test_answer_correctness_full_match():
    judge = MockJudge(
        cache_dir=tempfile.mkdtemp(),
        response_text='{"score": 5, "reasoning": "matches ground truth"}',
    )
    output = RagOutput(answer="Leonardo da Vinci painted it.", retrieved_contexts=[])
    result = AnswerCorrectness().score(_sample(), output, judge)
    assert result.metric_name == "answer_correctness"
    assert result.score == 1.0
    assert result.raw_score == 5.0


def test_answer_correctness_wrong():
    judge = MockJudge(
        cache_dir=tempfile.mkdtemp(),
        response_text='{"score": 1, "reasoning": "wrong artist"}',
    )
    output = RagOutput(answer="Picasso painted the Mona Lisa.", retrieved_contexts=[])
    result = AnswerCorrectness().score(_sample(), output, judge)
    assert result.score == 0.0


def test_answer_correctness_refusal_scores_low():
    """Refusing to answer is not correct, even if it's faithful."""
    judge = MockJudge(
        cache_dir=tempfile.mkdtemp(),
        response_text='{"score": 1, "reasoning": "refused to answer"}',
    )
    output = RagOutput(
        answer="The context does not contain information about this.",
        retrieved_contexts=[],
    )
    result = AnswerCorrectness().score(_sample(), output, judge)
    assert result.score == 0.0


def _make_run(n: int = 3) -> RunResult:
    samples = [
        SampleResult(
            sample_id=f"q{i}",
            rag_output=RagOutput(answer=f"answer {i}", retrieved_contexts=["ctx"]),
            metrics={
                "faithfulness": MetricResult(
                    metric_name="faithfulness", score=0.8, raw_score=4, reasoning="ok"
                ),
            },
        )
        for i in range(n)
    ]
    return RunResult(
        run_id="run-test-123",
        config_name="test-config",
        dataset_name="test-dataset",
        timestamp=datetime.now(timezone.utc),
        samples=samples,
        total_cost_usd=0.0042,
    )


def test_save_and_load_run_round_trip():
    run = _make_run(3)
    with tempfile.TemporaryDirectory() as tmp:
        path = save_run(run, results_dir=tmp)
        assert path.exists()
        loaded = load_run(path)

    assert loaded.run_id == run.run_id
    assert loaded.config_name == run.config_name
    assert len(loaded.samples) == 3
    assert loaded.total_cost_usd == 0.0042
    assert loaded.samples[0].metrics["faithfulness"].score == 0.8


def test_list_runs_returns_headers_only():
    run = _make_run(2)
    with tempfile.TemporaryDirectory() as tmp:
        save_run(run, results_dir=tmp)
        # Save a second run with a different ID
        run2 = _make_run(1)
        run2.run_id = "run-test-456"
        save_run(run2, results_dir=tmp)

        listings = list_runs(tmp)

    assert len(listings) == 2
    ids = {h["run_id"] for h in listings}
    assert ids == {"run-test-123", "run-test-456"}
    # Each header has counts and metric names
    for h in listings:
        assert "n_samples" in h
        assert "metric_names" in h
        assert "_path" in h


def test_list_runs_empty_dir():
    with tempfile.TemporaryDirectory() as tmp:
        assert list_runs(tmp) == []


def test_list_runs_missing_dir():
    assert list_runs("/nonexistent/path/that/does/not/exist") == []


def test_run_file_format_is_jsonl():
    """Each line of the saved file must be valid JSON, header first."""
    import json

    run = _make_run(2)
    with tempfile.TemporaryDirectory() as tmp:
        path = save_run(run, results_dir=tmp)
        lines = path.read_text().splitlines()

    assert len(lines) == 1 + 2  # header + samples
    header = json.loads(lines[0])
    assert header["_kind"] == "header"
    assert header["n_samples"] == 2
    for line in lines[1:]:
        data = json.loads(line)
        assert "sample_id" in data
        assert "metrics" in data
