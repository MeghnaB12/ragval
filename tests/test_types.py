from datetime import datetime, timezone

from ragval.types import EvalSample, MetricResult, RagOutput, RunResult, SampleResult


def test_eval_sample_minimal():
    s = EvalSample(id="x", question="?", ground_truth_answer="!")
    assert s.id == "x"
    assert s.ground_truth_contexts == []
    assert s.metadata == {}


def test_rag_output_serializes():
    o = RagOutput(answer="hi", retrieved_contexts=["a", "b"])
    data = o.model_dump()
    assert data["answer"] == "hi"
    assert data["retrieved_contexts"] == ["a", "b"]


def test_metric_result_score_range():
    m = MetricResult(metric_name="faith", score=0.5, raw_score=3.0)
    assert m.score == 0.5
    assert m.raw_score == 3.0


def test_run_result_metric_helpers():
    samples = [
        SampleResult(
            sample_id=f"s{i}",
            rag_output=RagOutput(answer="a", retrieved_contexts=[]),
            metrics={"f": MetricResult(metric_name="f", score=float(i) / 10)},
        )
        for i in range(5)
    ]
    rr = RunResult(
        run_id="r1",
        config_name="c",
        dataset_name="d",
        timestamp=datetime.now(timezone.utc),
        samples=samples,
    )
    assert rr.metric_names() == ["f"]
    assert rr.metric_scores("f") == [0.0, 0.1, 0.2, 0.3, 0.4]
