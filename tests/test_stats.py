"""Tests for the statistical layer."""

from datetime import datetime, timezone

import numpy as np
import pytest

from ragval.stats import (
    bootstrap_ci,
    compare_runs,
    paired_bootstrap_test,
    permutation_test,
    summarize_metric,
    summarize_run,
)
from ragval.types import MetricResult, RagOutput, RunResult, SampleResult


def _make_run(config: str, scores: list[float], metric: str = "m") -> RunResult:
    samples = [
        SampleResult(
            sample_id=f"s{i}",
            rag_output=RagOutput(answer="a", retrieved_contexts=[]),
            metrics={metric: MetricResult(metric_name=metric, score=sc)},
        )
        for i, sc in enumerate(scores)
    ]
    return RunResult(
        run_id=f"{config}-test",
        config_name=config,
        dataset_name="test",
        timestamp=datetime.now(timezone.utc),
        samples=samples,
    )


def test_bootstrap_ci_contains_mean():
    rng = np.random.default_rng(0)
    scores = rng.uniform(0, 1, size=200).tolist()
    mean, lo, hi = bootstrap_ci(scores)
    assert lo <= mean <= hi
    assert abs(mean - np.mean(scores)) < 1e-9


def test_bootstrap_ci_narrows_with_n():
    rng = np.random.default_rng(0)
    small = rng.uniform(0, 1, size=20).tolist()
    large = rng.uniform(0, 1, size=2000).tolist()
    _, lo_s, hi_s = bootstrap_ci(small)
    _, lo_l, hi_l = bootstrap_ci(large)
    assert (hi_l - lo_l) < (hi_s - lo_s)


def test_bootstrap_ci_deterministic():
    scores = [0.2, 0.4, 0.6, 0.8, 1.0]
    assert bootstrap_ci(scores, seed=7) == bootstrap_ci(scores, seed=7)


def test_bootstrap_ci_rejects_empty():
    with pytest.raises(ValueError):
        bootstrap_ci([])


def test_paired_bootstrap_detects_real_difference():
    rng = np.random.default_rng(1)
    b = rng.uniform(0.3, 0.7, size=100)
    a = np.clip(b + 0.15 + rng.normal(0, 0.05, size=100), 0, 1)  # a clearly better
    diff, lo, hi, p = paired_bootstrap_test(a, b)
    assert diff > 0.1
    assert lo > 0  # CI excludes zero
    assert p < 0.01


def test_paired_bootstrap_no_false_positive_on_noise():
    rng = np.random.default_rng(2)
    base = rng.uniform(0.3, 0.7, size=100)
    a = np.clip(base + rng.normal(0, 0.1, size=100), 0, 1)
    b = np.clip(base + rng.normal(0, 0.1, size=100), 0, 1)
    _, lo, hi, p = paired_bootstrap_test(a, b)
    assert lo <= 0 <= hi
    assert p > 0.05


def test_permutation_test_agrees_directionally():
    rng = np.random.default_rng(3)
    b = rng.uniform(0.3, 0.7, size=80)
    a = np.clip(b + 0.2, 0, 1)
    assert permutation_test(a, b) < 0.01
    assert permutation_test(b, b) == 1.0


def test_compare_runs_pairs_by_sample_id():
    run_a = _make_run("a", [0.9] * 30)
    run_b = _make_run("b", [0.5] * 30)
    c = compare_runs(run_a, run_b, "m")
    assert c.n == 30
    assert c.mean_diff == pytest.approx(0.4)
    assert c.significant


def test_compare_runs_rejects_disjoint():
    run_a = _make_run("a", [0.9] * 5)
    run_b = _make_run("b", [0.5] * 5, metric="other")
    with pytest.raises(ValueError):
        compare_runs(run_a, run_b, "m")


def test_summarize_run_and_metric():
    run = _make_run("a", [0.2, 0.4, 0.6, 0.8])
    summary = summarize_metric(run, "m")
    assert summary.n == 4
    assert summary.mean == pytest.approx(0.5)
    assert summarize_run(run)[0].metric_name == "m"
