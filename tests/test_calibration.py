"""Tests for judge calibration."""

import json

import pytest

from ragval.calibration import (
    CalibrationExample,
    calibrate,
    load_calibration_file,
    quadratic_weighted_kappa,
    spearman_correlation,
)
from ragval.judges import MockJudge
from ragval.metrics import Faithfulness


def _examples(n: int = 6, score: int = 4) -> list[CalibrationExample]:
    return [
        CalibrationExample(
            question=f"q{i}",
            answer=f"a{i}",
            contexts=[f"c{i}"],
            human_score=score,
            metric="faithfulness",
        )
        for i in range(n)
    ]


def test_perfect_agreement():
    judge = MockJudge(response_text='{"score": 4, "reasoning": "mock"}')
    report = calibrate(Faithfulness(), judge, _examples(score=4), show_progress=False)
    assert report.exact_agreement == 1.0
    assert report.within_one_agreement == 1.0
    assert report.mean_bias == 0.0
    assert report.usable


def test_systematic_bias_detected():
    judge = MockJudge(response_text='{"score": 5, "reasoning": "lenient"}')
    report = calibrate(Faithfulness(), judge, _examples(score=3), show_progress=False)
    assert report.exact_agreement == 0.0
    assert report.mean_bias == pytest.approx(2.0)
    assert not report.usable


def test_parse_failures_counted_not_crashed():
    # MockJudge caching means all identical prompts return the same garbage;
    # use distinct questions so each is a separate call.
    judge = MockJudge(response_text="not json at all")
    with pytest.raises(ValueError, match="parseable"):
        calibrate(Faithfulness(), judge, _examples(), show_progress=False)


def test_requires_minimum_examples():
    judge = MockJudge()
    with pytest.raises(ValueError, match="at least 5"):
        calibrate(Faithfulness(), judge, _examples(n=3), show_progress=False)


def test_qwk_perfect_and_random():
    assert quadratic_weighted_kappa([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) == pytest.approx(1.0)
    # Anti-correlated should be strongly negative
    assert quadratic_weighted_kappa([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]) < 0


def test_spearman():
    assert spearman_correlation([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman_correlation([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)
    assert spearman_correlation([1, 1, 1], [2, 3, 4]) == 0.0


def test_load_calibration_file(tmp_path):
    path = tmp_path / "cal.jsonl"
    rows = [
        {"question": "q", "answer": "a", "human_score": 4},
        {"question": "q2", "answer": "a2", "contexts": ["c"], "human_score": 2},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows))
    examples = load_calibration_file(path)
    assert len(examples) == 2
    assert examples[1].human_score == 2
