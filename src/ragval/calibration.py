"""Judge calibration.

Principle 2 of ragval: *if your judge isn't calibrated, your numbers are
theater.* This module measures how well an LLM judge's scores agree with
human labels on a small set (~20+) of human-labeled examples.

Workflow:

1. Write a JSONL calibration file. Each line is a `CalibrationExample`:
       {"question": ..., "answer": ..., "contexts": [...],
        "ground_truth_answer": ..., "human_score": <int 1-5>,
        "metric": "faithfulness"}
2. Call `calibrate(metric, judge, examples)`.
3. Read the `CalibrationReport`: exact agreement, within-1 agreement,
   quadratic-weighted Cohen's kappa, Spearman correlation, and mean bias.

Rules of thumb (documented, not enforced):
- within-1 agreement >= 0.85 and QWK >= 0.5 → judge is usable
- Spearman >= 0.6 → judge at least ranks outputs in the right order
- |bias| > 0.5 raw points → judge is systematically lenient/harsh; consider
  reporting judge scores relative to a calibrated offset.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from ragval.judges import Judge
from ragval.metrics import Metric
from ragval.types import EvalSample, RagOutput


class CalibrationExample(BaseModel):
    """One human-labeled example for judge calibration.

    `human_score` uses the same raw 1-5 scale as the metric rubrics.
    """

    question: str
    answer: str
    contexts: list[str] = Field(default_factory=list)
    ground_truth_answer: str = ""
    human_score: int = Field(ge=1, le=5)
    metric: str = ""
    id: str = ""


@dataclass
class CalibrationReport:
    """Agreement statistics between a judge and human labels."""

    metric_name: str
    judge_model: str
    n: int
    exact_agreement: float  # judge raw == human raw
    within_one_agreement: float  # |judge - human| <= 1
    quadratic_weighted_kappa: float
    spearman: float
    mean_bias: float  # mean(judge - human); positive = judge is lenient
    n_parse_failures: int
    judge_scores: list[float]
    human_scores: list[int]

    def __str__(self) -> str:
        return (
            f"Calibration [{self.metric_name} / {self.judge_model}] (n={self.n}):\n"
            f"  exact agreement:    {self.exact_agreement:.2f}\n"
            f"  within-1 agreement: {self.within_one_agreement:.2f}\n"
            f"  weighted kappa:     {self.quadratic_weighted_kappa:.3f}\n"
            f"  spearman:           {self.spearman:.3f}\n"
            f"  mean bias:          {self.mean_bias:+.2f} (positive = judge lenient)\n"
            f"  parse failures:     {self.n_parse_failures}"
        )

    @property
    def usable(self) -> bool:
        """Rule-of-thumb gate: is this judge trustworthy enough to report numbers?"""
        return self.within_one_agreement >= 0.85 and self.quadratic_weighted_kappa >= 0.5


def load_calibration_file(path: Path | str) -> list[CalibrationExample]:
    """Load calibration examples from JSONL."""
    examples: list[CalibrationExample] = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(CalibrationExample(**json.loads(line)))
    return examples


def quadratic_weighted_kappa(
    a: list[int] | np.ndarray, b: list[int] | np.ndarray, min_rating: int = 1, max_rating: int = 5
) -> float:
    """Quadratic-weighted Cohen's kappa for two raters on an ordinal scale.

    Standard metric for ordinal agreement (used e.g. in ASAP essay scoring).
    Implemented directly to avoid a scikit-learn dependency.
    """
    a_arr = np.asarray(a, dtype=int)
    b_arr = np.asarray(b, dtype=int)
    n_ratings = max_rating - min_rating + 1

    # Observed confusion matrix
    obs = np.zeros((n_ratings, n_ratings))
    for x, y in zip(a_arr, b_arr, strict=True):
        obs[x - min_rating, y - min_rating] += 1
    obs /= obs.sum()

    # Expected under independence
    hist_a = obs.sum(axis=1)
    hist_b = obs.sum(axis=0)
    exp = np.outer(hist_a, hist_b)

    # Quadratic weights
    idx = np.arange(n_ratings)
    w = (idx[:, None] - idx[None, :]) ** 2 / (n_ratings - 1) ** 2

    denom = (w * exp).sum()
    if denom == 0:
        return 1.0 if (w * obs).sum() == 0 else 0.0
    return float(1.0 - (w * obs).sum() / denom)


def spearman_correlation(a: list[float] | np.ndarray, b: list[float] | np.ndarray) -> float:
    """Spearman rank correlation, implemented with numpy (average ranks for ties)."""

    def _ranks(x: np.ndarray) -> np.ndarray:
        order = np.argsort(x, kind="stable")
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(x) + 1, dtype=float)
        # Average ranks for ties
        for v in np.unique(x):
            mask = x == v
            if mask.sum() > 1:
                ranks[mask] = ranks[mask].mean()
        return ranks

    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    ra, rb = _ranks(a_arr), _ranks(b_arr)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def calibrate(
    metric: Metric,
    judge: Judge,
    examples: list[CalibrationExample],
    show_progress: bool = True,
) -> CalibrationReport:
    """Score every calibration example with the judge and compare to human labels.

    Parse failures are excluded from agreement statistics but counted in the
    report — a judge that can't follow the output format is itself a finding.
    """
    if len(examples) < 5:
        raise ValueError(f"Need at least 5 calibration examples, got {len(examples)}")

    from tqdm import tqdm

    judge_raw: list[float] = []
    human_raw: list[int] = []
    parse_failures = 0

    iterator = tqdm(examples, desc=f"calibrate {metric.name}", disable=not show_progress)
    for i, ex in enumerate(iterator):
        sample = EvalSample(
            id=ex.id or f"cal-{i}",
            question=ex.question,
            ground_truth_answer=ex.ground_truth_answer,
        )
        output = RagOutput(answer=ex.answer, retrieved_contexts=ex.contexts)
        result = metric.score(sample, output, judge)
        if result.raw_score is None:
            parse_failures += 1
            continue
        judge_raw.append(result.raw_score)
        human_raw.append(ex.human_score)

    if len(judge_raw) < 2:
        raise ValueError(
            f"Only {len(judge_raw)} parseable judge responses ({parse_failures} failures) — "
            "cannot compute agreement."
        )

    j = np.asarray(judge_raw)
    h = np.asarray(human_raw, dtype=float)
    diffs = np.abs(j - h)

    return CalibrationReport(
        metric_name=metric.name,
        judge_model=judge.model_id,
        n=len(judge_raw),
        exact_agreement=float((diffs == 0).mean()),
        within_one_agreement=float((diffs <= 1).mean()),
        quadratic_weighted_kappa=quadratic_weighted_kappa(
            [int(round(x)) for x in judge_raw], human_raw
        ),
        spearman=spearman_correlation(j, h),
        mean_bias=float((j - h).mean()),
        n_parse_failures=parse_failures,
        judge_scores=judge_raw,
        human_scores=human_raw,
    )
