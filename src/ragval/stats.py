"""Statistical layer.

This is the reason ragval exists. Most eval tools report "config A: 0.74,
config B: 0.71" and stop. ragval answers the question that actually matters:
*is that difference real, or is it noise?*

Three tools:

1. `bootstrap_ci` — a 95% (by default) percentile bootstrap confidence
   interval around a metric mean. Nonparametric: no normality assumption,
   which matters because per-sample metric scores are bounded in [0, 1]
   and often heavily skewed toward the endpoints.

2. `paired_bootstrap_test` — a paired bootstrap test on the *per-sample
   score differences* between two runs on the same dataset. Pairing is
   essential: question difficulty varies enormously (HotpotQA "easy" vs
   "hard" bridge questions), and pairing removes that variance from the
   comparison. An unpaired test on the same data can easily be 5-10x less
   powerful.

3. `permutation_test` — a paired sign-flip permutation test as a second
   opinion. Under H0 (no difference), the sign of each per-sample
   difference is arbitrary, so we flip signs at random and see how often
   the shuffled mean difference is at least as extreme as the observed one.

All functions are seeded and deterministic by default (seed=42) so results
are reproducible, in keeping with the framework's reproducibility principle.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ragval.types import RunResult

DEFAULT_N_RESAMPLES = 10_000
DEFAULT_CONFIDENCE = 0.95
DEFAULT_SEED = 42


@dataclass
class MetricSummary:
    """Mean + bootstrap CI for one metric in one run."""

    metric_name: str
    n: int
    mean: float
    ci_low: float
    ci_high: float
    confidence: float
    std: float

    def __str__(self) -> str:
        pct = int(self.confidence * 100)
        return (
            f"{self.metric_name}: {self.mean:.3f} "
            f"[{pct}% CI {self.ci_low:.3f}–{self.ci_high:.3f}] (n={self.n})"
        )


@dataclass
class ComparisonResult:
    """Result of comparing one metric between two runs on the same dataset."""

    metric_name: str
    config_a: str
    config_b: str
    n: int
    mean_a: float
    mean_b: float
    mean_diff: float  # mean(a - b)
    diff_ci_low: float
    diff_ci_high: float
    p_value_bootstrap: float
    p_value_permutation: float
    confidence: float
    significant: bool = field(init=False)

    def __post_init__(self) -> None:
        alpha = 1.0 - self.confidence
        self.significant = self.p_value_bootstrap < alpha

    def __str__(self) -> str:
        verdict = "SIGNIFICANT" if self.significant else "not significant"
        pct = int(self.confidence * 100)
        return (
            f"{self.metric_name}: {self.config_a}={self.mean_a:.3f} vs "
            f"{self.config_b}={self.mean_b:.3f} | diff={self.mean_diff:+.3f} "
            f"[{pct}% CI {self.diff_ci_low:+.3f}–{self.diff_ci_high:+.3f}] "
            f"p_boot={self.p_value_bootstrap:.4f} p_perm={self.p_value_permutation:.4f} "
            f"→ {verdict}"
        )


def bootstrap_ci(
    scores: list[float] | np.ndarray,
    confidence: float = DEFAULT_CONFIDENCE,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for the mean of `scores`.

    Returns (mean, ci_low, ci_high).
    """
    arr = np.asarray(scores, dtype=float)
    if arr.size == 0:
        raise ValueError("scores must be non-empty")
    if arr.size == 1:
        return float(arr[0]), float(arr[0]), float(arr[0])

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_resamples, arr.size))
    boot_means = arr[idx].mean(axis=1)
    alpha = 1.0 - confidence
    lo, hi = np.quantile(boot_means, [alpha / 2, 1 - alpha / 2])
    return float(arr.mean()), float(lo), float(hi)


def summarize_metric(
    run: RunResult,
    metric_name: str,
    confidence: float = DEFAULT_CONFIDENCE,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> MetricSummary:
    """Mean + bootstrap CI for one metric in a run."""
    scores = run.metric_scores(metric_name)
    if not scores:
        raise ValueError(f"Run {run.run_id} has no scores for metric '{metric_name}'")
    mean, lo, hi = bootstrap_ci(scores, confidence, n_resamples, seed)
    return MetricSummary(
        metric_name=metric_name,
        n=len(scores),
        mean=mean,
        ci_low=lo,
        ci_high=hi,
        confidence=confidence,
        std=float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
    )


def summarize_run(run: RunResult, **kw) -> list[MetricSummary]:
    """Summaries for every metric in a run."""
    return [summarize_metric(run, name, **kw) for name in run.metric_names()]


def _paired_scores(
    run_a: RunResult, run_b: RunResult, metric_name: str
) -> tuple[np.ndarray, np.ndarray]:
    """Align per-sample scores between two runs by sample_id.

    Raises if the overlap is empty — comparing runs on disjoint datasets is
    a user error, not something to silently paper over.
    """
    a_scores = {
        s.sample_id: s.metrics[metric_name].score for s in run_a.samples if metric_name in s.metrics
    }
    b_scores = {
        s.sample_id: s.metrics[metric_name].score for s in run_b.samples if metric_name in s.metrics
    }
    common = sorted(set(a_scores) & set(b_scores))
    if not common:
        raise ValueError(
            f"No common samples with metric '{metric_name}' between "
            f"{run_a.run_id} and {run_b.run_id}"
        )
    return (
        np.array([a_scores[i] for i in common], dtype=float),
        np.array([b_scores[i] for i in common], dtype=float),
    )


def paired_bootstrap_test(
    a: np.ndarray | list[float],
    b: np.ndarray | list[float],
    confidence: float = DEFAULT_CONFIDENCE,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> tuple[float, float, float, float]:
    """Paired bootstrap on per-sample differences (a - b).

    Returns (mean_diff, diff_ci_low, diff_ci_high, p_value).

    The two-sided p-value is computed by shifting the bootstrap distribution
    of the mean difference to be centered at 0 (the null) and measuring how
    often a value at least as extreme as the observed mean difference occurs.
    """
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    if a_arr.shape != b_arr.shape:
        raise ValueError("paired arrays must have equal length")
    diffs = a_arr - b_arr
    observed = float(diffs.mean())

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diffs.size, size=(n_resamples, diffs.size))
    boot_means = diffs[idx].mean(axis=1)

    alpha = 1.0 - confidence
    lo, hi = np.quantile(boot_means, [alpha / 2, 1 - alpha / 2])

    # Shift to null: center bootstrap distribution at 0
    null_dist = boot_means - observed
    p = float((np.abs(null_dist) >= abs(observed)).mean())
    # Avoid reporting p=0 from a finite resample count
    p = max(p, 1.0 / n_resamples)
    return observed, float(lo), float(hi), p


def permutation_test(
    a: np.ndarray | list[float],
    b: np.ndarray | list[float],
    n_resamples: int = DEFAULT_N_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> float:
    """Paired sign-flip permutation test. Returns the two-sided p-value."""
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    if a_arr.shape != b_arr.shape:
        raise ValueError("paired arrays must have equal length")
    diffs = a_arr - b_arr
    observed = abs(float(diffs.mean()))
    if np.allclose(diffs, 0):
        return 1.0

    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_resamples, diffs.size))
    perm_means = np.abs((signs * diffs).mean(axis=1))
    p = float((perm_means >= observed).mean())
    return max(p, 1.0 / n_resamples)


def compare_runs(
    run_a: RunResult,
    run_b: RunResult,
    metric_name: str,
    confidence: float = DEFAULT_CONFIDENCE,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> ComparisonResult:
    """Compare one metric between two runs with paired tests.

    Alignment is by sample_id, so both runs must have been evaluated on
    (at least partially) the same dataset.
    """
    a, b = _paired_scores(run_a, run_b, metric_name)
    mean_diff, lo, hi, p_boot = paired_bootstrap_test(a, b, confidence, n_resamples, seed)
    p_perm = permutation_test(a, b, n_resamples, seed)
    return ComparisonResult(
        metric_name=metric_name,
        config_a=run_a.config_name,
        config_b=run_b.config_name,
        n=int(a.size),
        mean_a=float(a.mean()),
        mean_b=float(b.mean()),
        mean_diff=mean_diff,
        diff_ci_low=lo,
        diff_ci_high=hi,
        p_value_bootstrap=p_boot,
        p_value_permutation=p_perm,
        confidence=confidence,
    )


def compare_all_metrics(run_a: RunResult, run_b: RunResult, **kw) -> list[ComparisonResult]:
    """Compare every metric shared by both runs."""
    shared = sorted(set(run_a.metric_names()) & set(run_b.metric_names()))
    return [compare_runs(run_a, run_b, m, **kw) for m in shared]
