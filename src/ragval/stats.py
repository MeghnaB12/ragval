"""Statistical summaries and paired comparisons for evaluation runs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ragval.types import RunResult

DEFAULT_CONFIDENCE = 0.95
DEFAULT_N_RESAMPLES = 10_000
DEFAULT_SEED = 42


@dataclass
class MetricSummary:
    metric_name: str
    n: int
    mean: float
    ci_low: float
    ci_high: float
    confidence: float
    std: float


@dataclass
class RunComparison:
    metric_name: str
    n: int
    mean_a: float
    mean_b: float
    mean_diff: float
    diff_ci_low: float
    diff_ci_high: float
    p_value_bootstrap: float
    p_value_permutation: float
    significant: bool


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
    bounds = np.asarray(np.quantile(boot_means, [alpha / 2, 1 - alpha / 2]), dtype=float)
    lo = float(bounds[0])
    hi = float(bounds[1])
    return float(arr.mean()), lo, hi


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
    bounds = np.asarray(np.quantile(boot_means, [alpha / 2, 1 - alpha / 2]), dtype=float)
    lo = float(bounds[0])
    hi = float(bounds[1])

    # Shift to null: center bootstrap distribution at 0
    null_dist = boot_means - observed
    p = float((np.abs(null_dist) >= abs(observed)).mean())
    # Avoid reporting p=0 from a finite resample count
    p = max(p, 1.0 / n_resamples)
    return observed, lo, hi, p


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
    if observed == 0:
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
) -> RunComparison:
    """Paired comparison of one metric between two runs."""
    a, b = _paired_scores(run_a, run_b, metric_name)
    diff, lo, hi, p_boot = paired_bootstrap_test(
        a, b, confidence=confidence, n_resamples=n_resamples, seed=seed
    )
    p_perm = permutation_test(a, b, n_resamples=n_resamples, seed=seed)
    return RunComparison(
        metric_name=metric_name,
        n=len(a),
        mean_a=float(a.mean()),
        mean_b=float(b.mean()),
        mean_diff=diff,
        diff_ci_low=lo,
        diff_ci_high=hi,
        p_value_bootstrap=p_boot,
        p_value_permutation=p_perm,
        significant=p_boot < (1 - confidence),
    )


def compare_all_metrics(
    run_a: RunResult,
    run_b: RunResult,
    confidence: float = DEFAULT_CONFIDENCE,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> list[RunComparison]:
    """Compare every metric shared by two runs."""
    common = sorted(set(run_a.metric_names()) & set(run_b.metric_names()))
    return [
        compare_runs(
            run_a,
            run_b,
            metric,
            confidence=confidence,
            n_resamples=n_resamples,
            seed=seed,
        )
        for metric in common
    ]
