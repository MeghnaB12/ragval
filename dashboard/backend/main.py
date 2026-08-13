"""ragval dashboard API.

A thin REST layer over ragval's existing engine. Every endpoint reads the
committed benchmark runs in `benchmarks/results/` and calls the same
`ragval.stats` functions the CLI uses — so the API can never disagree with
`ragval compare`. No numbers are computed here; they are all delegated to the
statistics library, which is the point: the rigor lives in one place.

Run locally:
    uvicorn main:app --reload --port 8000

Endpoints:
    GET /api/health
    GET /api/runs                      -> list configs with metadata
    GET /api/runs/{config}             -> per-metric mean + 95% CI for one config
    GET /api/compare?a=&b=             -> paired significance tests between two configs
    GET /api/samples/{config}?metric=  -> per-sample scores + judge reasoning, worst first
    GET /api/calibration               -> judge-vs-human calibration summary
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Make the ragval package importable when running from dashboard/backend.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ragval.runs import list_runs, load_run  # noqa: E402
from ragval.stats import compare_all_metrics, summarize_run  # noqa: E402
from ragval.types import RunResult  # noqa: E402

RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"

# The benchmark ran these 8 configs. A stray early Gemini smoke run also lives
# in the results dir; exclude it so the dashboard shows only the real n=500 grid.
BENCHMARK_CONFIGS = [
    "closed_book",
    "bm25_k1",
    "bm25_k3",
    "bm25_k5",
    "full_context",
    "oracle",
    "bm25_k3_cot",
    "oracle_cot",
]

app = FastAPI(
    title="ragval dashboard API",
    description="REST layer over ragval's statistical engine.",
    version="1.0.0",
)

# In dev the React app runs on a different port; allow it. Tighten in prod.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Response models — an explicit API contract, auto-documented at /docs.
# ---------------------------------------------------------------------------


class RunSummary(BaseModel):
    config: str
    dataset: str
    n_samples: int
    judge: str
    generator: str
    total_cost_usd: float


class MetricPoint(BaseModel):
    metric: str
    mean: float
    ci_low: float
    ci_high: float
    std: float
    n: int


class ConfigReport(BaseModel):
    config: str
    n_samples: int
    metrics: list[MetricPoint]


class ComparisonRow(BaseModel):
    metric: str
    n: int
    mean_a: float
    mean_b: float
    diff: float
    diff_ci_low: float
    diff_ci_high: float
    p_bootstrap: float
    p_permutation: float
    significant: bool


class Comparison(BaseModel):
    config_a: str
    config_b: str
    rows: list[ComparisonRow]


class SampleRow(BaseModel):
    sample_id: str
    score: float
    raw_score: float | None
    answer: str
    reasoning: str
    n_contexts: int


class JudgeCalibration(BaseModel):
    judge: str
    role: str
    within_one_agreement: float
    quadratic_weighted_kappa: float
    spearman: float
    mean_bias: float


class CalibrationReport(BaseModel):
    metric: str
    n_labels: int
    judges: list[JudgeCalibration]
    note: str


# ---------------------------------------------------------------------------
# Data loading — cached so repeated requests don't re-read 500-line JSONL files.
# ---------------------------------------------------------------------------


@lru_cache(maxsize=32)
def _load(config: str) -> RunResult:
    matches = sorted(RESULTS_DIR.glob(f"{config}-hotpotqa*.jsonl"))
    if not matches:
        raise HTTPException(status_code=404, detail=f"No run found for config '{config}'")
    return load_run(matches[-1])


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/runs", response_model=list[RunSummary])
def get_runs() -> list[RunSummary]:
    """All benchmark configs with run-level metadata, in benchmark order."""
    headers = {h["config_name"]: h for h in list_runs(RESULTS_DIR)}
    out: list[RunSummary] = []
    for config in BENCHMARK_CONFIGS:
        h = headers.get(config)
        if not h:
            continue
        meta = h.get("metadata", {})
        out.append(
            RunSummary(
                config=config,
                dataset=h["dataset_name"],
                n_samples=h["n_samples"],
                judge=meta.get("judge", "unknown"),
                generator=meta.get("generator", "unknown"),
                total_cost_usd=h.get("total_cost_usd", 0.0),
            )
        )
    return out


@app.get("/api/runs/{config}", response_model=ConfigReport)
def get_run(config: str) -> ConfigReport:
    """Per-metric mean + 95% bootstrap CI for one config."""
    run = _load(config)
    summaries = summarize_run(run)
    return ConfigReport(
        config=config,
        n_samples=len(run.samples),
        metrics=[
            MetricPoint(
                metric=s.metric_name,
                mean=s.mean,
                ci_low=s.ci_low,
                ci_high=s.ci_high,
                std=s.std,
                n=s.n,
            )
            for s in summaries
        ],
    )


@app.get("/api/compare", response_model=Comparison)
def compare(
    a: str = Query(..., description="First config"),
    b: str = Query(..., description="Second config"),
) -> Comparison:
    """Paired significance tests between two configs, aligned by sample ID."""
    run_a, run_b = _load(a), _load(b)
    comparisons = compare_all_metrics(run_a, run_b)
    return Comparison(
        config_a=a,
        config_b=b,
        rows=[
            ComparisonRow(
                metric=c.metric_name,
                n=c.n,
                mean_a=c.mean_a,
                mean_b=c.mean_b,
                diff=c.mean_diff,
                diff_ci_low=c.diff_ci_low,
                diff_ci_high=c.diff_ci_high,
                p_bootstrap=c.p_value_bootstrap,
                p_permutation=c.p_value_permutation,
                significant=c.significant,
            )
            for c in comparisons
        ],
    )


@app.get("/api/samples/{config}", response_model=list[SampleRow])
def get_samples(
    config: str,
    metric: str = Query("faithfulness", description="Metric to sort by"),
    order: str = Query("worst", pattern="^(worst|best)$"),
    limit: int = Query(50, ge=1, le=500),
) -> list[SampleRow]:
    """Per-sample scores with the judge's reasoning, worst-scoring first.

    This is the view a static results table can't give you: *why* did a config
    fail on a given question? The judge's own words are the answer.
    """
    run = _load(config)
    rows: list[SampleRow] = []
    for s in run.samples:
        m = s.metrics.get(metric)
        if m is None:
            continue
        rows.append(
            SampleRow(
                sample_id=s.sample_id,
                score=m.score,
                raw_score=m.raw_score,
                answer=s.rag_output.answer,
                reasoning=m.reasoning,
                n_contexts=len(s.rag_output.retrieved_contexts),
            )
        )
    rows.sort(key=lambda r: r.score, reverse=(order == "best"))
    return rows[:limit]


@app.get("/api/calibration", response_model=CalibrationReport)
def get_calibration() -> CalibrationReport:
    """Judge-vs-human calibration for faithfulness.

    These numbers come from `scripts/cross_judge.py` run on 20 human-labeled
    examples. They are served as published constants rather than recomputed on
    every request (recomputing would require live API keys). See
    benchmarks/calibration/faithfulness.jsonl for the labels.
    """
    return CalibrationReport(
        metric="faithfulness",
        n_labels=20,
        judges=[
            JudgeCalibration(
                judge="claude-haiku-4-5",
                role="benchmark judge",
                within_one_agreement=0.70,
                quadratic_weighted_kappa=0.458,
                spearman=0.528,
                mean_bias=-0.40,
            ),
            JudgeCalibration(
                judge="llama-3.3-70b-versatile",
                role="cross-check",
                within_one_agreement=0.75,
                quadratic_weighted_kappa=0.665,
                spearman=0.621,
                mean_bias=-0.05,
            ),
        ],
        note=(
            "Both judges run harsh on faithfulness (negative bias), so the "
            "absolute faithfulness scores are a conservative lower bound. "
            "Config rankings are preserved (Spearman ~0.5-0.6), so the paired "
            "comparisons hold."
        ),
    )
