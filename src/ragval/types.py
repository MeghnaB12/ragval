"""Core data types used throughout ragval.

These are the contracts. Everything else in the framework consumes or produces them.
Keep them small, well-documented, and serializable to JSON.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class EvalSample(BaseModel):
    """A single question + ground truth from an evaluation dataset.

    `ground_truth_contexts` is the list of context chunks (or chunk IDs) that
    are known to contain the answer. Used for context recall scoring.
    """

    id: str
    question: str
    ground_truth_answer: str
    ground_truth_contexts: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RagOutput(BaseModel):
    """The output of a RAG system for a single question.

    A RAG system in ragval is just a callable that takes a question string
    and returns one of these.
    """

    answer: str
    retrieved_contexts: list[str]
    metadata: dict[str, Any] = Field(default_factory=dict)


class MetricResult(BaseModel):
    """The result of running one metric on one (sample, output) pair.

    `score` is normalized to [0, 1] regardless of the underlying scale.
    `raw_score` preserves whatever the judge actually returned (e.g. 1-5).
    `reasoning` is the judge's explanation, kept for debugging and the dashboard.
    """

    metric_name: str
    score: float
    raw_score: float | None = None
    reasoning: str = ""
    judge_model: str = ""
    cost_usd: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class SampleResult(BaseModel):
    """All metric results for a single sample, plus the RAG output that produced them."""

    sample_id: str
    rag_output: RagOutput
    metrics: dict[str, MetricResult]


class RunResult(BaseModel):
    """The full result of evaluating one RAG system on one dataset."""

    run_id: str
    config_name: str
    dataset_name: str
    timestamp: datetime
    samples: list[SampleResult]
    total_cost_usd: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)

    def metric_scores(self, metric_name: str) -> list[float]:
        """Return all per-sample scores for a given metric."""
        return [s.metrics[metric_name].score for s in self.samples if metric_name in s.metrics]

    def metric_names(self) -> list[str]:
        """All metric names that appear in this run."""
        names: set[str] = set()
        for s in self.samples:
            names.update(s.metrics.keys())
        return sorted(names)
