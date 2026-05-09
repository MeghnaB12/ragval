"""Runner — orchestrates a RAG system + metrics over a dataset.

A RAG system in ragval is just a callable: `(question: str) -> RagOutput`.
This makes it trivial to plug in anything: LangChain chains, raw functions,
LlamaIndex pipelines, or your own code.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from datetime import datetime, timezone

from tqdm import tqdm

from ragval.judges import Judge
from ragval.metrics import Metric
from ragval.types import EvalSample, RagOutput, RunResult, SampleResult

RagSystem = Callable[[str], RagOutput]


def run_eval(
    rag_system: RagSystem,
    dataset: list[EvalSample],
    metrics: list[Metric],
    judge: Judge,
    config_name: str = "unnamed",
    dataset_name: str = "unnamed",
    show_progress: bool = True,
) -> RunResult:
    """Evaluate a RAG system on a dataset with the given metrics.

    Args:
        rag_system: callable taking a question string, returning a RagOutput
        dataset: list of EvalSample
        metrics: list of Metric instances
        judge: Judge instance used by all metrics
        config_name: identifier for this RAG configuration
        dataset_name: identifier for the dataset
        show_progress: whether to display a progress bar
    """
    run_id = f"{config_name}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    sample_results: list[SampleResult] = []
    total_cost = 0.0

    iterator = tqdm(dataset, desc=f"eval {config_name}", disable=not show_progress)
    for sample in iterator:
        try:
            output = rag_system(sample.question)
        except Exception as e:
            output = RagOutput(answer=f"[RAG_ERROR: {e}]", retrieved_contexts=[])

        metric_results = {}
        for metric in metrics:
            result = metric.score(sample, output, judge)
            metric_results[metric.name] = result
            total_cost += result.cost_usd

        sample_results.append(
            SampleResult(
                sample_id=sample.id,
                rag_output=output,
                metrics=metric_results,
            )
        )

    return RunResult(
        run_id=run_id,
        config_name=config_name,
        dataset_name=dataset_name,
        timestamp=datetime.now(timezone.utc),
        samples=sample_results,
        total_cost_usd=total_cost,
    )
