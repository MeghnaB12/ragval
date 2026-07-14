"""ragval — rigorous RAG evaluation with confidence intervals,
significance testing, and judge calibration."""

from ragval.types import EvalSample, MetricResult, RagOutput, RunResult, SampleResult

__version__ = "0.1.0"

__all__ = [
    "EvalSample",
    "MetricResult",
    "RagOutput",
    "RunResult",
    "SampleResult",
    "__version__",
]
