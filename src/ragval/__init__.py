"""ragval — rigorous RAG evaluation with confidence intervals,
significance testing, and judge calibration."""


def _load_dotenv() -> None:
    """Load API keys from a .env file, so scripts and the CLI pick them up
    without the user having to `export` or `source` anything each session.

    Searches the current directory and its parents for a `.env` file (so it
    works whether ragval is run from the repo root or a subdirectory).
    Existing environment variables are NOT overridden — an explicit
    `export GROQ_API_KEY=...` always wins over the file. Silently does
    nothing if python-dotenv isn't installed or no .env exists.
    """
    try:
        from dotenv import find_dotenv, load_dotenv
    except ImportError:
        return
    path = find_dotenv(usecwd=True)
    if path:
        load_dotenv(path, override=False)


_load_dotenv()

from ragval.types import (  # noqa: E402
    EvalSample,
    MetricResult,
    RagOutput,
    RunResult,
    SampleResult,
)

__version__ = "0.1.0"

__all__ = [
    "EvalSample",
    "MetricResult",
    "RagOutput",
    "RunResult",
    "SampleResult",
    "__version__",
]
