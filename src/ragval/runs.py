"""Run persistence.

Every completed evaluation run is saved as a single JSONL file under
`benchmarks/results/<run_id>.jsonl`. Each line is one SampleResult.
A header line (the first line) carries run-level metadata.

This format is intentionally simple:
- One file per run = trivial to share, version, and load lazily.
- JSONL = grep-able, human-readable, streamable, append-friendly.
- The header line is distinguishable by its `_kind: "header"` field.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from pydantic import TypeAdapter

from ragval.types import RunResult, SampleResult

DEFAULT_RESULTS_DIR = Path("benchmarks") / "results"

_SAMPLE_ADAPTER = TypeAdapter(SampleResult)


def save_run(result: RunResult, results_dir: Path | str | None = None) -> Path:
    """Save a RunResult to disk as JSONL. Returns the file path."""
    base = Path(results_dir) if results_dir else DEFAULT_RESULTS_DIR
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"{result.run_id}.jsonl"

    header = {
        "_kind": "header",
        "run_id": result.run_id,
        "config_name": result.config_name,
        "dataset_name": result.dataset_name,
        "timestamp": result.timestamp.isoformat(),
        "total_cost_usd": result.total_cost_usd,
        "n_samples": len(result.samples),
        "metric_names": result.metric_names(),
        "metadata": result.metadata,
    }

    with path.open("w") as f:
        f.write(json.dumps(header) + "\n")
        for sr in result.samples:
            f.write(sr.model_dump_json() + "\n")

    return path


def load_run(path: Path | str) -> RunResult:
    """Load a RunResult from a JSONL file."""
    p = Path(path)
    samples: list[SampleResult] = []
    header: dict | None = None
    with p.open() as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if i == 0 and data.get("_kind") == "header":
                header = data
                continue
            samples.append(_SAMPLE_ADAPTER.validate_python(data))

    if header is None:
        raise ValueError(f"{p} has no header line")

    return RunResult(
        run_id=header["run_id"],
        config_name=header["config_name"],
        dataset_name=header["dataset_name"],
        timestamp=datetime.fromisoformat(header["timestamp"]),
        samples=samples,
        total_cost_usd=header.get("total_cost_usd", 0.0),
        metadata=header.get("metadata", {}),
    )


def list_runs(results_dir: Path | str | None = None) -> list[dict]:
    """List all runs in the results directory, returning header info only.

    Useful for dashboards and CLIs that want to show a run picker without
    loading every sample.
    """
    base = Path(results_dir) if results_dir else DEFAULT_RESULTS_DIR
    if not base.exists():
        return []
    headers: list[dict] = []
    for p in sorted(base.glob("*.jsonl")):
        with p.open() as f:
            first = f.readline().strip()
            if not first:
                continue
            data = json.loads(first)
            if data.get("_kind") != "header":
                continue
            data["_path"] = str(p)
            headers.append(data)
    # Newest first
    headers.sort(key=lambda h: h.get("timestamp", ""), reverse=True)
    return headers


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
