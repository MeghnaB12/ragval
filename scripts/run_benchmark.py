"""Run the full HotpotQA-500 benchmark: 8 configs x 500 questions.

Usage:
    export GROQ_API_KEY=...
    python scripts/run_benchmark.py                       # all 8 configs
    python scripts/run_benchmark.py --configs bm25_k3 oracle
    python scripts/run_benchmark.py --n 50                # subset for a dry run
    python scripts/run_benchmark.py --judge gemini        # requires GEMINI_API_KEY

Design notes:

- RESUME BY DEFAULT. Every completed sample is appended to
  benchmarks/results/partial/<config>.jsonl immediately. Re-running the
  script skips samples already done. Free-tier rate limits mean the full
  run takes hours; it WILL be interrupted, and that must be cheap.
- Judge calls are also disk-cached (see ragval.judges), so even deleting a
  partial file and re-running mostly hits cache.
- Generation and judging use separate cache-friendly prompts; the Groq
  generator is throttled to stay under the free-tier RPM.

Budget estimate (Groq free tier, ~24 RPM):
  Per sample: 1 generation + 3 judge calls (faithfulness, answer_relevance,
  answer_correctness) + 2 deterministic metrics (free) = 4 API calls.
  8 configs x 500 samples x 4 calls = 16,000 calls ≈ 11 hours of throttled
  runtime end-to-end. Run configs in separate sessions, or use --n 100
  first and extend.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tqdm import tqdm

from ragval.configs import CONFIG_NAMES, build_rag_for_sample
from ragval.datasets import load_hotpotqa
from ragval.judges import GeminiJudge, GroqJudge, Judge
from ragval.metrics import (
    AnswerCorrectness,
    AnswerRelevance,
    Faithfulness,
    RetrievalPrecision,
    RetrievalRecall,
)
from ragval.runs import save_run
from ragval.types import RagOutput, RunResult, SampleResult

PARTIAL_DIR = ROOT / "benchmarks" / "results" / "partial"


def make_judge(name: str) -> Judge:
    if name == "groq":
        return GroqJudge()
    if name == "gemini":
        return GeminiJudge()
    raise ValueError(f"Unknown judge: {name}")


def load_partial(config: str) -> dict[str, SampleResult]:
    """Load already-completed samples for a config from the partial file."""
    path = PARTIAL_DIR / f"{config}.jsonl"
    done: dict[str, SampleResult] = {}
    if not path.exists():
        return done
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                sr = SampleResult.model_validate_json(line)
                done[sr.sample_id] = sr
            except Exception:
                continue  # a torn write from an interrupt; the sample re-runs
    return done


def append_partial(config: str, sr: SampleResult) -> None:
    PARTIAL_DIR.mkdir(parents=True, exist_ok=True)
    with (PARTIAL_DIR / f"{config}.jsonl").open("a") as f:
        f.write(sr.model_dump_json() + "\n")


def run_config(config: str, dataset, generator_judge: Judge, judge: Judge) -> RunResult:
    metrics = [
        Faithfulness(),
        AnswerRelevance(),
        AnswerCorrectness(),
        RetrievalRecall(),
        RetrievalPrecision(),
    ]

    done = load_partial(config)
    todo = [s for s in dataset if s.id not in done]
    print(f"[{config}] {len(done)} done, {len(todo)} to go")

    def generate(prompt: str) -> str:
        return generator_judge.call(prompt).text

    for sample in tqdm(todo, desc=config):
        rag = build_rag_for_sample(config, sample, generate)
        try:
            output = rag(sample.question)
        except Exception as e:
            output = RagOutput(answer=f"[RAG_ERROR: {e}]", retrieved_contexts=[])

        metric_results = {}
        for metric in metrics:
            metric_results[metric.name] = metric.score(sample, output, judge)

        sr = SampleResult(sample_id=sample.id, rag_output=output, metrics=metric_results)
        append_partial(config, sr)
        done[sample.id] = sr

    # Assemble in dataset order
    samples = [done[s.id] for s in dataset if s.id in done]
    total_cost = sum(m.cost_usd for sr in samples for m in sr.metrics.values())
    return RunResult(
        run_id=f"{config}-hotpotqa{len(dataset)}",
        config_name=config,
        dataset_name=f"hotpotqa-{len(dataset)}",
        timestamp=datetime.now(timezone.utc),
        samples=samples,
        total_cost_usd=total_cost,
        metadata={"judge": judge.model_id, "generator": "llama-3.3-70b-versatile"},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", default=CONFIG_NAMES, choices=CONFIG_NAMES)
    parser.add_argument("--n", type=int, default=None, help="Limit to first N samples")
    parser.add_argument("--judge", default="groq", choices=["groq", "gemini"])
    args = parser.parse_args()

    dataset = load_hotpotqa()
    if args.n:
        dataset = dataset[: args.n]
    print(f"Dataset: {len(dataset)} samples. Configs: {args.configs}. Judge: {args.judge}")

    generator = GroqJudge()  # generator model: llama-3.3-70b-versatile
    judge = make_judge(args.judge)

    summary_rows = []
    for config in args.configs:
        result = run_config(config, dataset, generator, judge)
        path = save_run(result, ROOT / "benchmarks" / "results")
        print(f"[{config}] saved {path}")
        row = {"config": config}
        for m in result.metric_names():
            scores = result.metric_scores(m)
            row[m] = sum(scores) / len(scores) if scores else float("nan")
        summary_rows.append(row)

    print("\n=== Benchmark summary (means; run `ragval compare` for CIs and p-values) ===")
    print(json.dumps(summary_rows, indent=2))


if __name__ == "__main__":
    main()
