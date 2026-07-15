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


# Rough output-token sizes, used only by the preflight estimate.
_OUT_GEN, _OUT_JUDGE, _OUT_COT = 30, 60, 200
_JUDGE_CALLS_PER_SAMPLE = 3  # faithfulness, answer_relevance, answer_correctness
_CALLS_PER_SAMPLE = _JUDGE_CALLS_PER_SAMPLE + 1  # + 1 generation


def estimate_cost(config: str, dataset) -> tuple[int, float]:
    """Preflight estimate: (n_requests, n_tokens) for one config.

    Approximates tokens as chars/4. Exists because free-tier quotas are
    usually bound by tokens-per-day, not requests-per-minute — and finding
    that out at hour six of an overnight run is an expensive way to learn it.
    """
    from ragval.metrics import Faithfulness

    total_tokens = 0.0
    is_cot = config.endswith("_cot")
    for sample in dataset:
        captured: list[str] = []

        def capture(prompt: str, _c=captured) -> str:
            _c.append(prompt)
            return "A short answer."

        try:
            output = build_rag_for_sample(config, sample, capture)(sample.question)
        except Exception:
            continue
        total_tokens += len(captured[-1]) / 4 + (_OUT_COT if is_cot else _OUT_GEN)
        ctx = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(output.retrieved_contexts))
        judge_prompt = Faithfulness.PROMPT.format(
            question=sample.question, contexts=ctx or "(none)", answer=output.answer
        )
        # All three judge prompts are of a similar shape; scale the measured one.
        total_tokens += _JUDGE_CALLS_PER_SAMPLE * (len(judge_prompt) / 4 + _OUT_JUDGE)
    return len(dataset) * _CALLS_PER_SAMPLE, total_tokens


def print_preflight(configs: list[str], dataset, rpm: float) -> None:
    """Print the request/token/time budget before spending hours on a run."""
    print("\n=== Preflight estimate ===")
    print(f"{'config':<14} {'requests':>9} {'tokens':>11}")
    total_req = total_tok = 0
    for config in configs:
        req, tok = estimate_cost(config, dataset)
        total_req += req
        total_tok += tok
        print(f"{config:<14} {req:>9,} {tok:>11,.0f}")
    print("-" * 36)
    print(f"{'TOTAL':<14} {total_req:>9,} {total_tok:>11,.0f}")

    hours = (total_req / rpm) / 60 if rpm > 0 else 0
    print(f"\nAt {rpm:.0f} RPM: ~{hours:.1f} hours of wall clock.")
    print(
        "Groq FREE tier (llama-3.3-70b): 30 RPM, 1,000 req/day, 100K tokens/day.\n"
        f"  -> this run needs {total_req / 1000:.1f} days on the request cap "
        f"and {total_tok / 100_000:.1f} days on the TOKEN cap."
    )
    if total_tok > 100_000 or total_req > 1000:
        print(
            "  !! This run does NOT fit in one free-tier day. Either upgrade to the\n"
            "     Groq Developer tier (no daily cap; this run costs roughly "
            f"${total_tok / 1e6 * 0.59:.2f}),\n"
            "     or reduce --n / --configs."
        )
    print()


def make_judge(name: str, min_gap: float = 2.5) -> Judge:
    if name == "groq":
        return GroqJudge(min_seconds_between_calls=min_gap)
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
    parser.add_argument(
        "--rpm",
        type=float,
        default=24.0,
        help="Target requests/min ACROSS generator+judge combined. Default 24 is "
        "safe for Groq's 30 RPM free tier. On the Developer tier (~1000 RPM) "
        "pass something like --rpm 300 to finish far faster.",
    )
    parser.add_argument(
        "--estimate-only",
        action="store_true",
        help="Print the request/token budget and exit without calling any API.",
    )
    parser.add_argument("--yes", action="store_true", help="Skip the pre-run confirmation prompt.")
    args = parser.parse_args()

    dataset = load_hotpotqa()
    if args.n:
        dataset = dataset[: args.n]
    print(f"Dataset: {len(dataset)} samples. Configs: {args.configs}. Judge: {args.judge}")

    print_preflight(args.configs, dataset, args.rpm)
    if args.estimate_only:
        return
    if not args.yes and input("Proceed? [y/N] ").strip().lower() != "y":
        print("Aborted.")
        return

    # The generator and the judge may be the same model; the rate limiter is
    # shared per model_id, so the throttle below applies to their COMBINED rate.
    gap = 60.0 / args.rpm if args.rpm > 0 else 0.0
    generator = GroqJudge(min_seconds_between_calls=gap)
    judge = make_judge(args.judge, gap)

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
