"""Run the HotpotQA benchmark across RAG configurations.

Usage:
    export GROQ_API_KEY=...

    # Always start here — costs nothing, calls no API:
    python scripts/run_benchmark.py --estimate-only --n 150 \
        --configs closed_book bm25_k3 oracle

    # Free-tier plan: cheap generator, 70B judge, 2 judge metrics
    python scripts/run_benchmark.py --n 150 --configs closed_book bm25_k3 oracle

    # Paid tier: everything, fast
    python scripts/run_benchmark.py --n 500 --rpm 300 --yes \
        --generator-model llama-3.3-70b-versatile \
        --metrics faithfulness answer_relevance answer_correctness

Design notes:

- RESUME BY DEFAULT. Every completed sample is appended to
  benchmarks/results/partial/<config>.jsonl immediately. Re-running skips
  samples already done. Free-tier daily caps guarantee interruption, so
  interruption must be cheap.
- Judge calls are also disk-cached, so even deleting a partial file and
  re-running mostly hits cache.
- GENERATOR and JUDGE are separate models. The generator (the RAG system
  under test) defaults to the high-quota llama-3.1-8b-instant; the scarce
  llama-3.3-70b budget is reserved for judging, where model quality
  actually moves the numbers. A weaker generator is scientifically fine —
  arguably better, since it leans on retrieval instead of memorization,
  which is exactly what the benchmark is trying to measure.
- On a DAILY quota error the run checkpoints and exits cleanly (status 2)
  rather than burning retry backoff against a cap that won't clear for hours.

IMPORTANT: partial files are keyed by config name only — not by judge, model,
or dataset size. If you change models or metrics, delete
benchmarks/results/partial/ first or stale samples will silently merge in.
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
from ragval.judges import ClaudeJudge, GeminiJudge, GroqJudge, Judge, _DailyQuotaError
from ragval.metrics import METRIC_REGISTRY, Faithfulness
from ragval.runs import save_run
from ragval.types import RagOutput, RunResult, SampleResult

PARTIAL_DIR = ROOT / "benchmarks" / "results" / "partial"

DEFAULT_METRICS = ["faithfulness", "answer_correctness"]
FREE_METRICS = ["retrieval_recall", "retrieval_precision"]  # deterministic, no API cost

# Rough output-token sizes, used only by the preflight estimate.
_OUT_GEN, _OUT_JUDGE, _OUT_COT = 30, 60, 200


def build_metrics(names: list[str]) -> list:
    """Judge-based metrics from `names`, plus the free deterministic ones."""
    return [METRIC_REGISTRY[n]() for n in list(names) + FREE_METRICS]


def estimate_config(config: str, dataset, n_judge_metrics: int) -> tuple[int, float, float]:
    """Preflight estimate: (n_requests, generator_tokens, judge_tokens).

    Generator and judge tokens are reported separately because they may run on
    different providers with different quotas — e.g. generation on Groq's free
    tier, judging on paid Claude. Approximates tokens as chars/4.
    """
    gen_tokens = 0.0
    judge_tokens = 0.0
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
        gen_tokens += len(captured[-1]) / 4 + (_OUT_COT if is_cot else _OUT_GEN)
        ctx = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(output.retrieved_contexts))
        judge_prompt = Faithfulness.PROMPT.format(
            question=sample.question, contexts=ctx or "(none)", answer=output.answer
        )
        judge_tokens += n_judge_metrics * (len(judge_prompt) / 4 + _OUT_JUDGE)
    return len(dataset) * (n_judge_metrics + 1), gen_tokens, judge_tokens


# Claude judge pricing per 1M tokens, for the cost estimate.
_CLAUDE_PRICING = {"claude-haiku-4-5": (1.0, 5.0), "claude-sonnet-4-6": (3.0, 15.0)}
# Free-tier Groq daily token caps observed on the generator model.
_GROQ_GEN_TPD = {"llama-3.1-8b-instant": 500_000, "llama-3.3-70b-versatile": 100_000}


def print_preflight(
    configs: list[str],
    dataset,
    rpm: float,
    n_judge_metrics: int,
    judge: str = "groq",
    claude_model: str = "claude-haiku-4-5",
    generator_model: str = "llama-3.1-8b-instant",
) -> None:
    """Print the request/token budget before spending hours (or dollars) on a run."""
    print("\n=== Preflight estimate ===")
    print(f"{'config':<14} {'requests':>9} {'gen tok':>11} {'judge tok':>11}")
    total_req, total_gen, total_judge = 0, 0.0, 0.0
    for config in configs:
        req, gen, jud = estimate_config(config, dataset, n_judge_metrics)
        total_req += req
        total_gen += gen
        total_judge += jud
        print(f"{config:<14} {req:>9,} {gen:>11,.0f} {jud:>11,.0f}")
    print("-" * 48)
    print(f"{'TOTAL':<14} {total_req:>9,} {total_gen:>11,.0f} {total_judge:>11,.0f}")

    # Generator: free Groq, bound by tokens-per-day.
    gen_cap = _GROQ_GEN_TPD.get(generator_model, 500_000)
    print(f"\nGENERATOR ({generator_model}, free Groq tier):")
    print(
        f"  {total_gen:,.0f} tokens / {gen_cap:,} per day = "
        f"~{total_gen / gen_cap:.1f} day(s) of generation."
    )
    if total_gen > gen_cap:
        print(
            "  Generation spans multiple days on the free cap. The run checkpoints\n"
            "  and resumes, so this is unattended: one command per day until done."
        )

    # Judge: depends on choice.
    print(f"\nJUDGE ({judge}):")
    if judge == "claude":
        cin, cout = _CLAUDE_PRICING.get(claude_model, (1.0, 5.0))
        # Rough in/out split: judge prompts are input-heavy, ~60-token outputs.
        out_tok = n_judge_metrics * len(dataset) * len(configs) * _OUT_JUDGE / len(configs)
        out_tok = total_judge * 0.12  # ~12% output is a safe over-estimate
        in_tok = total_judge - out_tok
        cost = in_tok / 1e6 * cin + out_tok / 1e6 * cout
        print(
            f"  {claude_model}: ~{total_judge:,.0f} tokens -> ~${cost:.2f}. "
            "No daily cap; runs in one sitting."
        )
    elif judge == "groq":
        print(
            f"  {total_judge:,.0f} tokens / 100,000 per day (llama-3.3-70b free) = "
            f"~{total_judge / 100_000:.0f} DAYS. Groq Developer tier is required to go\n"
            "  faster, but is currently unavailable. Consider --judge claude."
        )
    else:
        print(f"  {total_judge:,.0f} judge tokens.")
    print()


def make_judge(name: str, min_gap: float, claude_model: str = "claude-haiku-4-5") -> Judge:
    if name == "groq":
        return GroqJudge(min_seconds_between_calls=min_gap)
    if name == "gemini":
        return GeminiJudge()
    if name == "claude":
        # Claude is paid with generous limits; it does not need the free-tier
        # Groq throttle. A small gap still avoids hammering the endpoint.
        return ClaudeJudge(model=claude_model, min_seconds_between_calls=0.1)
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


def run_config(
    config: str, dataset, generator_judge: Judge, judge: Judge, metric_names: list[str]
) -> RunResult:
    metrics = build_metrics(metric_names)
    done = load_partial(config)
    todo = [s for s in dataset if s.id not in done]
    print(f"[{config}] {len(done)} done, {len(todo)} to go")

    def generate(prompt: str) -> str:
        return generator_judge.call(prompt).text

    for sample in tqdm(todo, desc=config):
        rag = build_rag_for_sample(config, sample, generate)
        try:
            output = rag(sample.question)
        except _DailyQuotaError:
            raise  # checkpoint and stop; never record a fabricated answer
        except Exception as e:
            output = RagOutput(answer=f"[RAG_ERROR: {e}]", retrieved_contexts=[])

        # A sample is written only once every metric succeeds. A half-scored
        # sample would look complete on resume and silently corrupt the run.
        metric_results = {m.name: m.score(sample, output, judge) for m in metrics}

        sr = SampleResult(sample_id=sample.id, rag_output=output, metrics=metric_results)
        append_partial(config, sr)
        done[sample.id] = sr

    samples = [done[s.id] for s in dataset if s.id in done]
    total_cost = sum(m.cost_usd for sr in samples for m in sr.metrics.values())
    return RunResult(
        run_id=f"{config}-hotpotqa{len(dataset)}",
        config_name=config,
        dataset_name=f"hotpotqa-{len(dataset)}",
        timestamp=datetime.now(timezone.utc),
        samples=samples,
        total_cost_usd=total_cost,
        metadata={"judge": judge.model_id, "generator": generator_judge.model_id},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the HotpotQA benchmark.")
    parser.add_argument("--configs", nargs="+", default=CONFIG_NAMES, choices=CONFIG_NAMES)
    parser.add_argument("--n", type=int, default=None, help="Limit to first N samples")
    parser.add_argument("--judge", default="groq", choices=["groq", "gemini", "claude"])
    parser.add_argument(
        "--claude-model",
        default="claude-haiku-4-5",
        help="Claude model when --judge claude (claude-haiku-4-5 or claude-sonnet-4-6).",
    )
    parser.add_argument(
        "--generator-model",
        default="llama-3.1-8b-instant",
        help="Model that ANSWERS questions (the RAG system under test). Defaults to "
        "the high-quota 8B model so the scarce 70B budget goes to judging.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=DEFAULT_METRICS,
        choices=sorted(set(METRIC_REGISTRY) - set(FREE_METRICS)),
        help=f"Judge-based metrics (default: {DEFAULT_METRICS}). The deterministic "
        f"metrics {FREE_METRICS} are always included and cost nothing.",
    )
    parser.add_argument(
        "--rpm",
        type=float,
        default=24.0,
        help="Target requests/min per model. Default 24 is safe under a 30 RPM free "
        "tier. On a paid tier (~1000 RPM) try --rpm 300.",
    )
    parser.add_argument("--estimate-only", action="store_true", help="Print budget and exit.")
    parser.add_argument("--yes", action="store_true", help="Skip the confirmation prompt.")
    args = parser.parse_args()

    dataset = load_hotpotqa()
    if args.n:
        dataset = dataset[: args.n]
    print(f"Dataset: {len(dataset)} samples. Configs: {args.configs}. Judge: {args.judge}")

    print_preflight(
        args.configs,
        dataset,
        args.rpm,
        len(args.metrics),
        args.judge,
        args.claude_model,
        args.generator_model,
    )
    if args.estimate_only:
        return
    if not args.yes and input("Proceed? [y/N] ").strip().lower() != "y":
        print("Aborted.")
        return

    gap = 60.0 / args.rpm if args.rpm > 0 else 0.0
    generator = GroqJudge(model=args.generator_model, min_seconds_between_calls=gap)
    judge = make_judge(args.judge, gap, args.claude_model)
    print(f"Generator: {generator.model_id} | Judge: {judge.model_id}")
    print(f"Metrics: {args.metrics} (+ free: {FREE_METRICS})")

    summary_rows = []
    for config in args.configs:
        try:
            result = run_config(config, dataset, generator, judge, args.metrics)
        except _DailyQuotaError as e:
            print(
                f"\n\n!! DAILY QUOTA EXHAUSTED while running '{config}'.\n"
                f"   {e}\n"
                "   Completed samples are checkpointed in benchmarks/results/partial/.\n"
                "   Re-run the same command tomorrow and it resumes from here.\n"
            )
            raise SystemExit(2) from e
        path = save_run(result, ROOT / "benchmarks" / "results")
        print(f"[{config}] saved {path}")
        row = {"config": config}
        for m in result.metric_names():
            scores = result.metric_scores(m)
            row[m] = sum(scores) / len(scores) if scores else float("nan")
        summary_rows.append(row)

    print("\n=== Benchmark summary (means; use `ragval compare` for CIs and p-values) ===")
    print(json.dumps(summary_rows, indent=2))


if __name__ == "__main__":
    main()
