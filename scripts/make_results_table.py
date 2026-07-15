"""Generate the README results table from saved runs.

Usage:
    python scripts/make_results_table.py                       # all runs
    python scripts/make_results_table.py --baseline bm25_k3    # compare vs a baseline
    python scripts/make_results_table.py >> README.md

Emits two markdown tables:
  1. Per-config metric means with 95% bootstrap CIs.
  2. Pairwise comparisons against a baseline config, with p-values.

Numbers come straight from benchmarks/results/*.jsonl, so the table can
never drift from the runs it claims to describe.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ragval.runs import list_runs, load_run  # noqa: E402
from ragval.stats import compare_runs, summarize_metric  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=ROOT / "benchmarks" / "results")
    parser.add_argument(
        "--baseline",
        default=None,
        help="Config to compare every other config against (e.g. bm25_k3)",
    )
    args = parser.parse_args()

    headers = list_runs(args.results_dir)
    if not headers:
        print(f"No runs found in {args.results_dir}", file=sys.stderr)
        raise SystemExit(1)

    # Newest run wins if a config was run more than once.
    runs = {}
    for h in headers:
        runs.setdefault(h["config_name"], load_run(h["_path"]))

    all_metrics: list[str] = sorted({m for r in runs.values() for m in r.metric_names()})
    configs = sorted(runs)

    n = len(next(iter(runs.values())).samples)
    print(f"## Results — HotpotQA-{n}\n")
    judge = next(iter(runs.values())).metadata.get("judge", "unknown")
    print(f"Judge: `{judge}`. Every cell is mean [95% bootstrap CI], n={n}.\n")

    # Table 1: means + CIs
    print("| config | " + " | ".join(all_metrics) + " |")
    print("|---" * (len(all_metrics) + 1) + "|")
    for config in configs:
        run = runs[config]
        cells = []
        for m in all_metrics:
            if m not in run.metric_names():
                cells.append("—")
                continue
            s = summarize_metric(run, m)
            cells.append(f"{s.mean:.3f} [{s.ci_low:.2f}–{s.ci_high:.2f}]")
        print(f"| `{config}` | " + " | ".join(cells) + " |")

    # Table 2: comparisons vs baseline
    if args.baseline:
        if args.baseline not in runs:
            print(f"\nBaseline '{args.baseline}' not found in {configs}", file=sys.stderr)
            raise SystemExit(1)
        base = runs[args.baseline]
        print(f"\n### Compared against `{args.baseline}` (paired, per-sample)\n")
        print("| config | metric | diff | 95% CI | p (boot) | p (perm) | verdict |")
        print("|---|---|---|---|---|---|---|")
        for config in configs:
            if config == args.baseline:
                continue
            run = runs[config]
            for m in sorted(set(run.metric_names()) & set(base.metric_names())):
                try:
                    c = compare_runs(run, base, m)
                except ValueError:
                    continue
                verdict = "**significant**" if c.significant else "not significant"
                print(
                    f"| `{config}` | {m} | {c.mean_diff:+.3f} | "
                    f"[{c.diff_ci_low:+.3f}, {c.diff_ci_high:+.3f}] | "
                    f"{c.p_value_bootstrap:.4f} | {c.p_value_permutation:.4f} | {verdict} |"
                )


if __name__ == "__main__":
    main()
