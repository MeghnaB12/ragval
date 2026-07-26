"""Cross-judge validation: how well does the cheap Groq judge agree with
Claude on the same human-labeled examples?

This is the payoff of the two-judge setup. The full benchmark is judged by
Groq's llama-3.3-70b (cheap, matches the framework's stated judge). This
script checks that judge against two references on a small labeled set:

  1. HUMAN labels (the ground truth you wrote)
  2. CLAUDE, a different model family — which is the important check, because
     a Llama judge scoring Llama-generated answers can exhibit same-family
     self-preference. Claude has no such stake.

If Groq agrees well with BOTH humans and Claude, the benchmark's absolute
numbers are defensible. If Groq agrees with humans but not Claude, or vice
versa, that disagreement is itself a finding worth reporting.

Usage:
    export GROQ_API_KEY=...
    export ANTHROPIC_API_KEY=...
    python scripts/cross_judge.py benchmarks/calibration/faithfulness.jsonl \
        --metric faithfulness

    # Use Sonnet as the stronger (pricier) reference:
    python scripts/cross_judge.py <file> --metric faithfulness --claude-model claude-sonnet-4-6

Cost: one Groq call + one Claude call per example. ~20 examples on Haiku is
under $0.05.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from ragval.calibration import (  # noqa: E402  # noqa: E402
    calibrate,
    load_calibration_file,
    quadratic_weighted_kappa,
    spearman_correlation,
)
from ragval.judges import ClaudeJudge, GroqJudge  # noqa: E402
from ragval.metrics import METRIC_REGISTRY  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path, help="JSONL of human-labeled CalibrationExamples")
    parser.add_argument("--metric", default="faithfulness")
    parser.add_argument("--claude-model", default="claude-haiku-4-5")
    args = parser.parse_args()

    if args.metric not in METRIC_REGISTRY:
        raise SystemExit(f"Unknown metric '{args.metric}'. Known: {sorted(METRIC_REGISTRY)}")

    examples = load_calibration_file(args.file)
    metric_cls = METRIC_REGISTRY[args.metric]

    print(f"Calibrating '{args.metric}' on {len(examples)} labeled examples.\n")

    print("=== Groq (llama-3.3-70b) vs human ===")
    groq = GroqJudge(min_seconds_between_calls=2.5)
    groq_report = calibrate(metric_cls(), groq, examples)
    print(groq_report)
    print()

    print(f"=== Claude ({args.claude_model}) vs human ===")
    claude = ClaudeJudge(model=args.claude_model)
    claude_report = calibrate(metric_cls(), claude, examples)
    print(claude_report)
    print()

    # Cross-judge agreement, on the examples both judges scored successfully.
    print("=== Groq vs Claude (the self-preference check) ===")
    # judge_scores are ordered by successfully-parsed examples; align on the
    # shorter list defensively (a judge may fail to parse a few).
    n = min(len(groq_report.judge_scores), len(claude_report.judge_scores))
    gj = np.array(groq_report.judge_scores[:n])
    cj = np.array(claude_report.judge_scores[:n])
    diffs = np.abs(gj - cj)
    print(f"  n compared:         {n}")
    print(f"  exact agreement:    {(diffs == 0).mean():.2f}")
    print(f"  within-1 agreement: {(diffs <= 1).mean():.2f}")
    print(
        f"  weighted kappa:     "
        f"{quadratic_weighted_kappa([round(x) for x in gj], [round(x) for x in cj]):.3f}"
    )
    print(f"  spearman:           {spearman_correlation(gj, cj):.3f}")
    print(f"  mean(Groq - Claude):{(gj - cj).mean():+.2f}  (positive = Groq more lenient)")
    print()

    print("--- verdict ---")
    if groq_report.usable and claude_report.usable:
        print("Both judges track human labels. Groq's benchmark numbers are defensible;")
        print("report the Claude cross-check as corroboration.")
    elif not groq_report.usable and claude_report.usable:
        print("Groq disagrees with humans but Claude agrees — consider Claude as the")
        print("primary judge, or report Groq numbers only as relative comparisons.")
    else:
        print("Calibration is weak. Revisit the metric rubric or the human labels")
        print("before trusting absolute scores from either judge.")


if __name__ == "__main__":
    main()
