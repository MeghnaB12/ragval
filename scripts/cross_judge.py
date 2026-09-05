"""Cross-judge calibration on the same human-labeled examples.

The published HotpotQA-500 benchmark uses Claude Haiku 4.5 as the primary
judge and Llama 3.3 70B as an independent cross-check. This script evaluates
both judges against the same human labels and also reports judge-to-judge
agreement.

The goal is diagnostic, not to prove that either model is ground truth: a small
human-labeled set can reveal bias, weak ordinal agreement, or model-family
disagreement that should qualify how benchmark scores are interpreted.

Usage:
    export GROQ_API_KEY=...
    export ANTHROPIC_API_KEY=...
    python scripts/cross_judge.py benchmarks/calibration/faithfulness.jsonl \
        --metric faithfulness

    # Use Sonnet as a stronger (pricier) Claude reference:
    python scripts/cross_judge.py <file> --metric faithfulness \
        --claude-model claude-sonnet-4-6
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from ragval.calibration import (  # noqa: E402
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

    print(f"=== Claude ({args.claude_model}) vs human ===")
    claude = ClaudeJudge(model=args.claude_model)
    claude_report = calibrate(metric_cls(), claude, examples)
    print(claude_report)
    print()

    print("=== Llama 3.3 70B (Groq) vs human ===")
    groq = GroqJudge(min_seconds_between_calls=2.5)
    groq_report = calibrate(metric_cls(), groq, examples)
    print(groq_report)
    print()

    print("=== Claude vs Llama (cross-judge diagnostic) ===")
    # Reports preserve successfully parsed scores in input order. Align to the
    # common prefix defensively if one provider has a parse failure.
    n = min(len(groq_report.judge_scores), len(claude_report.judge_scores))
    gj = np.array(groq_report.judge_scores[:n])
    cj = np.array(claude_report.judge_scores[:n])
    diffs = np.abs(gj - cj)
    print(f"  n compared:          {n}")
    print(f"  exact agreement:     {(diffs == 0).mean():.2f}")
    print(f"  within-1 agreement:  {(diffs <= 1).mean():.2f}")
    print(
        "  weighted kappa:      "
        f"{quadratic_weighted_kappa([round(x) for x in cj], [round(x) for x in gj]):.3f}"
    )
    print(f"  spearman:            {spearman_correlation(cj, gj):.3f}")
    print(f"  mean(Claude - Llama):{(cj - gj).mean():+.2f}")
    print()

    print("--- interpretation ---")
    print(
        "Treat these values as calibration evidence, not certification. Stronger "
        "agreement supports more confidence in relative judge-based comparisons; "
        "weak agreement or systematic bias is a reason to qualify absolute scores, "
        "revisit the rubric, or collect more human labels."
    )


if __name__ == "__main__":
    main()
