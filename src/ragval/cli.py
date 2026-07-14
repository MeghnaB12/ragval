"""ragval CLI.

Commands:
    ragval runs                     list saved benchmark runs
    ragval report RUN               per-metric means + 95% bootstrap CIs
    ragval compare RUN_A RUN_B      paired significance tests between two runs
    ragval calibrate FILE           judge-vs-human agreement on a labeled file
    ragval smoke                    run the smoke test
    ragval version                  print version
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="ragval — rigorous RAG evaluation.")
console = Console()

DEFAULT_RESULTS = Path("benchmarks") / "results"


def _resolve_run(name_or_path: str, results_dir: Path) -> Path:
    """Accept either a path to a .jsonl file or a run/config name in results_dir."""
    p = Path(name_or_path)
    if p.exists():
        return p
    candidate = results_dir / f"{name_or_path}.jsonl"
    if candidate.exists():
        return candidate
    # Prefix match on run files (run ids embed the config name)
    matches = sorted(results_dir.glob(f"{name_or_path}*.jsonl"))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise typer.BadParameter(f"'{name_or_path}' is ambiguous: {[m.name for m in matches]}")
    raise typer.BadParameter(f"No run found for '{name_or_path}' in {results_dir}")


@app.command()
def runs(results_dir: Path = typer.Option(DEFAULT_RESULTS, help="Results directory")):
    """List saved runs."""
    from ragval.runs import list_runs

    headers = list_runs(results_dir)
    if not headers:
        console.print(f"No runs found in {results_dir}")
        raise typer.Exit()

    table = Table(title="ragval runs")
    table.add_column("run_id")
    table.add_column("config")
    table.add_column("dataset")
    table.add_column("n", justify="right")
    table.add_column("cost $", justify="right")
    table.add_column("timestamp")
    for h in headers:
        table.add_row(
            h["run_id"],
            h["config_name"],
            h["dataset_name"],
            str(h["n_samples"]),
            f"{h.get('total_cost_usd', 0):.4f}",
            h["timestamp"][:19],
        )
    console.print(table)


@app.command()
def report(
    run: str = typer.Argument(..., help="Run id, config name, or path to a run .jsonl"),
    results_dir: Path = typer.Option(DEFAULT_RESULTS),
    confidence: float = typer.Option(0.95),
):
    """Per-metric means with bootstrap confidence intervals for one run."""
    from ragval.runs import load_run
    from ragval.stats import summarize_run

    result = load_run(_resolve_run(run, results_dir))
    summaries = summarize_run(result, confidence=confidence)

    table = Table(title=f"{result.config_name} on {result.dataset_name} (n={len(result.samples)})")
    table.add_column("metric")
    table.add_column("mean", justify="right")
    table.add_column(f"{int(confidence * 100)}% CI", justify="right")
    table.add_column("std", justify="right")
    for s in summaries:
        table.add_row(
            s.metric_name, f"{s.mean:.3f}", f"[{s.ci_low:.3f}, {s.ci_high:.3f}]", f"{s.std:.3f}"
        )
    console.print(table)


@app.command()
def compare(
    run_a: str = typer.Argument(..., help="First run (id, config name, or path)"),
    run_b: str = typer.Argument(..., help="Second run"),
    metric: str = typer.Option(None, help="Compare only this metric (default: all shared)"),
    results_dir: Path = typer.Option(DEFAULT_RESULTS),
    confidence: float = typer.Option(0.95),
):
    """Paired significance tests between two runs on the same dataset."""
    from ragval.runs import load_run
    from ragval.stats import compare_all_metrics, compare_runs

    a = load_run(_resolve_run(run_a, results_dir))
    b = load_run(_resolve_run(run_b, results_dir))

    comparisons = (
        [compare_runs(a, b, metric, confidence=confidence)]
        if metric
        else compare_all_metrics(a, b, confidence=confidence)
    )

    table = Table(title=f"{a.config_name} vs {b.config_name} (paired, n per metric shown)")
    table.add_column("metric")
    table.add_column("n", justify="right")
    table.add_column(a.config_name, justify="right")
    table.add_column(b.config_name, justify="right")
    table.add_column("diff", justify="right")
    table.add_column("diff CI", justify="right")
    table.add_column("p (boot)", justify="right")
    table.add_column("p (perm)", justify="right")
    table.add_column("verdict")
    for c in comparisons:
        verdict = "[green]SIGNIFICANT[/green]" if c.significant else "[dim]not significant[/dim]"
        table.add_row(
            c.metric_name,
            str(c.n),
            f"{c.mean_a:.3f}",
            f"{c.mean_b:.3f}",
            f"{c.mean_diff:+.3f}",
            f"[{c.diff_ci_low:+.3f}, {c.diff_ci_high:+.3f}]",
            f"{c.p_value_bootstrap:.4f}",
            f"{c.p_value_permutation:.4f}",
            verdict,
        )
    console.print(table)


@app.command()
def calibrate(
    file: Path = typer.Argument(..., help="JSONL of human-labeled CalibrationExamples"),
    metric: str = typer.Option("faithfulness", help="Metric whose rubric to calibrate"),
    judge: str = typer.Option("groq", help="groq | gemini"),
):
    """Measure judge-vs-human agreement on a labeled calibration file."""
    from ragval.calibration import calibrate as run_calibration
    from ragval.calibration import load_calibration_file
    from ragval.judges import GeminiJudge, GroqJudge
    from ragval.metrics import METRIC_REGISTRY

    if metric not in METRIC_REGISTRY:
        raise typer.BadParameter(f"Unknown metric '{metric}'. Known: {sorted(METRIC_REGISTRY)}")
    judge_obj = GroqJudge() if judge == "groq" else GeminiJudge()
    examples = load_calibration_file(file)
    report_obj = run_calibration(METRIC_REGISTRY[metric](), judge_obj, examples)
    console.print(str(report_obj))
    gate = (
        "[green]USABLE[/green]"
        if report_obj.usable
        else "[red]NOT USABLE — do not trust this judge's absolute numbers[/red]"
    )
    console.print(f"Verdict: {gate}")


@app.command()
def smoke():
    """Run the smoke test."""
    from ragval.smoke_test import main

    main()


@app.command()
def version():
    """Print version."""
    from ragval import __version__

    typer.echo(__version__)


if __name__ == "__main__":
    app()
