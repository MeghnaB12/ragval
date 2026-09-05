"""Smoke test: run ragval against a small slice of HotpotQA.

Run with:
    GROQ_API_KEY=... python -m ragval.smoke_test
    GEMINI_API_KEY=... python -m ragval.smoke_test
    python -m ragval.smoke_test --toy
    python -m ragval.smoke_test --n 10

The HotpotQA dataset must be prepared first:
    python scripts/prepare_hotpotqa.py

With no provider key, the command falls back to MockJudge so the execution
path can still be checked without spending API quota.
"""

from __future__ import annotations

import argparse
import os

from rank_bm25 import BM25Okapi
from rich.console import Console
from rich.table import Table

from ragval.datasets import load_hotpotqa, load_toy_dataset, toy_corpus
from ragval.judges import GeminiJudge, GroqJudge, MockJudge
from ragval.metrics import AnswerCorrectness, AnswerRelevance, Faithfulness
from ragval.retrieval import per_question_bm25_rag
from ragval.runner import run_eval
from ragval.types import RagOutput


def build_toy_rag(corpus: list[str], generator_judge):
    """Toy BM25 RAG over a fixed Wikipedia-style corpus."""
    tokenized = [doc.lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized)

    def rag(question: str) -> RagOutput:
        scores = bm25.get_scores(question.lower().split())
        top_indices = sorted(range(len(scores)), key=lambda i: -scores[i])[:3]
        retrieved = [corpus[i] for i in top_indices]
        prompt = (
            "Answer the question based ONLY on the provided context. "
            "Be concise (one or two sentences).\n\n"
            f"Context:\n{chr(10).join(retrieved)}\n\n"
            f"Question: {question}\n\nAnswer:"
        )
        response = generator_judge.call(prompt)
        return RagOutput(answer=response.text.strip(), retrieved_contexts=retrieved)

    return rag


def _select_provider(console: Console):
    if os.environ.get("GROQ_API_KEY"):
        console.print("[green]Using Groq (Llama 3.3 70B)[/green]")
        return GroqJudge(), GroqJudge(), "groq"
    if os.environ.get("GEMINI_API_KEY"):
        console.print("[yellow]No Groq key — using Gemini[/yellow]")
        return GeminiJudge(), GeminiJudge(), "gemini"

    console.print("[yellow]No API keys — using MockJudge (scores are synthetic)[/yellow]")
    return MockJudge(), MockJudge(response_text="A mocked answer."), "mock"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--toy", action="store_true", help="use toy 6-question dataset")
    parser.add_argument("--n", type=int, default=5, help="HotpotQA samples (ignored with --toy)")
    args = parser.parse_args()

    console = Console()
    judge, generator, provider = _select_provider(console)
    metrics = [Faithfulness(), AnswerRelevance(), AnswerCorrectness()]

    if args.toy:
        dataset = load_toy_dataset()
        rag = build_toy_rag(toy_corpus(), generator)
        config_name = f"toy-bm25-{provider}"
        dataset_name = "toy"
    else:
        try:
            full = load_hotpotqa()
        except FileNotFoundError as exc:
            console.print(f"[red]{exc}[/red]")
            console.print(
                "[yellow]Run `python scripts/prepare_hotpotqa.py` first, or use --toy.[/yellow]"
            )
            return

        dataset = full[: args.n]
        config_name = f"hotpot-bm25-top3-{provider}"
        dataset_name = f"hotpotqa-{args.n}"

        # HotpotQA stores a per-question 10-paragraph corpus on each sample.
        # Adapt the runner's `(question) -> RagOutput` contract with a lookup.
        question_to_sample = {s.question: s for s in dataset}

        def _gen(prompt: str) -> str:
            return generator.call(prompt).text.strip()

        def rag(question: str) -> RagOutput:
            sample = question_to_sample[question]
            answer, retrieved = per_question_bm25_rag(sample, _gen, top_k=3)
            return RagOutput(answer=answer, retrieved_contexts=retrieved)

    result = run_eval(
        rag_system=rag,
        dataset=dataset,
        metrics=metrics,
        judge=judge,
        config_name=config_name,
        dataset_name=dataset_name,
    )

    table = Table(title=f"Run: {result.run_id}", show_lines=True)
    table.add_column("ID", overflow="fold", max_width=12)
    table.add_column("Question", overflow="fold", max_width=40)
    table.add_column("Answer", overflow="fold", max_width=30)
    table.add_column("Truth", overflow="fold", max_width=20)
    for metric_name in result.metric_names():
        table.add_column(metric_name, justify="right")

    sample_lookup = {s.id: s for s in dataset}
    for sample_result in result.samples:
        sample = sample_lookup[sample_result.sample_id]
        row = [
            sample_result.sample_id[:12],
            sample.question,
            sample_result.rag_output.answer[:120],
            sample.ground_truth_answer[:60],
        ]
        for metric_name in result.metric_names():
            row.append(f"{sample_result.metrics[metric_name].score:.2f}")
        table.add_row(*row)

    console.print(table)
    console.print(f"\n[bold]Total cost:[/bold] ${result.total_cost_usd:.4f}")
    for metric_name in result.metric_names():
        scores = result.metric_scores(metric_name)
        avg = sum(scores) / len(scores) if scores else 0
        console.print(f"  {metric_name}: mean={avg:.3f}  (n={len(scores)})")


if __name__ == "__main__":
    main()
