"""Smoke test: run ragval against a small slice of real HotpotQA.

Run with:
    GEMINI_API_KEY=... python -m ragval.smoke_test            # 5 HotpotQA samples
    GEMINI_API_KEY=... python -m ragval.smoke_test --toy      # original toy dataset
    GEMINI_API_KEY=... python -m ragval.smoke_test --n 10     # 10 HotpotQA samples

The HotpotQA dataset must be prepared first:
    python scripts/prepare_hotpotqa.py
"""

from __future__ import annotations

import argparse
import os

from ragval.metrics import AnswerCorrectness, AnswerRelevance, Faithfulness
from rank_bm25 import BM25Okapi
from rich.console import Console
from rich.table import Table

from ragval.datasets import load_hotpotqa, load_toy_dataset, toy_corpus
from ragval.judges import GeminiJudge, MockJudge, GroqJudge
from ragval.metrics import AnswerCorrectness, AnswerRelevance, Faithfulness
from ragval.retrieval import per_question_bm25_rag
from ragval.runner import run_eval
from ragval.types import RagOutput



def build_toy_rag(corpus: list[str], generator_judge):
    """Toy BM25 RAG over a fixed Wikipedia-style corpus. Used in --toy mode."""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--toy", action="store_true", help="use toy 6-question dataset")
    parser.add_argument("--n", type=int, default=5, help="HotpotQA samples (ignored with --toy)")
    args = parser.parse_args()

    console = Console()
    

    if os.environ.get("GROQ_API_KEY"):
        console.print("[green]Using GroqJudge (Llama 3.3 70B)[/green]")
        judge = GroqJudge()
        generator = GroqJudge()
    elif os.environ.get("GEMINI_API_KEY"):
        console.print("[yellow]No Groq key — falling back to Gemini (limited free tier)[/yellow]")
        judge = GeminiJudge()
        generator = GeminiJudge()
    else:
        console.print("[yellow]No API keys — using MockJudge (results will be fake)[/yellow]")
        judge = MockJudge()
        generator = MockJudge(response_text="A mocked answer.")

    # if os.environ.get("GROQ_API_KEY"):
    #     console.print("[green]Using GroqJudge (Llama 3.3 70B)[/green]")
    #     judge = GroqJudge()
    #     generator = GroqJudge()
    # elif os.environ.get("GEMINI_API_KEY"):
    #     console.print("[yellow]No Groq key — falling back to Gemini (limited free tier)[/yellow]")
    #     judge = GeminiJudge()
    #     generator = GeminiJudge()
    # else:
    #     console.print("[yellow]No API keys — using MockJudge (results will be fake)[/yellow]")
    #     judge = MockJudge()
    #     generator = MockJudge(response_text="A mocked answer.")
        

    # if os.environ.get("GEMINI_API_KEY"):
    #     console.print("[green]Using GeminiJudge[/green]")
    #     judge = GeminiJudge()
    #     generator = GeminiJudge()
    # else:
    #     console.print("[yellow]No GEMINI_API_KEY — using MockJudge (results will be fake)[/yellow]")
    #     judge = MockJudge()
    #     generator = MockJudge(response_text="A mocked answer.")

    metrics = [Faithfulness(), AnswerRelevance(), AnswerCorrectness()]

    if args.toy:
        dataset = load_toy_dataset()
        rag = build_toy_rag(toy_corpus(), generator)
        config_name = "toy-bm25-gemini"
        dataset_name = "toy"
    else:
        try:
            full = load_hotpotqa()
        except FileNotFoundError as e:
            console.print(f"[red]{e}[/red]")
            console.print(
                "[yellow]Run `python scripts/prepare_hotpotqa.py` first, or use --toy.[/yellow]"
            )
            return

        dataset = full[: args.n]
        config_name = "hotpot-bm25-top3-gemini"
        dataset_name = f"hotpotqa-{args.n}"

        # HotpotQA retrieval needs the *whole sample* (the per-question 10-paragraph
        # mini-corpus lives in sample.metadata.paragraphs). The runner contract is
        # `(question) -> RagOutput`, so we adapt by looking up the sample by question.
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

    # Render
    table = Table(title=f"Run: {result.run_id}", show_lines=True)
    table.add_column("ID", overflow="fold", max_width=12)
    table.add_column("Question", overflow="fold", max_width=40)
    table.add_column("Answer", overflow="fold", max_width=30)
    table.add_column("Truth", overflow="fold", max_width=20)
    for m in result.metric_names():
        table.add_column(m, justify="right")

    sample_lookup = {s.id: s for s in dataset}
    for sr in result.samples:
        s = sample_lookup[sr.sample_id]
        row = [
            sr.sample_id[:12],
            s.question,
            sr.rag_output.answer[:120],
            s.ground_truth_answer[:60],
        ]
        for m in result.metric_names():
            row.append(f"{sr.metrics[m].score:.2f}")
        table.add_row(*row)

    console.print(table)

    console.print(f"\n[bold]Total cost:[/bold] ${result.total_cost_usd:.4f}")
    for m in result.metric_names():
        scores = result.metric_scores(m)
        avg = sum(scores) / len(scores) if scores else 0
        console.print(f"  {m}: mean={avg:.3f}  (n={len(scores)})")


if __name__ == "__main__":
    main()
