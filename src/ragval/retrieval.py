"""Retrieval helpers.

Two retriever flavors:

- BM25Retriever: standard BM25 over a fixed corpus. Used when the corpus is
  the same across all questions.
- PerQuestionBM25: a thin wrapper that builds a fresh BM25 index from each
  question's metadata.paragraphs. This is what HotpotQA's distractor setting
  needs — each question has its own 10-paragraph mini-corpus.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from rank_bm25 import BM25Okapi

from ragval.types import EvalSample

# Simple tokenizer: lowercase, split on non-word, drop empties.
_TOKEN_RE = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Retriever:
    """BM25 retriever over a fixed list of documents."""

    def __init__(self, documents: list[str], top_k: int = 5):
        if not documents:
            raise ValueError("documents must be non-empty")
        self.documents = documents
        self.top_k = top_k
        self._bm25 = BM25Okapi([_tokenize(d) for d in documents])

    def retrieve(self, query: str, top_k: int | None = None) -> list[str]:
        k = top_k if top_k is not None else self.top_k
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [self.documents[i] for i in ranked]


def per_question_bm25_rag(
    sample: EvalSample,
    generator: Callable[[str], str],
    top_k: int = 3,
) -> tuple[str, list[str]]:
    """One step of a per-question BM25 RAG over HotpotQA's distractor paragraphs.

    Returns (answer, retrieved_contexts).

    `generator(prompt) -> answer_text` is any callable that produces an answer
    given a prompt. In the smoke test we wrap a Judge for this; in week 2
    when we have the full RAG configurations, we will pass real generator
    functions.
    """
    paragraphs: list[dict[str, Any]] = sample.metadata.get("paragraphs") or []
    if not paragraphs:
        raise ValueError(f"Sample {sample.id} has no paragraphs in metadata")

    # Index "title: text" so titles influence retrieval (titles are very informative
    # in HotpotQA — they are Wikipedia article titles).
    docs = [f"{p['title']}: {p['text']}" for p in paragraphs]
    retriever = BM25Retriever(docs, top_k=top_k)
    retrieved = retriever.retrieve(sample.question)

    prompt = (
        "Answer the question based ONLY on the provided context. "
        "Be concise (one sentence when possible). "
        "If the answer is not in the context, say so.\n\n"
        f"Context:\n{chr(10).join(retrieved)}\n\n"
        f"Question: {sample.question}\n\nAnswer:"
    )
    answer = generator(prompt)
    return answer, retrieved
