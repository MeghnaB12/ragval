"""The 8 RAG configurations benchmarked on HotpotQA-500.

The configuration grid is designed to answer four questions, each of which
the statistical layer can then answer with a p-value instead of vibes:

1. Does retrieval depth matter?         bm25_k1 vs bm25_k3 vs bm25_k5
2. How far is BM25 from the ceiling?    bm25_k3 vs oracle
3. Does the model already know?         closed_book vs everything else
   (HotpotQA is Wikipedia-based; a 70B model has seen much of it. If
   closed_book scores high on answer_correctness, "RAG improvements" on
   this dataset are partly measuring memorization.)
4. Does chain-of-thought generation help, and does it interact with
   retrieval quality?                   bm25_k3 vs bm25_k3_cot,
                                        oracle vs oracle_cot

Config table:

    name          retrieval                     prompt
    ------------  ----------------------------  --------
    closed_book   none                          concise
    bm25_k1       BM25 top-1                    concise
    bm25_k3       BM25 top-3                    concise
    bm25_k5       BM25 top-5                    concise
    full_context  all 10 paragraphs             concise
    oracle        gold supporting paragraphs    concise
    bm25_k3_cot   BM25 top-3                    CoT
    oracle_cot    gold supporting paragraphs    CoT

All configs share one generator (a callable `prompt -> answer`), so the
only variables are retrieval and prompting.

Because HotpotQA's distractor setting gives each question its own
10-paragraph mini-corpus, RAG systems here are built *per sample*: use
`build_rag_for_sample(config_name, sample, generate)` inside the benchmark
loop rather than a single global callable.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ragval.retrieval import BM25Retriever
from ragval.types import EvalSample, RagOutput

Generator = Callable[[str], str]

CONCISE_PROMPT = (
    "Answer the question based ONLY on the provided context. "
    "Be concise (one sentence when possible). "
    "If the answer is not in the context, say so.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\nAnswer:"
)

COT_PROMPT = (
    "Answer the question based ONLY on the provided context.\n"
    "First, think step by step: identify which parts of the context are "
    "relevant and how they connect. Then give your final answer.\n"
    "End your response with a line of the exact form:\n"
    "FINAL ANSWER: <the answer, as briefly as possible>\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}"
)

CLOSED_BOOK_PROMPT = (
    "Answer the question from your own knowledge. "
    "Be concise (one sentence when possible).\n\n"
    "Question: {question}\n\nAnswer:"
)


def _paragraph_docs(sample: EvalSample) -> list[str]:
    paragraphs: list[dict[str, Any]] = sample.metadata.get("paragraphs") or []
    if not paragraphs:
        raise ValueError(f"Sample {sample.id} has no paragraphs in metadata")
    return [f"{p['title']}: {p['text']}" for p in paragraphs]


def _oracle_docs(sample: EvalSample) -> list[str]:
    """Gold supporting paragraphs, in corpus order."""
    gold_titles = set(sample.metadata.get("supporting_titles") or [])
    docs = []
    for p in sample.metadata.get("paragraphs") or []:
        if p["title"] in gold_titles:
            docs.append(f"{p['title']}: {p['text']}")
    if not docs:
        raise ValueError(f"Sample {sample.id} has no gold paragraphs")
    return docs


def _bm25_docs(sample: EvalSample, top_k: int) -> list[str]:
    docs = _paragraph_docs(sample)
    retriever = BM25Retriever(docs, top_k=top_k)
    return retriever.retrieve(sample.question)


def extract_final_answer(text: str) -> str:
    """Pull the answer out of a CoT response. Falls back to the full text."""
    marker = "FINAL ANSWER:"
    if marker in text:
        return text.rsplit(marker, 1)[1].strip()
    return text.strip()


def _make_output(
    contexts: list[str],
    sample: EvalSample,
    generate: Generator,
    prompt_template: str,
    cot: bool,
) -> RagOutput:
    if contexts:
        prompt = prompt_template.format(context="\n\n".join(contexts), question=sample.question)
    else:
        prompt = prompt_template.format(question=sample.question)
    raw = generate(prompt)
    answer = extract_final_answer(raw) if cot else raw.strip()
    return RagOutput(
        answer=answer,
        retrieved_contexts=contexts,
        metadata={"raw_generation": raw} if cot else {},
    )


def build_rag_for_sample(
    config_name: str, sample: EvalSample, generate: Generator
) -> Callable[[str], RagOutput]:
    """Return a `(question) -> RagOutput` callable for one config + sample.

    The question argument of the returned callable is ignored in favor of
    `sample.question` — retrieval was already bound to this sample's
    mini-corpus, so answering a different question would be incoherent.
    """

    def rag(_question: str) -> RagOutput:
        if config_name == "closed_book":
            return _make_output([], sample, generate, CLOSED_BOOK_PROMPT, cot=False)
        if config_name == "bm25_k1":
            return _make_output(_bm25_docs(sample, 1), sample, generate, CONCISE_PROMPT, cot=False)
        if config_name == "bm25_k3":
            return _make_output(_bm25_docs(sample, 3), sample, generate, CONCISE_PROMPT, cot=False)
        if config_name == "bm25_k5":
            return _make_output(_bm25_docs(sample, 5), sample, generate, CONCISE_PROMPT, cot=False)
        if config_name == "full_context":
            return _make_output(
                _paragraph_docs(sample), sample, generate, CONCISE_PROMPT, cot=False
            )
        if config_name == "oracle":
            return _make_output(_oracle_docs(sample), sample, generate, CONCISE_PROMPT, cot=False)
        if config_name == "bm25_k3_cot":
            return _make_output(_bm25_docs(sample, 3), sample, generate, COT_PROMPT, cot=True)
        if config_name == "oracle_cot":
            return _make_output(_oracle_docs(sample), sample, generate, COT_PROMPT, cot=True)
        raise ValueError(f"Unknown config: {config_name!r}. Known: {CONFIG_NAMES}")

    return rag


CONFIG_NAMES: list[str] = [
    "closed_book",
    "bm25_k1",
    "bm25_k3",
    "bm25_k5",
    "full_context",
    "oracle",
    "bm25_k3_cot",
    "oracle_cot",
]
