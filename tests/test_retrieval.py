import pytest

from ragval.retrieval import BM25Retriever, _tokenize, per_question_bm25_rag
from ragval.types import EvalSample


def test_tokenize_lowercase_and_punctuation():
    assert _tokenize("Hello, World!") == ["hello", "world"]
    assert _tokenize("Don't stop") == ["don", "t", "stop"]


def test_bm25_retriever_basic():
    docs = [
        "The cat sat on the mat",
        "Dogs chase cars in the park",
        "Cats love sunny windowsills",
    ]
    r = BM25Retriever(docs, top_k=2)
    top = r.retrieve("cat")
    assert len(top) == 2
    # The two cat-related docs should rank above the dog one
    assert "Dogs chase cars" not in top[0]


def test_bm25_retriever_rejects_empty():
    with pytest.raises(ValueError):
        BM25Retriever([])


def test_per_question_bm25_rag():
    sample = EvalSample(
        id="hp-1",
        question="What color is the sky?",
        ground_truth_answer="Blue",
        ground_truth_contexts=["Sky"],
        metadata={
            "paragraphs": [
                {"title": "Sky", "text": "The sky appears blue due to Rayleigh scattering."},
                {"title": "Grass", "text": "Grass is green because of chlorophyll."},
                {"title": "Roses", "text": "Roses are typically red."},
            ]
        },
    )
    received_prompt = []

    def fake_gen(prompt: str) -> str:
        received_prompt.append(prompt)
        return "Blue"

    answer, retrieved = per_question_bm25_rag(sample, fake_gen, top_k=2)
    assert answer == "Blue"
    assert len(retrieved) == 2
    # The Sky paragraph should be retrieved first
    assert "Sky:" in retrieved[0]
    # The prompt must include the question
    assert "What color is the sky?" in received_prompt[0]


def test_per_question_bm25_rag_no_paragraphs_raises():
    sample = EvalSample(id="x", question="q", ground_truth_answer="a")
    with pytest.raises(ValueError):
        per_question_bm25_rag(sample, lambda _: "a")
