"""Minimal ragval example that runs without API keys.

This example uses only deterministic retrieval metrics and a MockJudge placeholder,
so it is safe to run locally or in CI. Replace MockJudge with GroqJudge,
GeminiJudge, or ClaudeJudge when you add judge-based metrics.
"""

from ragval.judges import MockJudge
from ragval.metrics import RetrievalPrecision, RetrievalRecall
from ragval.runner import run_eval
from ragval.stats import summarize_run
from ragval.types import EvalSample, RagOutput


SAMPLES = [
    EvalSample(
        id="1",
        question="Who wrote 1984?",
        ground_truth_answer="George Orwell",
        metadata={"supporting_titles": ["George Orwell"]},
    ),
    EvalSample(
        id="2",
        question="What is the capital of Australia?",
        ground_truth_answer="Canberra",
        metadata={"supporting_titles": ["Canberra"]},
    ),
]


ANSWERS = {
    "Who wrote 1984?": RagOutput(
        answer="George Orwell",
        retrieved_contexts=[
            "George Orwell: George Orwell wrote the novel 1984.",
            "Aldous Huxley: Aldous Huxley wrote Brave New World.",
        ],
    ),
    "What is the capital of Australia?": RagOutput(
        answer="Canberra",
        retrieved_contexts=["Canberra: Canberra is the capital of Australia."],
    ),
}


def my_rag(question: str) -> RagOutput:
    return ANSWERS[question]


result = run_eval(
    rag_system=my_rag,
    dataset=SAMPLES,
    metrics=[RetrievalRecall(), RetrievalPrecision()],
    judge=MockJudge(),  # deterministic metrics do not call the judge
    config_name="basic-example",
    dataset_name="tiny-demo",
    show_progress=False,
)

for summary in summarize_run(result):
    print(summary)
