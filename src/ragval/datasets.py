"""Dataset loaders.

Datasets in ragval are lists of EvalSample. JSONL is the canonical format.
"""

from __future__ import annotations

import json
from pathlib import Path

from ragval.types import EvalSample


def load_jsonl(path: Path | str) -> list[EvalSample]:
    """Load a dataset from a JSONL file. Each line is one EvalSample."""
    p = Path(path)
    samples: list[EvalSample] = []
    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            samples.append(EvalSample(**data))
    return samples


def save_jsonl(samples: list[EvalSample], path: Path | str) -> None:
    """Save a dataset to JSONL."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        for s in samples:
            f.write(s.model_dump_json() + "\n")


def load_hotpotqa(path: Path | str | None = None) -> list[EvalSample]:
    """Load the prepared HotpotQA-500 dataset.

    Defaults to benchmarks/hotpotqa-500.jsonl in the project root.

    The original HotpotQA rows have rich metadata (per-question paragraphs,
    supporting fact titles and sentence IDs). All of that lives in
    `EvalSample.metadata` so the RAG system can use it for retrieval.
    """
    if path is None:
        # Project root assumed two levels up from this file: src/ragval/datasets.py
        root = Path(__file__).resolve().parent.parent.parent
        path = root / "benchmarks" / "hotpotqa-500.jsonl"
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{p} not found. Run `python scripts/prepare_hotpotqa.py` first.")
    return load_jsonl(p)

    """Tiny in-memory dataset for smoke tests. 5 questions, all answerable from
    a small Wikipedia-style corpus.
    """
    return [
        EvalSample(
            id="toy-1",
            question="Who wrote the novel '1984'?",
            ground_truth_answer="George Orwell",
            ground_truth_contexts=[
                "George Orwell wrote 1984, a dystopian novel published in 1949."
            ],
        ),
        EvalSample(
            id="toy-2",
            question="What is the capital of Australia?",
            ground_truth_answer="Canberra",
            ground_truth_contexts=["Canberra is the capital city of Australia."],
        ),
        EvalSample(
            id="toy-3",
            question="What year did the Berlin Wall fall?",
            ground_truth_answer="1989",
            ground_truth_contexts=["The Berlin Wall fell on November 9, 1989."],
        ),
        EvalSample(
            id="toy-4",
            question="What does DNA stand for?",
            ground_truth_answer="Deoxyribonucleic acid",
            ground_truth_contexts=[
                "DNA stands for deoxyribonucleic acid, the molecule that carries genetic information."
            ],
        ),
        EvalSample(
            id="toy-5",
            question="Who painted the Mona Lisa?",
            ground_truth_answer="Leonardo da Vinci",
            ground_truth_contexts=[
                "The Mona Lisa was painted by Leonardo da Vinci between 1503 and 1519."
            ],
        ),
    ]


def load_toy_dataset() -> list[EvalSample]:
    """Tiny in-memory dataset for smoke tests. Not part of the framework's
    public API — used only by the optional --toy mode of smoke_test.
    """
    return [
        EvalSample(
            id="toy-1",
            question="Who wrote the novel '1984'?",
            ground_truth_answer="George Orwell",
            ground_truth_contexts=[
                "George Orwell wrote 1984, a dystopian novel published in 1949."
            ],
        ),
        EvalSample(
            id="toy-2",
            question="What is the capital of Australia?",
            ground_truth_answer="Canberra",
            ground_truth_contexts=["Canberra is the capital city of Australia."],
        ),
        EvalSample(
            id="toy-3",
            question="What year did the Berlin Wall fall?",
            ground_truth_answer="1989",
            ground_truth_contexts=["The Berlin Wall fell on November 9, 1989."],
        ),
        EvalSample(
            id="toy-4",
            question="What does DNA stand for?",
            ground_truth_answer="Deoxyribonucleic acid",
            ground_truth_contexts=[
                "DNA stands for deoxyribonucleic acid, the molecule that carries genetic information."
            ],
        ),
        EvalSample(
            id="toy-5",
            question="Who painted the Mona Lisa?",
            ground_truth_answer="Leonardo da Vinci",
            ground_truth_contexts=[
                "The Mona Lisa was painted by Leonardo da Vinci between 1503 and 1519."
            ],
        ),
        EvalSample(
            id="toy-6",
            question="What is the speed of light in a vacuum?",
            ground_truth_answer="299,792,458 meters per second",
            ground_truth_contexts=["The speed of light in a vacuum is exactly 299,792,458 m/s."],
        ),
    ]


def toy_corpus() -> list[str]:
    """A small corpus of Wikipedia-style passages used by the smoke-test RAG."""
    return [
        "George Orwell wrote 1984, a dystopian novel published in 1949. The book explores themes of totalitarianism, mass surveillance, and authoritarian rule.",
        "Animal Farm is another novel by George Orwell, published in 1945, which uses farm animals as an allegory for the Russian Revolution.",
        "Canberra is the capital city of Australia. It was selected as the site of the capital in 1908 as a compromise between Sydney and Melbourne.",
        "Sydney is the largest city in Australia and the capital of New South Wales, but it is not the national capital.",
        "The Berlin Wall fell on November 9, 1989, leading to German reunification in 1990.",
        "World War II ended in 1945 with the surrender of Germany in May and Japan in September.",
        "DNA stands for deoxyribonucleic acid, the molecule that carries genetic information in living organisms.",
        "RNA, or ribonucleic acid, plays a key role in protein synthesis and gene expression.",
        "The Mona Lisa was painted by Leonardo da Vinci between 1503 and 1519. It is housed at the Louvre Museum in Paris.",
        "The Last Supper is another famous painting by Leonardo da Vinci, located in Milan.",
        "Mount Everest is the highest mountain on Earth, with a peak at 8,849 meters above sea level.",
        "The Pacific Ocean is the largest ocean on Earth, covering about 63 million square miles.",
    ]
