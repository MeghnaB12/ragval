"""Download HotpotQA validation set and prepare a 500-sample stratified subset.

Produces: benchmarks/hotpotqa-500.jsonl

Schema (per JSONL line):
  id, question, ground_truth_answer, ground_truth_contexts, metadata{
    level, type, paragraphs:[{title,text}], supporting_titles, supporting_sent_ids
  }

Usage:
    python scripts/prepare_hotpotqa.py
    python scripts/prepare_hotpotqa.py --n 500 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download


def paragraph_text(sentences: list[str]) -> str:
    return " ".join(s.strip() for s in sentences if s.strip())


def convert_row(row: dict) -> dict:
    titles = row["context"]["title"]
    sent_lists = row["context"]["sentences"]
    paragraphs = [
        {"title": t, "text": paragraph_text(list(s))}
        for t, s in zip(titles, sent_lists, strict=True)
    ]
    supporting_titles = list(row["supporting_facts"]["title"])
    supporting_sent_ids = [int(x) for x in row["supporting_facts"]["sent_id"]]
    return {
        "id": row["id"],
        "question": row["question"],
        "ground_truth_answer": row["answer"],
        "ground_truth_contexts": list(set(supporting_titles)),
        "metadata": {
            "level": row["level"],
            "type": row["type"],
            "paragraphs": paragraphs,
            "supporting_titles": supporting_titles,
            "supporting_sent_ids": supporting_sent_ids,
        },
    }


def stratified_sample(rows: list[dict], n: int, seed: int) -> list[dict]:
    by_level: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_level[r["metadata"]["level"]].append(r)

    rng = random.Random(seed)
    levels = sorted(by_level.keys())
    total = sum(len(by_level[lvl]) for lvl in levels)
    sampled: list[dict] = []
    for lvl in levels:
        share = round(n * len(by_level[lvl]) / total)
        pool = by_level[lvl]
        rng.shuffle(pool)
        sampled.extend(pool[:share])

    rng.shuffle(sampled)
    if len(sampled) < n:
        seen = {r["id"] for r in sampled}
        leftovers = [r for r in rows if r["id"] not in seen]
        rng.shuffle(leftovers)
        sampled.extend(leftovers[: n - len(sampled)])
    return sampled[:n]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent.parent / "benchmarks" / "hotpotqa-500.jsonl",
    )
    args = parser.parse_args()

    print("Downloading HotpotQA validation parquet from Hugging Face (~50 MB)…")
    parquet_path = hf_hub_download(
        repo_id="hotpotqa/hotpot_qa",
        filename="distractor/validation-00000-of-00001.parquet",
        repo_type="dataset",
    )
    print(f"  parquet: {parquet_path}")

    print("Reading parquet…")
    table = pq.read_table(parquet_path)
    rows = table.to_pylist()
    print(f"  loaded {len(rows)} validation rows")

    print("Converting…")
    converted = [convert_row(r) for r in rows]

    print(f"Stratified sampling {args.n} (seed={args.seed})…")
    sampled = stratified_sample(converted, args.n, args.seed)

    levels: dict[str, int] = defaultdict(int)
    types: dict[str, int] = defaultdict(int)
    for r in sampled:
        levels[r["metadata"]["level"]] += 1
        types[r["metadata"]["type"]] += 1
    print(f"  level: {dict(levels)}")
    print(f"  type:  {dict(types)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for r in sampled:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {args.out} ({args.out.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
