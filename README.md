# ragval

> Rigorous RAG evaluation with confidence intervals, significance testing, and judge calibration.

## Why ragval

Most RAG evaluation tools tell you that config A scored 0.74 and config B scored 0.71. They don't tell you whether that difference is real or noise.

`ragval` is built around three principles other RAG eval tools handle loosely:

1. **Statistical rigor.** Every metric reports a 95% bootstrap confidence interval. Every comparison reports a *paired* significance test (bootstrap + sign-flip permutation) on per-sample differences aligned by sample ID. Every benchmark answers the question "is this difference real?"
2. **Judge calibration.** Provide ~20 human-labeled examples; `ragval` reports exact agreement, within-1 agreement, quadratic-weighted kappa, Spearman correlation, and mean bias of your LLM judge against them. If your judge isn't calibrated, your numbers are theater.
3. **Reproducibility by default.** Aggressive disk caching of judge calls. Seeded, deterministic statistics. Every run is a single JSONL file. Every result is reproducible from the same inputs.

## Status

- [x] Core data types
- [x] Judge abstraction (Gemini, Groq, mock) with disk caching, rate limiting, retry
- [x] Metrics: faithfulness, answer relevance, answer correctness
- [x] Metrics: context precision, context recall (judge-based)
- [x] Metrics: retrieval recall, retrieval precision (deterministic, judge-free — for datasets with gold supporting docs)
- [x] Statistical layer: bootstrap CIs, paired bootstrap test, permutation test
- [x] Judge calibration (agreement, weighted kappa, Spearman, bias)
- [x] Runner + JSONL run persistence
- [x] CLI: `runs`, `report`, `compare`, `calibrate`, `smoke`
- [x] Streamlit dashboard
- [x] HotpotQA-500 benchmark harness: 8 configs, resumable
- [ ] Full benchmark results + write-up (in progress)

## Install (dev)

```bash
git clone https://github.com/MeghnaB12/ragval
cd ragval
pip install -e ".[dev,dashboard]"
```

## Quick start

```python
from ragval import EvalSample, RagOutput
from ragval.judges import GroqJudge
from ragval.metrics import Faithfulness, AnswerRelevance, AnswerCorrectness
from ragval.runner import run_eval
from ragval.runs import save_run
from ragval.stats import summarize_run

def my_rag(question: str) -> RagOutput:
    # plug in anything: LangChain, LlamaIndex, raw functions, production code
    ...

dataset = [EvalSample(id="1", question="...", ground_truth_answer="...")]
judge = GroqJudge()  # export GROQ_API_KEY=...
result = run_eval(my_rag, dataset, [Faithfulness(), AnswerRelevance(), AnswerCorrectness()], judge)
save_run(result)

for summary in summarize_run(result):
    print(summary)  # faithfulness: 0.812 [95% CI 0.771–0.849] (n=500)
```

## The statistics that make it "rigorous"

```python
from ragval.runs import load_run
from ragval.stats import compare_runs

a = load_run("benchmarks/results/oracle-hotpotqa500.jsonl")
b = load_run("benchmarks/results/bm25_k3-hotpotqa500.jsonl")

print(compare_runs(a, b, "answer_correctness"))
# answer_correctness: oracle=0.802 vs bm25_k3=0.641 | diff=+0.161
# [95% CI +0.118–+0.204] p_boot=0.0001 p_perm=0.0001 → SIGNIFICANT
```

Comparisons are **paired by sample ID**. Question difficulty varies enormously (HotpotQA "easy" vs "hard"), and pairing removes that variance — an unpaired test on the same data can be 5–10× less powerful.

## CLI

```bash
ragval runs                          # list saved benchmark runs
ragval report bm25_k3                # means + 95% bootstrap CIs
ragval compare oracle bm25_k3        # paired significance tests
ragval calibrate cal.jsonl --metric faithfulness --judge groq
```

## Dashboard

```bash
pip install -e ".[dashboard]"
streamlit run src/ragval/dashboard.py
```

Three tabs: per-run metric means with CI error bars, run-vs-run comparison with p-values, and a sample explorer sorted worst-first with judge reasoning (invaluable for debugging *why* a config loses).

## The HotpotQA-500 benchmark

8 RAG configurations over one axis of retrieval quality and one of prompting, on a stratified 500-question sample of HotpotQA (distractor setting):

| config | retrieval | prompt |
|---|---|---|
| `closed_book` | none | concise |
| `bm25_k1` | BM25 top-1 | concise |
| `bm25_k3` | BM25 top-3 | concise |
| `bm25_k5` | BM25 top-5 | concise |
| `full_context` | all 10 paragraphs | concise |
| `oracle` | gold supporting paragraphs | concise |
| `bm25_k3_cot` | BM25 top-3 | chain-of-thought |
| `oracle_cot` | gold supporting paragraphs | chain-of-thought |

The grid is designed to answer four questions with p-values instead of vibes: does retrieval depth matter (k1/k3/k5), how far is BM25 from the ceiling (vs oracle), does the model already know the answers (closed_book — HotpotQA is Wikipedia-based, so a 70B model has memorized much of it), and does CoT help and does it interact with retrieval quality.

```bash
# dataset is committed at benchmarks/hotpotqa-500.jsonl; to regenerate:
python scripts/prepare_hotpotqa.py

export GROQ_API_KEY=...
python scripts/run_benchmark.py --n 50 --configs bm25_k3 oracle   # dry run
python scripts/run_benchmark.py                                    # everything
```

The benchmark **resumes by default** — every completed sample is checkpointed to `benchmarks/results/partial/`, and judge calls are disk-cached, so interruptions (inevitable on free-tier rate limits) cost nothing.

## Design

A RAG system in `ragval` is just a callable: `(question: str) -> RagOutput`. No framework lock-in.

Metrics come in two flavors, deliberately:

- **Judge-based** (faithfulness, answer relevance, answer correctness, context precision/recall) — flexible, works on any dataset, but must be calibrated.
- **Deterministic** (retrieval recall/precision against gold supporting docs) — free, exact, reproducible. Use them to sanity-check the judge: if judge-based context recall disagrees wildly with deterministic retrieval recall, the judge is the problem.

## License

MIT
