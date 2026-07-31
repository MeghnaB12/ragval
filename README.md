# ragval

[![ci](https://github.com/MeghnaB12/ragval/actions/workflows/ci.yml/badge.svg)](https://github.com/MeghnaB12/ragval/actions/workflows/ci.yml)

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
- [x] Preflight quota/token estimator + provider quota checker
- [ ] **Benchmark results — not yet run.** No numbers are published in this README yet; every figure shown is a placeholder.
- [ ] Judge calibration labels (the framework is built; the ~20 human labels are not written yet)
- [ ] Write-up

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
    print(summary)  # e.g. "faithfulness: 0.NNN [95% CI 0.NNN–0.NNN] (n=150)"
```

## The statistics that make it "rigorous"

```python
from ragval.runs import load_run
from ragval.stats import compare_runs

a = load_run("benchmarks/results/oracle-hotpotqa150.jsonl")
b = load_run("benchmarks/results/bm25_k3-hotpotqa150.jsonl")

print(compare_runs(a, b, "answer_correctness"))
```

The output format (values here are placeholders — real results are pending, see Status):

```
answer_correctness: oracle=0.NNN vs bm25_k3=0.NNN | diff=+0.NNN
[95% CI +0.NNN–+0.NNN] p_boot=0.NNNN p_perm=0.NNNN → SIGNIFICANT
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

8 RAG configurations over one axis of retrieval quality and one of prompting, on a stratified sample of HotpotQA (distractor setting). The prepared dataset holds 500 questions; **benchmarks run on the first 150 by default**, because that is what a power analysis justifies — see Sample size below.

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

# 1. What are my provider's actual limits?
python scripts/check_quota.py

# 2. What will this run cost? (no API calls)
python scripts/run_benchmark.py --estimate-only --n 150 \
    --configs closed_book bm25_k3 oracle

# 3. Run it.
python scripts/run_benchmark.py --n 150 --configs closed_book bm25_k3 oracle
```

The benchmark **resumes by default** — every completed sample is checkpointed to `benchmarks/results/partial/`, and judge calls are disk-cached, so interruptions cost nothing. On a free tier the binding constraint is usually *tokens per day*, not requests per minute; `run_benchmark.py` detects daily-quota exhaustion, checkpoints, and exits cleanly so you can resume tomorrow.

By default the answer **generator** runs on a high-quota cheap model (`llama-3.1-8b-instant`) while the scarce strong-model budget is spent on **judging**, where model quality actually moves the numbers. A weaker generator is defensible here — arguably preferable, since it leans on retrieval rather than memorization, which is what the benchmark measures.

### Sample size

n=500 is not free. A power analysis — run through ragval's own paired bootstrap — gives:

| n | power to detect d=0.10 | d=0.15 |
|---|---|---|
| 100 | 0.67 | 0.94 |
| **150** | **0.81** | **0.98** |
| 500 | 0.99 | 1.00 |

n=150 clears the conventional 0.80 threshold for a 10-point difference. n=500 only adds the ability to resolve ~5-point gaps, which sit inside judge noise. The framework should size its own samples rather than default to a round number.

## Design

A RAG system in `ragval` is just a callable: `(question: str) -> RagOutput`. No framework lock-in.

Metrics come in two flavors, deliberately:

- **Judge-based** (faithfulness, answer relevance, answer correctness, context precision/recall) — flexible, works on any dataset, but must be calibrated.
- **Deterministic** (retrieval recall/precision against gold supporting docs) — free, exact, reproducible. Use them to sanity-check the judge: if judge-based context recall disagrees wildly with deterministic retrieval recall, the judge is the problem.

## License

MIT
## Results — HotpotQA-500

Judge: `claude-haiku-4-5`. Every cell is mean [95% bootstrap CI], n=500.

| config | answer_correctness | answer_relevance | context_precision | context_recall | faithfulness | retrieval_precision | retrieval_recall |
|---|---|---|---|---|---|---|---|
| `bm25_k1` | 0.278 [0.24–0.32] | — | — | — | 0.859 [0.83–0.88] | 0.800 [0.77–0.83] | 0.400 [0.38–0.42] |
| `bm25_k3` | 0.512 [0.47–0.56] | — | — | — | 0.866 [0.84–0.89] | 0.459 [0.44–0.48] | 0.687 [0.66–0.71] |
| `bm25_k3_cot` | 0.626 [0.58–0.67] | — | — | — | 0.714 [0.68–0.75] | 0.459 [0.44–0.48] | 0.687 [0.66–0.71] |
| `bm25_k5` | 0.587 [0.54–0.63] | — | — | — | 0.878 [0.85–0.90] | 0.318 [0.31–0.33] | 0.791 [0.77–0.81] |
| `closed_book` | 0.344 [0.30–0.38] | — | — | — | 0.262 [0.22–0.30] | 0.000 [0.00–0.00] | 0.000 [0.00–0.00] |
| `full_context` | 0.724 [0.69–0.76] | — | — | — | 0.891 [0.87–0.92] | 0.202 [0.20–0.21] | 1.000 [1.00–1.00] |
| `hotpot-bm25-top3-gemini` | 0.575 [0.40–0.74] | 0.625 [0.47–0.78] | 0.319 [0.26–0.39] | 0.800 [0.70–0.90] | 0.942 [0.86–1.00] | — | — |
| `oracle` | 0.784 [0.75–0.82] | — | — | — | 0.909 [0.89–0.93] | 1.000 [1.00–1.00] | 1.000 [1.00–1.00] |
| `oracle_cot` | 0.868 [0.84–0.90] | — | — | — | 0.895 [0.87–0.92] | 1.000 [1.00–1.00] | 1.000 [1.00–1.00] |

### Compared against `bm25_k3` (paired, per-sample)

| config | metric | diff | 95% CI | p (boot) | p (perm) | verdict |
|---|---|---|---|---|---|---|
| `bm25_k1` | answer_correctness | -0.234 | [-0.280, -0.189] | 0.0001 | 0.0001 | **significant** |
| `bm25_k1` | faithfulness | -0.007 | [-0.040, +0.027] | 0.6882 | 0.7106 | not significant |
| `bm25_k1` | retrieval_precision | +0.341 | [+0.308, +0.374] | 0.0001 | 0.0001 | **significant** |
| `bm25_k1` | retrieval_recall | -0.287 | [-0.312, -0.263] | 0.0001 | 0.0001 | **significant** |
| `bm25_k3_cot` | answer_correctness | +0.114 | [+0.074, +0.153] | 0.0001 | 0.0001 | **significant** |
| `bm25_k3_cot` | faithfulness | -0.151 | [-0.194, -0.109] | 0.0001 | 0.0001 | **significant** |
| `bm25_k3_cot` | retrieval_precision | +0.000 | [+0.000, +0.000] | 1.0000 | 1.0000 | not significant |
| `bm25_k3_cot` | retrieval_recall | +0.000 | [+0.000, +0.000] | 1.0000 | 1.0000 | not significant |
| `bm25_k5` | answer_correctness | +0.074 | [+0.041, +0.109] | 0.0001 | 0.0001 | **significant** |
| `bm25_k5` | faithfulness | +0.013 | [-0.015, +0.042] | 0.3750 | 0.3847 | not significant |
| `bm25_k5` | retrieval_precision | -0.141 | [-0.153, -0.129] | 0.0001 | 0.0001 | **significant** |
| `bm25_k5` | retrieval_recall | +0.104 | [+0.086, +0.122] | 0.0001 | 0.0001 | **significant** |
| `closed_book` | answer_correctness | -0.168 | [-0.220, -0.116] | 0.0001 | 0.0001 | **significant** |
| `closed_book` | faithfulness | -0.604 | [-0.649, -0.556] | 0.0001 | 0.0001 | **significant** |
| `closed_book` | retrieval_precision | -0.459 | [-0.475, -0.442] | 0.0001 | 0.0001 | **significant** |
| `closed_book` | retrieval_recall | -0.687 | [-0.712, -0.662] | 0.0001 | 0.0001 | **significant** |
| `full_context` | answer_correctness | +0.211 | [+0.166, +0.257] | 0.0001 | 0.0001 | **significant** |
| `full_context` | faithfulness | +0.026 | [-0.006, +0.059] | 0.1255 | 0.1346 | not significant |
| `full_context` | retrieval_precision | -0.256 | [-0.273, -0.240] | 0.0001 | 0.0001 | **significant** |
| `full_context` | retrieval_recall | +0.313 | [+0.288, +0.338] | 0.0001 | 0.0001 | **significant** |
| `hotpot-bm25-top3-gemini` | answer_correctness | +0.225 | [+0.083, +0.383] | 0.0046 | 0.0100 | **significant** |
| `hotpot-bm25-top3-gemini` | faithfulness | +0.083 | [-0.017, +0.200] | 0.1594 | 0.2154 | not significant |
| `oracle` | answer_correctness | +0.272 | [+0.227, +0.317] | 0.0001 | 0.0001 | **significant** |
| `oracle` | faithfulness | +0.044 | [+0.014, +0.076] | 0.0063 | 0.0090 | **significant** |
| `oracle` | retrieval_precision | +0.541 | [+0.525, +0.558] | 0.0001 | 0.0001 | **significant** |
| `oracle` | retrieval_recall | +0.313 | [+0.288, +0.338] | 0.0001 | 0.0001 | **significant** |
| `oracle_cot` | answer_correctness | +0.355 | [+0.309, +0.400] | 0.0001 | 0.0001 | **significant** |
| `oracle_cot` | faithfulness | +0.029 | [-0.004, +0.064] | 0.0961 | 0.1045 | not significant |
| `oracle_cot` | retrieval_precision | +0.541 | [+0.525, +0.558] | 0.0001 | 0.0001 | **significant** |
| `oracle_cot` | retrieval_recall | +0.313 | [+0.288, +0.338] | 0.0001 | 0.0001 | **significant** |
## Results — HotpotQA-500

Judge: `claude-haiku-4-5`. Every cell is mean [95% bootstrap CI], n=500.

| config | answer_correctness | answer_relevance | context_precision | context_recall | faithfulness | retrieval_precision | retrieval_recall |
|---|---|---|---|---|---|---|---|
| `bm25_k1` | 0.278 [0.24–0.32] | — | — | — | 0.859 [0.83–0.88] | 0.800 [0.77–0.83] | 0.400 [0.38–0.42] |
| `bm25_k3` | 0.512 [0.47–0.56] | — | — | — | 0.866 [0.84–0.89] | 0.459 [0.44–0.48] | 0.687 [0.66–0.71] |
| `bm25_k3_cot` | 0.626 [0.58–0.67] | — | — | — | 0.714 [0.68–0.75] | 0.459 [0.44–0.48] | 0.687 [0.66–0.71] |
| `bm25_k5` | 0.587 [0.54–0.63] | — | — | — | 0.878 [0.85–0.90] | 0.318 [0.31–0.33] | 0.791 [0.77–0.81] |
| `closed_book` | 0.344 [0.30–0.38] | — | — | — | 0.262 [0.22–0.30] | 0.000 [0.00–0.00] | 0.000 [0.00–0.00] |
| `full_context` | 0.724 [0.69–0.76] | — | — | — | 0.891 [0.87–0.92] | 0.202 [0.20–0.21] | 1.000 [1.00–1.00] |
| `hotpot-bm25-top3-gemini` | 0.575 [0.40–0.74] | 0.625 [0.47–0.78] | 0.319 [0.26–0.39] | 0.800 [0.70–0.90] | 0.942 [0.86–1.00] | — | — |
| `oracle` | 0.784 [0.75–0.82] | — | — | — | 0.909 [0.89–0.93] | 1.000 [1.00–1.00] | 1.000 [1.00–1.00] |
| `oracle_cot` | 0.868 [0.84–0.90] | — | — | — | 0.895 [0.87–0.92] | 1.000 [1.00–1.00] | 1.000 [1.00–1.00] |

### Compared against `bm25_k3` (paired, per-sample)

| config | metric | diff | 95% CI | p (boot) | p (perm) | verdict |
|---|---|---|---|---|---|---|
| `bm25_k1` | answer_correctness | -0.234 | [-0.280, -0.189] | 0.0001 | 0.0001 | **significant** |
| `bm25_k1` | faithfulness | -0.007 | [-0.040, +0.027] | 0.6882 | 0.7106 | not significant |
| `bm25_k1` | retrieval_precision | +0.341 | [+0.308, +0.374] | 0.0001 | 0.0001 | **significant** |
| `bm25_k1` | retrieval_recall | -0.287 | [-0.312, -0.263] | 0.0001 | 0.0001 | **significant** |
| `bm25_k3_cot` | answer_correctness | +0.114 | [+0.074, +0.153] | 0.0001 | 0.0001 | **significant** |
| `bm25_k3_cot` | faithfulness | -0.151 | [-0.194, -0.109] | 0.0001 | 0.0001 | **significant** |
| `bm25_k3_cot` | retrieval_precision | +0.000 | [+0.000, +0.000] | 1.0000 | 1.0000 | not significant |
| `bm25_k3_cot` | retrieval_recall | +0.000 | [+0.000, +0.000] | 1.0000 | 1.0000 | not significant |
| `bm25_k5` | answer_correctness | +0.074 | [+0.041, +0.109] | 0.0001 | 0.0001 | **significant** |
| `bm25_k5` | faithfulness | +0.013 | [-0.015, +0.042] | 0.3750 | 0.3847 | not significant |
| `bm25_k5` | retrieval_precision | -0.141 | [-0.153, -0.129] | 0.0001 | 0.0001 | **significant** |
| `bm25_k5` | retrieval_recall | +0.104 | [+0.086, +0.122] | 0.0001 | 0.0001 | **significant** |
| `closed_book` | answer_correctness | -0.168 | [-0.220, -0.116] | 0.0001 | 0.0001 | **significant** |
| `closed_book` | faithfulness | -0.604 | [-0.649, -0.556] | 0.0001 | 0.0001 | **significant** |
| `closed_book` | retrieval_precision | -0.459 | [-0.475, -0.442] | 0.0001 | 0.0001 | **significant** |
| `closed_book` | retrieval_recall | -0.687 | [-0.712, -0.662] | 0.0001 | 0.0001 | **significant** |
| `full_context` | answer_correctness | +0.211 | [+0.166, +0.257] | 0.0001 | 0.0001 | **significant** |
| `full_context` | faithfulness | +0.026 | [-0.006, +0.059] | 0.1255 | 0.1346 | not significant |
| `full_context` | retrieval_precision | -0.256 | [-0.273, -0.240] | 0.0001 | 0.0001 | **significant** |
| `full_context` | retrieval_recall | +0.313 | [+0.288, +0.338] | 0.0001 | 0.0001 | **significant** |
| `hotpot-bm25-top3-gemini` | answer_correctness | +0.225 | [+0.083, +0.383] | 0.0046 | 0.0100 | **significant** |
| `hotpot-bm25-top3-gemini` | faithfulness | +0.083 | [-0.017, +0.200] | 0.1594 | 0.2154 | not significant |
| `oracle` | answer_correctness | +0.272 | [+0.227, +0.317] | 0.0001 | 0.0001 | **significant** |
| `oracle` | faithfulness | +0.044 | [+0.014, +0.076] | 0.0063 | 0.0090 | **significant** |
| `oracle` | retrieval_precision | +0.541 | [+0.525, +0.558] | 0.0001 | 0.0001 | **significant** |
| `oracle` | retrieval_recall | +0.313 | [+0.288, +0.338] | 0.0001 | 0.0001 | **significant** |
| `oracle_cot` | answer_correctness | +0.355 | [+0.309, +0.400] | 0.0001 | 0.0001 | **significant** |
| `oracle_cot` | faithfulness | +0.029 | [-0.004, +0.064] | 0.0961 | 0.1045 | not significant |
| `oracle_cot` | retrieval_precision | +0.541 | [+0.525, +0.558] | 0.0001 | 0.0001 | **significant** |
| `oracle_cot` | retrieval_recall | +0.313 | [+0.288, +0.338] | 0.0001 | 0.0001 | **significant** |
