# ragval

[![ci](https://github.com/MeghnaB12/ragval/actions/workflows/ci.yml/badge.svg)](https://github.com/MeghnaB12/ragval/actions/workflows/ci.yml)

> Rigorous RAG evaluation with confidence intervals, significance testing, and judge calibration.

**🔗 Live dashboard: [ragval.vercel.app](https://ragval.vercel.app)**

## Why ragval

Most RAG evaluation tools tell you that config A scored 0.74 and config B scored 0.71. They don't tell you whether that difference is real or noise.

ragval is built around three principles other RAG eval tools handle loosely:

**Statistical rigor.** Every metric reports a 95% bootstrap confidence interval. Every comparison reports a paired significance test (bootstrap + sign-flip permutation) on per-sample differences aligned by sample ID. Every benchmark answers the question "is this difference real?"

**Judge calibration.** Provide ~20 human-labeled examples; ragval reports exact agreement, within-1 agreement, quadratic-weighted kappa, Spearman correlation, and mean bias of your LLM judge against them. If your judge isn't calibrated, your numbers are theater.

**Reproducibility by default.** Aggressive disk caching of judge calls. Seeded, deterministic statistics. Every run is a single JSONL file. Every result is reproducible from the same inputs.

## Status

- [x] Core data types
- [x] Judge abstraction (Claude, Gemini, Groq, mock) with disk caching, rate limiting, retry
- [x] Metrics: faithfulness, answer relevance, answer correctness
- [x] Metrics: context precision, context recall (judge-based)
- [x] Metrics: retrieval recall, retrieval precision (deterministic, judge-free)
- [x] Statistical layer: bootstrap CIs, paired bootstrap test, permutation test
- [x] Judge calibration (agreement, weighted kappa, Spearman, bias)
- [x] Runner + JSONL run persistence
- [x] CLI: runs, report, compare, calibrate, smoke
- [x] Streamlit dashboard
- [x] HotpotQA-500 benchmark harness: 8 configs, resumable
- [x] Preflight quota/token estimator + provider quota checker
- [x] **Benchmark results: full 8-config run on HotpotQA-500 (see Results below)**
- [x] Judge calibration: 20 faithfulness labels, cross-validated (Claude vs Llama vs human)


## Install (dev)

```bash
git clone https://github.com/MeghnaB12/ragval
cd ragval
pip install -e ".[dev,dashboard,reference]"
```

API keys are read automatically from a `.env` file in the project root:

```
GROQ_API_KEY=gsk_...
ANTHROPIC_API_KEY=sk-ant-...
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
judge = GroqJudge()
result = run_eval(my_rag, dataset, [Faithfulness(), AnswerRelevance(), AnswerCorrectness()], judge)
save_run(result)

for summary in summarize_run(result):
    print(summary)  # e.g. "faithfulness: 0.866 [95% CI 0.84-0.89] (n=500)"
```

## The statistics that make it "rigorous"

```python
from ragval.runs import load_run
from ragval.stats import compare_runs

a = load_run("benchmarks/results/oracle-hotpotqa500.jsonl")
b = load_run("benchmarks/results/bm25_k3-hotpotqa500.jsonl")

print(compare_runs(a, b, "answer_correctness"))
# answer_correctness: oracle=0.784 vs bm25_k3=0.512 | diff=+0.272
# [95% CI +0.227-+0.317] p_boot=0.0001 p_perm=0.0001 -> SIGNIFICANT
```

Comparisons are **paired by sample ID**. Question difficulty varies enormously (HotpotQA "easy" vs "hard"), and pairing removes that variance — an unpaired test on the same data can be 5-10x less powerful.

## CLI

```bash
ragval runs                          # list saved benchmark runs
ragval report bm25_k3                # means + 95% bootstrap CIs
ragval compare oracle bm25_k3        # paired significance tests
ragval calibrate cal.jsonl --metric faithfulness --judge groq
```

## Dashboard

A full-stack web app (FastAPI + React) for exploring the results lives in
[`dashboard/`](dashboard/). It serves the same statistical engine over a REST
API and visualizes confidence intervals, paired significance tests, per-sample
judge reasoning, and judge calibration.

![dashboard overview](dashboard/docs/overview.png)

```bash
cd dashboard && ./dev.sh          # backend on :8000, frontend on :5173
```

See [`dashboard/README.md`](dashboard/README.md) for architecture and deploy
instructions.

There is also a lightweight Streamlit view for quick local inspection:

```bash
pip install -e ".[dashboard]"
streamlit run src/ragval/dashboard.py
```

## The HotpotQA-500 benchmark

8 RAG configurations over one axis of retrieval quality and one of prompting, on a 500-question sample of HotpotQA (distractor setting):

| config | retrieval | prompt |
|---|---|---|
| closed_book | none | concise |
| bm25_k1 | BM25 top-1 | concise |
| bm25_k3 | BM25 top-3 | concise |
| bm25_k5 | BM25 top-5 | concise |
| full_context | all 10 paragraphs | concise |
| oracle | gold supporting paragraphs | concise |
| bm25_k3_cot | BM25 top-3 | chain-of-thought |
| oracle_cot | gold supporting paragraphs | chain-of-thought |

The grid answers four questions with p-values instead of vibes: does retrieval depth matter (k1/k3/k5), how far is BM25 from the ceiling (vs oracle), does the model already know the answers (closed_book — HotpotQA is Wikipedia-based), and does CoT help and interact with retrieval quality.

```bash
# dataset is committed at benchmarks/hotpotqa-500.jsonl; to regenerate:
python scripts/prepare_hotpotqa.py

# 1. What are my provider's actual limits?
python scripts/check_quota.py

# 2. What will this run cost? (no API calls)
python scripts/run_benchmark.py --estimate-only

# 3. Run it. Generation on free Groq, judging on Claude.
python scripts/run_benchmark.py --judge claude --rpm 24 --yes
```

The benchmark **resumes by default** — every completed sample is checkpointed to `benchmarks/results/partial/`, and judge calls are disk-cached, so interruptions cost nothing. On a free tier the binding constraint is usually *tokens per day*, not requests per minute; `run_benchmark.py` detects daily-quota exhaustion, checkpoints, and exits cleanly so you can resume the next day.

The answer **generator** runs on a high-quota cheap model (`llama-3.1-8b-instant`) while the **judge** runs on a stronger model from a different family (`claude-haiku-4-5`). Splitting families removes the self-preference bias a same-family judge can have, and a weaker generator is defensible here — arguably preferable, since it leans on retrieval rather than memorization, which is what the benchmark measures.

### Sample size

A power analysis, run through ragval's own paired bootstrap:

| n | power to detect d=0.10 | d=0.15 |
|---|---|---|
| 100 | 0.67 | 0.94 |
| 150 | 0.81 | 0.98 |
| 500 | 0.99 | 1.00 |

n=150 already clears the conventional 0.80 threshold for a 10-point difference; the full run below uses n=500, which resolves gaps down to ~5 points.

## Design

A RAG system in ragval is just a callable: `(question: str) -> RagOutput`. No framework lock-in.

Metrics come in two flavors, deliberately:

- **Judge-based** (faithfulness, answer relevance, answer correctness, context precision/recall) — flexible, works on any dataset, but must be calibrated.
- **Deterministic** (retrieval recall/precision against gold supporting docs) — free, exact, reproducible. Use them to sanity-check the judge: if judge-based context recall disagrees wildly with deterministic retrieval recall, the judge is the problem.

## Results — HotpotQA-500

Generator: `llama-3.1-8b-instant`. Judge: `claude-haiku-4-5`. Every cell is mean [95% bootstrap CI], n=500.

| config | answer_correctness | faithfulness | retrieval_precision | retrieval_recall |
|---|---|---|---|---|
| closed_book | 0.344 [0.30-0.38] | 0.262 [0.22-0.30] | 0.000 | 0.000 |
| bm25_k1 | 0.278 [0.24-0.32] | 0.859 [0.83-0.88] | 0.800 [0.77-0.83] | 0.400 [0.38-0.42] |
| bm25_k3 | 0.512 [0.47-0.56] | 0.866 [0.84-0.89] | 0.459 [0.44-0.48] | 0.687 [0.66-0.71] |
| bm25_k5 | 0.587 [0.54-0.63] | 0.878 [0.85-0.90] | 0.318 [0.31-0.33] | 0.791 [0.77-0.81] |
| full_context | 0.724 [0.69-0.76] | 0.891 [0.87-0.92] | 0.202 [0.20-0.21] | 1.000 |
| oracle | 0.784 [0.75-0.82] | 0.909 [0.89-0.93] | 1.000 | 1.000 |
| bm25_k3_cot | 0.626 [0.58-0.67] | 0.714 [0.68-0.75] | 0.459 [0.44-0.48] | 0.687 [0.66-0.71] |
| oracle_cot | 0.868 [0.84-0.90] | 0.895 [0.87-0.92] | 1.000 | 1.000 |

### What the benchmark shows

All differences below are vs `bm25_k3`, paired bootstrap, n=500. Every claim is significant at p < 0.001 unless stated.

- **Retrieval quality drives correctness, monotonically.** answer_correctness climbs 0.28 -> 0.51 -> 0.59 -> 0.78 across bm25_k1 -> k3 -> k5 -> oracle as retrieval_recall rises 0.40 -> 0.69 -> 0.79 -> 1.00. Every step is significant.
- **Bad retrieval is worse than none.** `closed_book` (0.344) *beats* `bm25_k1` (0.278) on correctness — a single mis-retrieved document actively misleads the generator, where no context at least lets it fall back on parametric knowledge.
- **Perfect retrieval isn't perfect answers.** Even `oracle`, with gold documents and precision/recall = 1.0, caps at 0.784 correctness. The ~22-point gap is the 8B generator's own ceiling, not a retrieval failure.
- **Chain-of-thought recovers part of that ceiling — at a cost.** `oracle_cot` lifts correctness to 0.868 (+0.084 over `oracle`) with identical retrieval. But CoT *lowers* faithfulness: `bm25_k3_cot` faithfulness is 0.714 vs `bm25_k3`'s 0.866 (-0.151, significant). Reasoning makes answers more correct yet less strictly grounded in the retrieved text.
- **More context is not better.** `full_context` (all 10 paragraphs, precision 0.20) scores 0.724 — below `oracle`'s 0.784 despite identical recall. Noise dilutes even when the answer is present.

### Compared against `bm25_k3` (paired, per-sample)

| config | metric | diff | 95% CI | p (boot) | p (perm) | verdict |
|---|---|---|---|---|---|---|
| closed_book | answer_correctness | -0.168 | [-0.220, -0.116] | 0.0001 | 0.0001 | **significant** |
| closed_book | faithfulness | -0.604 | [-0.649, -0.556] | 0.0001 | 0.0001 | **significant** |
| bm25_k1 | answer_correctness | -0.234 | [-0.280, -0.189] | 0.0001 | 0.0001 | **significant** |
| bm25_k1 | faithfulness | -0.007 | [-0.040, +0.027] | 0.6882 | 0.7106 | not significant |
| bm25_k5 | answer_correctness | +0.074 | [+0.041, +0.109] | 0.0001 | 0.0001 | **significant** |
| bm25_k5 | retrieval_recall | +0.104 | [+0.086, +0.122] | 0.0001 | 0.0001 | **significant** |
| full_context | answer_correctness | +0.211 | [+0.166, +0.257] | 0.0001 | 0.0001 | **significant** |
| full_context | retrieval_precision | -0.256 | [-0.273, -0.240] | 0.0001 | 0.0001 | **significant** |
| oracle | answer_correctness | +0.272 | [+0.227, +0.317] | 0.0001 | 0.0001 | **significant** |
| oracle | faithfulness | +0.044 | [+0.014, +0.076] | 0.0063 | 0.0090 | **significant** |
| bm25_k3_cot | answer_correctness | +0.114 | [+0.074, +0.153] | 0.0001 | 0.0001 | **significant** |
| bm25_k3_cot | faithfulness | -0.151 | [-0.194, -0.109] | 0.0001 | 0.0001 | **significant** |
| oracle_cot | answer_correctness | +0.355 | [+0.309, +0.400] | 0.0001 | 0.0001 | **significant** |
| oracle_cot | faithfulness | +0.029 | [-0.004, +0.064] | 0.0961 | 0.1045 | not significant |

Full per-metric comparisons (including retrieval precision/recall for every config) are reproducible with `ragval compare <config> bm25_k3`.

### Judge calibration

The faithfulness judge was validated against 20 human-labeled examples (scored blind, before seeing any judge output):

| judge | within-1 agreement | quadratic-weighted κ | Spearman | mean bias |
|---|---|---|---|---|
| claude-haiku-4-5 (benchmark judge) | 0.70 | 0.458 | 0.528 | −0.40 |
| llama-3.3-70b (cross-check) | 0.75 | 0.665 | 0.621 | −0.05 |

Two judges from different model families agree with human labels moderately (κ ≈ 0.5–0.67) and with each other at κ = 0.52. **Both run harsh on faithfulness** — Claude notably so (−0.40), meaning the absolute faithfulness scores above are likely *conservative*. Ranking between configs is preserved (Spearman ≈ 0.5–0.6), so the paired comparisons hold; absolute faithfulness values should be read as a lower bound. This is exactly the caveat ragval is built to surface — an uncalibrated judge would report these numbers with false confidence.

## License

MIT
