# ragval Architecture

ragval separates evaluation logic, experiment persistence, statistical analysis, and visualization so that the same evaluation engine can be used from Python, the CLI, or the web dashboard.

## System overview

```mermaid
graph LR
    A[Dataset / EvalSample] --> B[RAG callable]
    B --> C[RagOutput]
    C --> D[Metric runner]
    D --> E1[Judge-based metrics]
    D --> E2[Deterministic retrieval metrics]
    E1 --> F[Run results]
    E2 --> F
    F --> G[JSONL persistence]
    G --> H[Statistical layer]
    H --> H1[Bootstrap confidence intervals]
    H --> H2[Paired bootstrap tests]
    H --> H3[Permutation tests]
    H --> I[CLI reports]
    H --> J[FastAPI API]
    J --> K[React dashboard]
```

## Core layers

### 1. Evaluation input

A RAG system is represented as a callable:

```python
(question: str) -> RagOutput
```

This keeps ragval framework-agnostic. The evaluated system can be implemented with LangChain, LlamaIndex, a custom pipeline, or ordinary Python functions.

### 2. Metric execution

Metrics are intentionally split into two families:

- **Judge-based metrics** for qualities such as faithfulness, answer relevance, answer correctness, and context quality.
- **Deterministic retrieval metrics** for exact evaluation against known supporting documents.

Judge calls support caching, retries, and rate limiting so repeated analysis does not need to pay for identical evaluations again.

### 3. Run persistence

Evaluation runs are stored as JSONL. Persisting per-sample results is important because statistical comparisons are paired by sample ID instead of comparing only aggregate means.

### 4. Statistical analysis

The statistics layer produces:

- bootstrap confidence intervals for metric means;
- paired bootstrap tests for configuration differences;
- sign-flip permutation tests;
- calibration statistics for LLM judges against human labels.

This is the central design choice of ragval: a result such as `0.74 vs 0.71` is not treated as meaningful until uncertainty is quantified.

### 5. Interfaces

The same evaluation artifacts can be explored through:

- the Python API;
- the `ragval` CLI;
- a lightweight Streamlit view;
- the full FastAPI + React dashboard.

The web dashboard does not reimplement the evaluation logic. It exposes the existing statistical engine through a REST API and visualizes runs, confidence intervals, paired comparisons, samples, and judge calibration.

## Benchmark data flow

```mermaid
graph TD
    A[HotpotQA-500] --> B[8 RAG configurations]
    B --> C[Generator]
    C --> D[Retrieved context + answer]
    D --> E[Judge metrics]
    D --> F[Deterministic retrieval metrics]
    E --> G[Per-sample JSONL results]
    F --> G
    G --> H[Paired statistical comparisons]
    H --> I[Benchmark tables]
    H --> J[Dashboard]
```

## Reliability and reproducibility choices

- **Seeded statistics** make confidence intervals and tests repeatable.
- **Disk-cached judge calls** reduce cost and keep reruns stable.
- **Per-sample persistence** enables paired comparisons after the original run has finished.
- **Resumable benchmarking** checkpoints partial work so provider quotas or interruptions do not force a full restart.
- **Cross-family generation and judging** reduce same-family judge preference in the published benchmark.

## Dashboard architecture

```mermaid
graph LR
    A[React / Vite frontend] -->|REST| B[FastAPI backend]
    B --> C[ragval Python package]
    C --> D[Saved benchmark runs]
    C --> E[Statistical comparison engine]
    C --> F[Calibration results]
```

For local dashboard setup and deployment details, see [`dashboard/README.md`](../dashboard/README.md).
