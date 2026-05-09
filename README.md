# ragval

> Rigorous RAG evaluation with confidence intervals, significance testing, and judge calibration.

**Status:** 🚧 Under active development. Targeting v0.1 launch in early June 2026.

## Why ragval

Most RAG evaluation tools tell you that config A scored 0.74 and config B scored 0.71. They don't tell you whether that difference is real or noise.

`ragval` is built around three principles other RAG eval tools handle loosely:

1. **Statistical rigor.** Every metric reports a 95% bootstrap confidence interval. Every comparison reports a paired significance test. Every benchmark answers the question "is this difference real?"
2. **Judge calibration.** Provide ~20 human-labeled examples; `ragval` reports how often your LLM judge agrees with them. If your judge isn't calibrated, your numbers are theater.
3. **Reproducibility by default.** Aggressive disk caching of judge calls. Every run is a single JSONL file. Every result is reproducible from the same inputs.

## Status

- [x] Core data types
- [x] Judge abstraction (Gemini, Groq, mock)
- [x] Disk caching
- [x] Faithfulness + answer relevance metrics
- [x] Runner
- [x] Smoke test
- [ ] Context precision + context recall (week 2)
- [ ] Statistical layer (week 2)
- [ ] Judge calibration (week 2)
- [ ] CLI (week 3)
- [ ] Streamlit dashboard (week 3)
- [ ] HotpotQA-500 benchmark (week 3)
- [ ] Blog post (week 4)

## Install (dev)

```bash
git clone https://github.com/yourname/ragval
cd ragval
pip install -e ".[dev,dashboard]"
```

## Try it

```bash
export GEMINI_API_KEY=...   # free at https://aistudio.google.com/apikey
python -m ragval.smoke_test
```

## Design

A RAG system in `ragval` is just a callable: `(question: str) -> RagOutput`. This means you can plug in anything — LangChain, LlamaIndex, raw functions, your production code. No framework lock-in.

```python
from ragval import EvalSample, RagOutput
from ragval.judges import GeminiJudge
from ragval.metrics import Faithfulness, AnswerRelevance
from ragval.runner import run_eval

def my_rag(question: str) -> RagOutput:
    # whatever you want here
    ...

dataset = [EvalSample(id="1", question="...", ground_truth_answer="...")]
judge = GeminiJudge()
result = run_eval(my_rag, dataset, [Faithfulness(), AnswerRelevance()], judge)
```

## Comparison with other tools

(To be filled in honestly after week 2.)

Week 1 day 1: scaffold + types + judges + metrics + runner

- Project setup: pyproject.toml, ruff, pytest, GitHub Actions
- Core data types: EvalSample, RagOutput, MetricResult, RunResult
- Judge abstraction: GeminiJudge (2.5-flash), GroqJudge, MockJudge
  with disk caching, rate limiting, exponential-backoff retry
- Metrics: faithfulness and answer_relevance with explicit rubrics
  and tolerant JSON parsing
- Runner orchestrating RAG + metrics over a dataset
- Smoke test confirms metrics discriminate good vs partial answers
- 18 passing tests

## License

MIT

