# Contributing to ragval

Thanks for your interest in improving ragval.

## Development setup

```bash
git clone https://github.com/MeghnaB12/ragval.git
cd ragval
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,reference]"
pre-commit install
```

Run the Python quality checks before opening a pull request:

```bash
ruff check src tests
ruff format --check src tests
pytest
```

## What makes a good contribution

Useful contributions include:

- new evaluation metrics with explicit scoring contracts and tests;
- statistical methods that preserve per-sample pairing and reproducibility;
- judge-provider integrations with caching, retry, and cost metadata;
- dashboard improvements that reuse the core `ragval` statistical engine rather than duplicating calculations;
- documentation, examples, and reproducibility improvements.

## Design constraints

Please keep these principles intact:

1. **Per-sample results are first-class.** Aggregate-only evaluation makes paired analysis impossible.
2. **Statistics must be reproducible.** New resampling methods should expose a seed where applicable.
3. **LLM judges are measurement instruments, not ground truth.** Judge-based metrics should preserve raw scores/reasoning and remain calibratable.
4. **The evaluated RAG system stays framework-agnostic.** Core APIs should not require LangChain, LlamaIndex, or another orchestration framework.
5. **Dashboard calculations belong in the core library.** The API/UI should present results, not create a second statistical implementation.

## Pull requests

Keep PRs focused. Include tests for behavior changes and explain any statistical or evaluation assumptions in the PR description.

Do not commit API keys, `.env` files, local caches, generated benchmark partials, or provider credentials.
