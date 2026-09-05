# ragval dashboard frontend

React + Vite frontend for the [`ragval`](../../README.md) benchmark dashboard.

**Live app:** https://ragval.vercel.app

The UI is intentionally thin: it does not reimplement evaluation or statistical logic. It calls the FastAPI backend, which delegates confidence intervals, paired comparisons, and run loading to the core `ragval` Python package.

## Views

- **Overview** — benchmark configurations with mean answer-correctness scores and 95% bootstrap confidence intervals.
- **Compare** — paired per-sample configuration comparisons with bootstrap and sign-flip permutation tests.
- **Sample explorer** — inspect individual answers and stored judge reasoning, sorted by metric score.
- **Judge calibration** — view published judge-vs-human agreement statistics and calibration caveats.

## Local development

From `dashboard/frontend`:

```bash
npm install
npm run dev
```

The Vite dev server proxies `/api` to `http://localhost:8000`. Start the backend separately from `dashboard/backend`:

```bash
pip install -e ../..
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

## Production configuration

The frontend reads the API origin from:

```env
VITE_API_URL=https://your-ragval-api.example.com
```

If `VITE_API_URL` is absent, the client falls back to `http://localhost:8000`.

Build the static bundle with:

```bash
npm run build
```

## Quality checks

```bash
npm run lint
npm run build
```

## Stack

React 19 · Vite · Oxlint · plain CSS · custom SVG confidence-interval visualizations.
