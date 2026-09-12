# ragval dashboard

A full-stack web app for exploring the ragval benchmark results — a FastAPI
backend serving the statistical engine, and a React frontend that visualizes
confidence intervals, paired significance tests, judge reasoning, and calibration.

**Live demo: [ragval.vercel.app](https://ragval.vercel.app)**

![Benchmark overview](docs/overview.png)

The four views:

| Compare (paired significance) | Sample explorer (judge reasoning) |
|---|---|
| ![Compare](docs/compare.png) | ![Samples](docs/samples.png) |

| Judge calibration | Mobile |
|---|---|
| ![Calibration](docs/calibration.png) | ![Mobile](docs/mobile.png) |

## Why this exists

The benchmark results live in the repo as JSONL run files and a README table.
That's reproducible but not *explorable* — you can't sort samples by score, read
the judge's reasoning for a given failure, or flip between config comparisons.
This dashboard reads the same run files and calls the same `ragval.stats` functions
the CLI uses, so the dashboard and CLI share one statistical implementation.

## Architecture

```text
 React (Vite)  ──HTTP──▶  FastAPI  ──calls──▶  ragval.stats / ragval.runs
 frontend/                backend/             (the core engine)
```

- **Backend** (`backend/main.py`) — a thin REST layer over the committed benchmark
  runs. Pydantic response models define the API contract and are auto-documented at `/docs`.
- **Frontend** (`frontend/`) — React + Vite, plain CSS design tokens, and custom
  SVG confidence-interval visualizations. See [`frontend/README.md`](frontend/README.md).
- **Production-like local stack** — Docker Compose builds a non-root FastAPI image
  and a multi-stage React/Nginx image, each with a health check.

## Endpoints

| method | path | returns |
|---|---|---|
| GET | `/api/health` | service health |
| GET | `/api/runs` | the 8 benchmark configs with run metadata |
| GET | `/api/runs/{config}` | per-metric mean + 95% bootstrap CI |
| GET | `/api/compare?a=&b=` | paired significance tests between two configs |
| GET | `/api/samples/{config}?metric=&order=` | per-sample scores + judge reasoning |
| GET | `/api/calibration` | published judge-vs-human agreement summary |

Interactive API docs are available at `http://localhost:8000/docs` once the backend is running.

## Run locally for development

One command from this directory:

```bash
./dev.sh
```

Or manually from the repository root, in two terminals:

```bash
# terminal 1 — backend
pip install -e ".[dashboard]"
uvicorn dashboard.backend.main:app --reload --port 8000

# terminal 2 — frontend
cd dashboard/frontend
npm ci
npm run dev
```

The frontend proxies `/api` to the backend in development.

## Run the production-like stack

From the repository root:

```bash
docker compose up --build
```

Then open:

- Web UI: `http://localhost:8080`
- API: `http://localhost:8000/api/health`
- OpenAPI docs: `http://localhost:8000/docs`

The Compose setup restricts API CORS to `http://localhost:8080`. For a deployed
backend, set `RAGVAL_CORS_ORIGINS` to the exact comma-separated frontend origins.

## Deploy

The frontend is a static bundle; the backend is a small Python service.

**Backend:**
- Build/install: `pip install -e ".[dashboard]"`
- Start: `uvicorn dashboard.backend.main:app --host 0.0.0.0 --port $PORT`
- Environment: `RAGVAL_CORS_ORIGINS=https://your-frontend.example`

**Frontend:**
- Root directory: `dashboard/frontend`
- Build command: `npm ci && npm run build`
- Output directory: `dist`
- Build-time environment variable: `VITE_API_URL=https://your-api.example`

Containerized deployment can use `dashboard/backend/Dockerfile` and
`dashboard/frontend/Dockerfile` directly.

## CI coverage

Pull requests validate all three application surfaces:

- Python 3.10 / 3.11 / 3.12: Ruff, formatting, mypy, pytest + coverage, package build
- React: deterministic `npm ci`, Oxlint, production Vite build
- Containers: Compose validation plus backend/frontend image builds

## Stack

FastAPI · Pydantic · React 19 · Vite · Oxlint · Docker · Nginx · GitHub Actions.
