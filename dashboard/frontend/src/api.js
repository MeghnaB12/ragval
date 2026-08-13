// Thin API client. Base URL comes from VITE_API_URL at build time so the same
// bundle works locally (proxy to :8000) and in production (deployed API URL).
const BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

async function get(path) {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

export const api = {
  runs: () => get("/api/runs"),
  run: (config) => get(`/api/runs/${config}`),
  compare: (a, b) => get(`/api/compare?a=${a}&b=${b}`),
  samples: (config, metric, order = "worst", limit = 40) =>
    get(`/api/samples/${config}?metric=${metric}&order=${order}&limit=${limit}`),
  calibration: () => get("/api/calibration"),
};
