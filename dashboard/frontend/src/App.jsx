import { useEffect, useState } from "react";
import { api } from "./api";
import { Overview } from "./views/Overview";
import { Compare } from "./views/Compare";
import { Samples } from "./views/Samples";
import { Calibration } from "./views/Calibration";

const NAV = [
  { id: "overview", label: "Overview" },
  { id: "compare", label: "Compare" },
  { id: "samples", label: "Sample explorer" },
  { id: "calibration", label: "Judge calibration" },
];

export function App() {
  const [runs, setRuns] = useState([]);
  const [view, setView] = useState("overview");
  const [focus, setFocus] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    api.runs().then(setRuns).catch((e) => setError(e.message));
  }, []);

  const pickConfig = (config) => {
    setFocus(config);
    setView("compare");
  };

  return (
    <div className="app">
      <aside className="rail">
        <div className="brand">ragval<span className="dot">.</span></div>
        <div className="brand-sub">rag evaluation, with error bars</div>
        <nav className="nav">
          <div className="nav-label">Views</div>
          {NAV.map((n) => (
            <button
              key={n.id}
              className={`nav-item ${view === n.id ? "active" : ""}`}
              onClick={() => setView(n.id)}
            >
              <span className="tick" />
              {n.label}
            </button>
          ))}
        </nav>
        <div style={{ marginTop: 32 }}>
          <div className="nav-label">Source</div>
          <a
            className="nav-item"
            href="https://github.com/MeghnaB12/ragval"
            target="_blank"
            rel="noreferrer"
            style={{ color: "var(--muted)" }}
          >
            <span className="tick" />
            github.com/MeghnaB12/ragval
          </a>
        </div>
      </aside>

      <main className="main">
        {error ? (
          <div className="empty">
            Can't reach the API ({error}). Start the backend with{" "}
            <span className="mono">uvicorn main:app --port 8000</span>.
          </div>
        ) : runs.length === 0 ? (
          <div className="loading">Connecting to ragval API…</div>
        ) : view === "overview" ? (
          <Overview runs={runs} onPick={pickConfig} />
        ) : view === "compare" ? (
          <Compare runs={runs} initialA={focus} />
        ) : view === "samples" ? (
          <Samples runs={runs} initialConfig={focus} />
        ) : (
          <Calibration />
        )}
      </main>
    </div>
  );
}
