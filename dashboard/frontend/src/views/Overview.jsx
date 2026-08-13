import { useEffect, useState } from "react";
import { api } from "../api";
import { Whisker } from "../Whisker";

const HEADLINE = "answer_correctness";

export function Overview({ runs, onPick }) {
  const [reports, setReports] = useState({});
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let alive = true;
    Promise.all(runs.map((r) => api.run(r.config)))
      .then((list) => {
        if (!alive) return;
        const byConfig = {};
        list.forEach((rep) => (byConfig[rep.config] = rep));
        setReports(byConfig);
        setLoading(false);
      })
      .catch(() => setLoading(false));
    return () => (alive = false);
  }, [runs]);

  if (loading) return <div className="loading">Loading runs…</div>;

  const meta = runs[0] || {};

  return (
    <>
      <div className="page-eyebrow">8 configs · HotpotQA-500</div>
      <h1 className="page-title">Benchmark overview</h1>
      <p className="page-lede">
        Eight retrieval and prompting configurations, each evaluated on 500 questions.
        Every bar is a mean with its 95% bootstrap confidence interval — the width is
        how sure the number is. Click a config to see it against the others.
      </p>

      <div className="stat-strip">
        <div className="stat-cell">
          <div className="k">Configs</div>
          <div className="v">{runs.length}</div>
        </div>
        <div className="stat-cell">
          <div className="k">Samples each</div>
          <div className="v">{meta.n_samples}</div>
        </div>
        <div className="stat-cell">
          <div className="k">Generator</div>
          <div className="v" style={{ fontSize: 13 }}>{meta.generator}</div>
        </div>
        <div className="stat-cell">
          <div className="k">Judge</div>
          <div className="v" style={{ fontSize: 13 }}>{meta.judge}</div>
        </div>
      </div>

      <div className="panel panel-pad">
        <div style={{ color: "var(--muted)", fontSize: 12, textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 14 }}>
          answer_correctness — mean & 95% CI
        </div>
        {runs.map((r) => {
          const rep = reports[r.config];
          const m = rep?.metrics.find((x) => x.metric === HEADLINE);
          if (!m) return null;
          return (
            <div
              key={r.config}
              className="metric-row"
              style={{ cursor: "pointer" }}
              onClick={() => onPick(r.config)}
              role="button"
              tabIndex={0}
              onKeyDown={(e) => e.key === "Enter" && onPick(r.config)}
            >
              <div className="metric-name">{r.config}</div>
              <Whisker mean={m.mean} low={m.ci_low} high={m.ci_high} color="var(--signal)" />
              <div className="metric-val">
                <b>{m.mean.toFixed(3)}</b> [{m.ci_low.toFixed(2)}–{m.ci_high.toFixed(2)}]
              </div>
            </div>
          );
        })}
      </div>

      <p className="footer-note">
        The scale is fixed 0–1 for every bar, so intervals are directly comparable.
        Where two intervals barely overlap, the Compare tab tells you whether the
        difference is real with a paired significance test.
      </p>
    </>
  );
}
