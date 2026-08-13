import { useEffect, useState } from "react";
import { api } from "../api";

const METRICS = ["faithfulness", "answer_correctness", "retrieval_recall", "retrieval_precision"];

function scoreColor(s) {
  if (s >= 0.75) return { color: "var(--signal)", background: "var(--signal-dim)" };
  if (s >= 0.4) return { color: "#e0c07a", background: "#5c4f2e" };
  return { color: "var(--warn)", background: "var(--warn-dim)" };
}

export function Samples({ runs, initialConfig }) {
  const configs = runs.map((r) => r.config);
  const [config, setConfig] = useState(initialConfig || "bm25_k3");
  const [metric, setMetric] = useState("faithfulness");
  const [order, setOrder] = useState("worst");
  const [rows, setRows] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    setLoading(true);
    api
      .samples(config, metric, order, 40)
      .then((d) => {
        setRows(d);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, [config, metric, order]);

  return (
    <>
      <div className="page-eyebrow">judge transcripts</div>
      <h1 className="page-title">Sample explorer</h1>
      <p className="page-lede">
        The view a results table can't give you: for any config, the actual answers
        and the judge's own reasoning for each score. Sorted worst-first — read these
        to understand <em>why</em> a config loses, not just that it does.
      </p>

      <div className="controls">
        <div className="field">
          <label>Config</label>
          <select value={config} onChange={(e) => setConfig(e.target.value)}>
            {configs.map((c) => <option key={c} value={c}>{c}</option>)}
          </select>
        </div>
        <div className="field">
          <label>Metric</label>
          <select value={metric} onChange={(e) => setMetric(e.target.value)}>
            {METRICS.map((m) => <option key={m} value={m}>{m}</option>)}
          </select>
        </div>
        <div className="field">
          <label>Order</label>
          <select value={order} onChange={(e) => setOrder(e.target.value)}>
            <option value="worst">Worst first</option>
            <option value="best">Best first</option>
          </select>
        </div>
      </div>

      {loading ? (
        <div className="loading">Loading samples…</div>
      ) : rows.length === 0 ? (
        <div className="empty">No samples for this metric.</div>
      ) : (
        rows.map((r) => (
          <div className="sample" key={r.sample_id}>
            <div className="sample-head">
              <span className="score-chip mono" style={scoreColor(r.score)}>
                {r.score.toFixed(2)}
              </span>
              <span className="mono" style={{ color: "var(--muted-dim)", fontSize: 11 }}>
                {r.sample_id.slice(0, 12)} · {r.n_contexts} context{r.n_contexts === 1 ? "" : "s"}
              </span>
            </div>
            <div className="sample-answer">
              <span className="lbl">Answer</span>
              {r.answer}
            </div>
            <div className="sample-why">
              <span className="lbl">Judge reasoning</span>
              {r.reasoning}
            </div>
          </div>
        ))
      )}
    </>
  );
}
