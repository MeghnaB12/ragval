import { useEffect, useState } from "react";
import { api } from "../api";
import { DiffWhisker } from "../Whisker";

export function Compare({ runs, initialA }) {
  const configs = runs.map((r) => r.config);
  const [a, setA] = useState(initialA || "oracle");
  const [b, setB] = useState("bm25_k3");
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (a === b) return;
    setLoading(true);
    api
      .compare(a, b)
      .then((d) => {
        setData(d);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, [a, b]);

  return (
    <>
      <div className="page-eyebrow">paired · per-sample · n=500</div>
      <h1 className="page-title">Compare configs</h1>
      <p className="page-lede">
        Is the gap between two configs real, or noise? Each metric is compared
        sample-by-sample (paired by question ID), then run through a bootstrap and a
        sign-flip permutation test. The whisker shows the difference and its 95% CI;
        if the interval clears zero, the difference is significant.
      </p>

      <div className="controls">
        <div className="field">
          <label>Config A</label>
          <select value={a} onChange={(e) => setA(e.target.value)}>
            {configs.map((c) => <option key={c} value={c}>{c}</option>)}
          </select>
        </div>
        <div className="vs">vs</div>
        <div className="field">
          <label>Config B (baseline)</label>
          <select value={b} onChange={(e) => setB(e.target.value)}>
            {configs.map((c) => <option key={c} value={c}>{c}</option>)}
          </select>
        </div>
      </div>

      {a === b ? (
        <div className="empty">Pick two different configs.</div>
      ) : loading || !data ? (
        <div className="loading">Running paired tests…</div>
      ) : (
        <div className="panel">
          <table className="data">
            <thead>
              <tr>
                <th>metric</th>
                <th>{a}</th>
                <th>{b}</th>
                <th>diff</th>
                <th style={{ width: 180 }}>diff & 95% CI (vs 0)</th>
                <th>p (boot)</th>
                <th>verdict</th>
              </tr>
            </thead>
            <tbody>
              {data.rows.map((r) => (
                <tr key={r.metric}>
                  <td style={{ color: "var(--paper)" }}>{r.metric}</td>
                  <td>{r.mean_a.toFixed(3)}</td>
                  <td>{r.mean_b.toFixed(3)}</td>
                  <td className={r.diff >= 0 ? "pos" : "neg"}>
                    {r.diff >= 0 ? "+" : ""}{r.diff.toFixed(3)}
                  </td>
                  <td style={{ padding: "6px 12px" }}>
                    <DiffWhisker diff={r.diff} low={r.diff_ci_low} high={r.diff_ci_high} />
                  </td>
                  <td>{r.p_bootstrap.toFixed(4)}</td>
                  <td>
                    <span className={`tag ${r.significant ? "sig" : "nsig"}`}>
                      {r.significant ? "significant" : "noise"}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <p className="footer-note">
        A "significant" verdict means the 95% CI on the paired difference excludes zero
        (p &lt; 0.05, bootstrap). Pairing removes question-difficulty variance, which
        makes the test far more sensitive than comparing the two means in isolation.
      </p>
    </>
  );
}
