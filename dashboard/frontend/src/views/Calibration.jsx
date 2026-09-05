import { useEffect, useState } from "react";
import { api } from "../api";

export function Calibration() {
  const [data, setData] = useState(null);
  useEffect(() => {
    api.calibration().then(setData).catch(() => {});
  }, []);

  if (!data) return <div className="loading">Loading calibration…</div>;

  return (
    <>
      <div className="page-eyebrow">judge vs human · n={data.n_labels}</div>
      <h1 className="page-title">Judge calibration</h1>
      <p className="page-lede">
        Every judge-based score above is assigned by an LLM. How closely does that
        measurement agree with a human-labeled sample? These statistics compare two
        judges from different model families against {data.n_labels} blind human labels:
        Claude (the published benchmark judge) and Llama (an independent cross-check).
      </p>

      <div className="panel">
        <table className="data">
          <thead>
            <tr>
              <th>judge</th>
              <th>role</th>
              <th>within-1 agree</th>
              <th>weighted κ</th>
              <th>spearman</th>
              <th>mean bias</th>
            </tr>
          </thead>
          <tbody>
            {data.judges.map((j) => (
              <tr key={j.judge}>
                <td style={{ color: "var(--paper)" }}>{j.judge}</td>
                <td style={{ color: "var(--muted)" }}>{j.role}</td>
                <td>{j.within_one_agreement.toFixed(2)}</td>
                <td>{j.quadratic_weighted_kappa.toFixed(3)}</td>
                <td>{j.spearman.toFixed(3)}</td>
                <td className={j.mean_bias < -0.2 ? "neg" : ""}>
                  {j.mean_bias >= 0 ? "+" : ""}{j.mean_bias.toFixed(2)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="panel panel-pad" style={{ marginTop: 20 }}>
        <div style={{ color: "var(--signal)", fontFamily: "var(--font-mono)", fontSize: 11, textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 8 }}>
          What this means
        </div>
        <p style={{ color: "var(--muted)", fontSize: 13.5 }}>{data.note}</p>
      </div>

      <p className="footer-note">
        Negative bias means a judge scored this calibration sample lower than the human
        labels on average. With only {data.n_labels} labels, these results are a diagnostic,
        not proof that every benchmark judgment is correct. They make judge uncertainty
        visible and help distinguish relative comparisons from absolute score claims.
      </p>
    </>
  );
}
