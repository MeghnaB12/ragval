// The signature element: a 95% confidence interval drawn as a real error bar.
// A point marks the mean; a horizontal whisker spans [ci_low, ci_high] on a
// fixed 0..1 scale (all ragval metrics are bounded in [0,1]). This one motif
// carries the project's whole thesis — every number shows its uncertainty.

export function Whisker({ mean, low, high, color = "var(--paper)", height = 20 }) {
  const pct = (v) => `${Math.max(0, Math.min(1, v)) * 100}%`;
  return (
    <svg width="100%" height={height} style={{ overflow: "visible", display: "block" }}>
      {/* baseline track */}
      <line x1="0" x2="100%" y1="50%" y2="50%" stroke="var(--line)" strokeWidth="1" />
      {/* interval bar */}
      <line x1={pct(low)} x2={pct(high)} y1="50%" y2="50%" stroke={color} strokeWidth="2" opacity="0.55" />
      {/* end caps */}
      <line x1={pct(low)} x2={pct(low)} y1="30%" y2="70%" stroke={color} strokeWidth="1.5" opacity="0.7" />
      <line x1={pct(high)} x2={pct(high)} y1="30%" y2="70%" stroke={color} strokeWidth="1.5" opacity="0.7" />
      {/* mean point */}
      <circle cx={pct(mean)} cy="50%" r="3.5" fill={color} />
    </svg>
  );
}

// A diff whisker centered on zero, spanning [-1, 1]. Green if the CI excludes
// zero (significant), warn-colored if it straddles zero.
export function DiffWhisker({ diff, low, high, height = 20 }) {
  const toPct = (v) => `${((Math.max(-1, Math.min(1, v)) + 1) / 2) * 100}%`;
  const significant = low > 0 || high < 0;
  const color = significant ? "var(--signal)" : "var(--warn)";
  return (
    <svg width="100%" height={height} style={{ overflow: "visible", display: "block" }}>
      <line x1="0" x2="100%" y1="50%" y2="50%" stroke="var(--line)" strokeWidth="1" />
      {/* zero reference */}
      <line x1="50%" x2="50%" y1="15%" y2="85%" stroke="var(--line-bright)" strokeWidth="1" strokeDasharray="2 2" />
      <line x1={toPct(low)} x2={toPct(high)} y1="50%" y2="50%" stroke={color} strokeWidth="2" opacity="0.55" />
      <line x1={toPct(low)} x2={toPct(low)} y1="30%" y2="70%" stroke={color} strokeWidth="1.5" opacity="0.7" />
      <line x1={toPct(high)} x2={toPct(high)} y1="30%" y2="70%" stroke={color} strokeWidth="1.5" opacity="0.7" />
      <circle cx={toPct(diff)} cy="50%" r="3.5" fill={color} />
    </svg>
  );
}
