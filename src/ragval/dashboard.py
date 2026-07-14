"""Streamlit dashboard for browsing runs, CIs, comparisons, and judge reasoning.

Run with:
    streamlit run src/ragval/dashboard.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ragval.runs import list_runs, load_run
from ragval.stats import compare_all_metrics, summarize_run

RESULTS_DIR = Path("benchmarks") / "results"

st.set_page_config(page_title="ragval", layout="wide")
st.title("ragval — rigorous RAG evaluation")

headers = list_runs(RESULTS_DIR)
if not headers:
    st.warning(f"No runs found in `{RESULTS_DIR}`. Run the benchmark first.")
    st.stop()

run_labels = {f"{h['config_name']}  ({h['run_id']})": h["_path"] for h in headers}

tab_overview, tab_compare, tab_samples = st.tabs(
    ["Run overview", "Compare runs", "Sample explorer"]
)

with tab_overview:
    choice = st.selectbox("Run", list(run_labels), key="overview_run")
    run = load_run(run_labels[choice])
    summaries = summarize_run(run)

    st.caption(
        f"dataset={run.dataset_name} · n={len(run.samples)} · cost=${run.total_cost_usd:.4f}"
    )

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=[s.metric_name for s in summaries],
            y=[s.mean for s in summaries],
            error_y={
                "type": "data",
                "symmetric": False,
                "array": [s.ci_high - s.mean for s in summaries],
                "arrayminus": [s.mean - s.ci_low for s in summaries],
            },
        )
    )
    fig.update_layout(yaxis_range=[0, 1], yaxis_title="score", title="Metric means with 95% CI")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        pd.DataFrame(
            [
                {
                    "metric": s.metric_name,
                    "mean": round(s.mean, 3),
                    "ci_low": round(s.ci_low, 3),
                    "ci_high": round(s.ci_high, 3),
                    "std": round(s.std, 3),
                    "n": s.n,
                }
                for s in summaries
            ]
        ),
        use_container_width=True,
    )

with tab_compare:
    col1, col2 = st.columns(2)
    with col1:
        a_choice = st.selectbox("Run A", list(run_labels), key="cmp_a")
    with col2:
        b_choice = st.selectbox(
            "Run B", list(run_labels), index=min(1, len(run_labels) - 1), key="cmp_b"
        )

    if a_choice == b_choice:
        st.info("Pick two different runs.")
    else:
        run_a = load_run(run_labels[a_choice])
        run_b = load_run(run_labels[b_choice])
        comparisons = compare_all_metrics(run_a, run_b)
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "metric": c.metric_name,
                        "n": c.n,
                        run_a.config_name: round(c.mean_a, 3),
                        run_b.config_name: round(c.mean_b, 3),
                        "diff": round(c.mean_diff, 3),
                        "diff 95% CI": f"[{c.diff_ci_low:+.3f}, {c.diff_ci_high:+.3f}]",
                        "p (bootstrap)": round(c.p_value_bootstrap, 4),
                        "p (permutation)": round(c.p_value_permutation, 4),
                        "significant": "✅" if c.significant else "—",
                    }
                    for c in comparisons
                ]
            ),
            use_container_width=True,
        )
        st.caption(
            "Paired tests on per-sample score differences (aligned by sample_id). "
            "A significant result means the difference is unlikely to be noise at 95% confidence."
        )

with tab_samples:
    choice = st.selectbox("Run", list(run_labels), key="samples_run")
    run = load_run(run_labels[choice])
    metric_names = run.metric_names()
    metric = st.selectbox("Sort by metric (ascending — worst first)", metric_names)

    rows = []
    for s in run.samples:
        if metric in s.metrics:
            rows.append(
                {
                    "sample_id": s.sample_id,
                    "score": s.metrics[metric].score,
                    "answer": s.rag_output.answer[:160],
                    "reasoning": s.metrics[metric].reasoning[:200],
                }
            )
    df = pd.DataFrame(rows).sort_values("score")
    st.dataframe(df, use_container_width=True, height=500)
