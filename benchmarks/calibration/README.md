# Judge calibration files

Before trusting any benchmark number, calibrate the judge:

1. Take ~20 (question, answer, contexts) triples from a completed run
   (`benchmarks/results/partial/<config>.jsonl` is a convenient source —
   pick a mix of what looks like good, mediocre, and bad answers).
2. Score each one **yourself** on the metric's 1–5 rubric (the rubrics are
   the `PROMPT` docstrings in `src/ragval/metrics.py`). Do this *before*
   looking at what the judge said, to avoid anchoring.
3. Write them into a JSONL file here, one example per line:

```json
{"question": "...", "answer": "...", "contexts": ["...", "..."], "ground_truth_answer": "...", "human_score": 4, "metric": "faithfulness"}
```

4. Run:

```bash
ragval calibrate benchmarks/calibration/faithfulness.jsonl --metric faithfulness --judge groq
```

Interpretation (rules of thumb, documented in `ragval/calibration.py`):

- **within-1 agreement ≥ 0.85 and weighted kappa ≥ 0.5** → judge is usable.
- **Spearman ≥ 0.6** → judge at least ranks outputs correctly, so *relative*
  comparisons between configs are trustworthy even if absolute numbers drift.
- **|mean bias| > 0.5** → the judge is systematically lenient or harsh.
  Report comparisons, not absolute scores.

Calibrate each judge-based metric you plan to report (faithfulness,
answer_relevance, answer_correctness at minimum). The deterministic metrics
(retrieval_recall, retrieval_precision) need no calibration.
