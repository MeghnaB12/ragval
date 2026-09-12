from fastapi.testclient import TestClient

from dashboard.backend.main import app

client = TestClient(app)


def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_runs_lists_published_benchmark_configs():
    response = client.get("/api/runs")
    assert response.status_code == 200
    rows = response.json()
    assert len(rows) == 8
    assert rows[0]["config"] == "closed_book"
    assert rows[-1]["config"] == "oracle_cot"
    assert all(row["n_samples"] == 500 for row in rows)


def test_run_report_has_metrics():
    response = client.get("/api/runs/bm25_k3")
    assert response.status_code == 200
    payload = response.json()
    assert payload["config"] == "bm25_k3"
    assert payload["n_samples"] == 500
    metric_names = {row["metric"] for row in payload["metrics"]}
    assert "faithfulness" in metric_names
    assert "answer_correctness" in metric_names


def test_unknown_run_returns_404():
    response = client.get("/api/runs/not-a-config")
    assert response.status_code == 404


def test_compare_returns_paired_results():
    response = client.get("/api/compare", params={"a": "oracle", "b": "bm25_k3"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["config_a"] == "oracle"
    assert payload["config_b"] == "bm25_k3"
    assert payload["rows"]
    assert all(row["n"] == 500 for row in payload["rows"])


def test_samples_respects_limit_and_order():
    response = client.get(
        "/api/samples/bm25_k3",
        params={"metric": "faithfulness", "order": "worst", "limit": 5},
    )
    assert response.status_code == 200
    rows = response.json()
    assert len(rows) == 5
    scores = [row["score"] for row in rows]
    assert scores == sorted(scores)


def test_invalid_sample_order_is_rejected():
    response = client.get(
        "/api/samples/bm25_k3",
        params={"metric": "faithfulness", "order": "sideways"},
    )
    assert response.status_code == 422


def test_calibration_contract():
    response = client.get("/api/calibration")
    assert response.status_code == 200
    payload = response.json()
    assert payload["metric"] == "faithfulness"
    assert payload["n_labels"] == 20
    assert len(payload["judges"]) == 2
