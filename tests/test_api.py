"""Tests for the FastAPI app (api/main.py).

Run with: pytest tests/test_api.py -v

Unlike the ML tests (which use small synthetic data), these hit the real
local Postgres database - that's normal for API tests, since the whole
point of an API is to serve real data. Make sure Docker/Postgres is
running and the ingestion + feature + backtest + prediction scripts have
been run at least once before running these.
"""
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_list_currencies():
    resp = client.get("/currencies")
    assert resp.status_code == 200
    assert "MYR/IDR" in resp.json()["currencies"]


def test_get_prediction_worked_example():
    """This one should already pass - it's the fully-written example."""
    resp = client.get("/prediction/MYR-IDR")
    assert resp.status_code == 200

    body = resp.json()
    assert body["currency_pair"] == "MYR/IDR"
    assert isinstance(body["predicted_rate"], float)
    assert isinstance(body["current_rate"], float)
    assert body["direction"] in {"Bullish", "Bearish", "Neutral"}


def test_get_prediction_unknown_pair_is_404():
    resp = client.get("/prediction/USD-JPY")
    assert resp.status_code == 404


def test_get_historical():
    resp = client.get("/historical/MYR-IDR")
    assert resp.status_code == 200

    body = resp.json()
    assert isinstance(body, list)
    assert len(body) > 1000  # several years of daily data
    assert set(body[0].keys()) == {"date", "close"}
    assert body[0]["date"] < body[-1]["date"]  # oldest first, not newest first


def test_get_indicators():
    resp = client.get("/indicators/MYR-IDR")
    assert resp.status_code == 200

    body = resp.json()
    names = {row["name"] for row in body}
    assert "Malaysia OPR" in names
    assert "Brent Crude" in names


def test_get_model_performance():
    resp = client.get("/model-performance")
    assert resp.status_code == 200

    body = resp.json()
    model_names = {row["model"] for row in body}
    assert "xgboost" in model_names
    assert "naive" in model_names
