"""FastAPI backend (§18) - serves predictions, history, indicators, and model results.

Run it with:
    uvicorn api.main:app --reload

Then open http://127.0.0.1:8000/docs for FastAPI's auto-generated,
interactive API docs (built entirely from the type hints and Pydantic
models below).
"""
import json
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from sqlalchemy import bindparam, text

sys.path.append(str(Path(__file__).resolve().parent.parent))
from db.connection import get_engine
from ml.backtest import MODEL_PERFORMANCE_PATH
from api.schemas import HistoricalPoint, IndicatorValue, ModelPerformanceEntry, PredictionResponse

app = FastAPI(title="Currency Predictor API")

SUPPORTED_PAIRS = ["MYR/IDR"]

# Feature names worth surfacing as "economic factors" (§19), with a
# human-readable label for each. Anything in the features table not
# listed here just isn't shown on this endpoint.
INDICATOR_FEATURES = {
    "interest_rate_malaysia": "Malaysia OPR",
    "interest_rate_indonesia": "Indonesia Interest Rate",
    "brent_price": "Brent Crude",
    "wti_price": "WTI Crude",
    "palm_oil_price": "Palm Oil",
    "sentiment_score": "News Sentiment (GDELT)",
}


def normalize_pair(pair: str) -> str:
    """URLs use MYR-IDR (a slash isn't safe in a URL path); the database
    stores "MYR/IDR". Also where an unsupported pair gets rejected with a
    clean 404 instead of an empty/confusing result further down.
    """
    currency_pair = pair.replace("-", "/").upper()
    if currency_pair not in SUPPORTED_PAIRS:
        raise HTTPException(status_code=404, detail=f"Unknown currency pair: {pair}")
    return currency_pair


@app.get("/currencies")
def list_currencies():
    return {"currencies": SUPPORTED_PAIRS}


@app.get("/prediction/{pair}", response_model=PredictionResponse)
def get_prediction(pair: str):
    """WORKED EXAMPLE - read this one first.

    `{pair}` in the route path is a PATH PARAMETER - FastAPI passes
    whatever the caller put there (e.g. "MYR-IDR") into the `pair`
    argument automatically, no manual URL parsing needed.

    `response_model=PredictionResponse` tells FastAPI "whatever this
    function returns, validate and serialize it as a PredictionResponse."
    Below we build a dict; FastAPI converts it for us and would raise a
    clear error if a field were missing or the wrong type - catching bugs
    before a broken response ever reaches a caller.
    """
    currency_pair = normalize_pair(pair)
    engine = get_engine()

    with engine.connect() as conn:
        current_row = conn.execute(
            text("SELECT date, close FROM exchange_rates WHERE currency_pair = :p ORDER BY date DESC LIMIT 1"),
            {"p": currency_pair},
        ).mappings().first()

        prediction_row = conn.execute(
            text("SELECT * FROM predictions WHERE currency_pair = :p ORDER BY timestamp DESC LIMIT 1"),
            {"p": currency_pair},
        ).mappings().first()

    if current_row is None or prediction_row is None:
        # HTTPException is how FastAPI routes report an error to the caller -
        # it turns into a proper HTTP 404 response, not a crash.
        raise HTTPException(
            status_code=404,
            detail=f"No prediction available for {currency_pair} yet - run ml/generate_prediction.py first",
        )

    current_rate = float(current_row["close"])
    predicted_rate = float(prediction_row["predicted_rate"])

    return {
        "currency_pair": currency_pair,
        "as_of_date": current_row["date"],
        "current_rate": current_rate,
        "predicted_rate": predicted_rate,
        "expected_movement_pct": (predicted_rate / current_rate - 1) * 100,
        "direction": prediction_row["direction"],
        "lower_bound": float(prediction_row["lower_bound"]),
        "upper_bound": float(prediction_row["upper_bound"]),
        "confidence": float(prediction_row["confidence"]),
        "model_version": prediction_row["model_version"],
        "generated_at": prediction_row["timestamp"],
    }


@app.get("/historical/{pair}", response_model=list[HistoricalPoint])
def get_historical(pair: str):
    """Every (date, close) row for this pair, oldest first - feeds the historical chart (§19/§20)."""
    currency_pair = normalize_pair(pair)
    engine = get_engine()

    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT date, close FROM exchange_rates WHERE currency_pair = :p ORDER BY date"),
            {"p": currency_pair},
        ).mappings().all()

    return [{"date": row["date"], "close": float(row["close"])} for row in rows]


@app.get("/indicators/{pair}", response_model=list[IndicatorValue])
def get_indicators(pair: str):
    """Latest value of each feature in INDICATOR_FEATURES (§19/§20 "Economic Factors").

    The features table has many rows per feature_name (one per date), so a
    subquery picks out just the most recent date's rows. `feature_name IN
    :names` needs the bind parameter marked as "expanding" - a plain :name
    only ever substitutes ONE value, but IN needs to expand to N values
    (one per entry in INDICATOR_FEATURES), which is what bindparam(...,
    expanding=True) tells SQLAlchemy to do.
    """
    currency_pair = normalize_pair(pair)
    engine = get_engine()

    query = text(
        """
        SELECT feature_name, feature_value, date
        FROM features
        WHERE currency_pair = :p
          AND feature_name IN :names
          AND date = (SELECT MAX(date) FROM features WHERE currency_pair = :p)
        """
    ).bindparams(bindparam("names", expanding=True))

    with engine.connect() as conn:
        rows = conn.execute(query, {"p": currency_pair, "names": list(INDICATOR_FEATURES.keys())}).mappings().all()

    return [
        {
            "name": INDICATOR_FEATURES[row["feature_name"]],
            "value": float(row["feature_value"]),
            "as_of_date": row["date"],
        }
        for row in rows
    ]


@app.get("/model-performance", response_model=list[ModelPerformanceEntry])
def get_model_performance():
    """Reads the cached backtest summary (data/processed/model_performance.json)
    instead of touching Postgres or re-running the full walk-forward
    backtest (6 models x 5 years) on every request - see ml/backtest.py's
    save_model_performance().
    """
    if not MODEL_PERFORMANCE_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="No cached model performance yet - run ml/backtest.py first",
        )

    payload = json.loads(MODEL_PERFORMANCE_PATH.read_text())
    return payload["models"]
