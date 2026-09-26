"""FastAPI backend (§18) - serves predictions, history, indicators, and model results.

Run it with:
    uvicorn api.main:app --reload

Then open http://127.0.0.1:8000/docs for FastAPI's auto-generated,
interactive API docs (built entirely from the type hints and Pydantic
models below - a big part of why FastAPI is worth learning).

WORKED EXAMPLE: get_prediction() below is written out in full, with heavy
comments, as the pattern for the other three routes. YOUR TURN covers
get_historical, get_indicators, and get_model_performance further down -
each follows the same shape: read from Postgres (or the cached JSON, for
model performance), shape the result into the matching schema from
api/schemas.py, return it.
"""
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from sqlalchemy import text

sys.path.append(str(Path(__file__).resolve().parent.parent))
from db.connection import get_engine
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
    """YOUR TURN.

    Return every (date, close) row from exchange_rates for this currency
    pair, oldest first - this feeds the historical chart (§19/§20).

    Steps (mirror get_prediction above):
    1. currency_pair = normalize_pair(pair)
    2. Open a connection: engine = get_engine(); with engine.connect() as conn:
    3. Run a SELECT for date, close FROM exchange_rates WHERE
       currency_pair = :p ORDER BY date (ascending, not DESC this time -
       a chart needs oldest-to-newest).
    4. conn.execute(...).mappings().all() gets you every matching row (vs
       .first() which only got one, above).
    5. Return a list of dicts shaped like HistoricalPoint (date, close) -
       FastAPI/Pydantic will validate + serialize the whole list because
       of response_model=list[HistoricalPoint].

    Optional stretch, once the basic version works: add a `limit: int |
    None = None` parameter to the function signature - FastAPI
    automatically turns that into an optional ?limit=90 query parameter,
    no extra code needed. Use it to return only the most recent N rows.
    """
    raise NotImplementedError("See the docstring above for the steps")


@app.get("/indicators/{pair}", response_model=list[IndicatorValue])
def get_indicators(pair: str):
    """YOUR TURN.

    Return the MOST RECENT value of each feature listed in
    INDICATOR_FEATURES (defined near the top of this file) - this feeds
    the "Economic Factors" panel (§19/§20).

    The tricky part: the features table has many rows per feature_name
    (one per date), and you only want the latest one for each. A
    subquery is the clean way to ask for that in SQL:

        SELECT feature_name, feature_value, date
        FROM features
        WHERE currency_pair = :p
          AND feature_name IN :names
          AND date = (SELECT MAX(date) FROM features WHERE currency_pair = :p)

    (":names" as a tuple/list works with SQLAlchemy's text() + IN, same
    style as :p elsewhere in this file.)

    Steps:
    1. currency_pair = normalize_pair(pair)
    2. Run that query (or your own equivalent) with
       conn.execute(...).mappings().all()
    3. For each row, look up INDICATOR_FEATURES[row["feature_name"]] to
       get its human-readable name
    4. Return a list of dicts shaped like IndicatorValue (name, value,
       as_of_date)
    """
    raise NotImplementedError("See the docstring above for the steps")


@app.get("/model-performance", response_model=list[ModelPerformanceEntry])
def get_model_performance():
    """YOUR TURN.

    Unlike the other routes, this one doesn't touch Postgres at all - it
    reads the JSON file ml/backtest.py writes at the end of its run
    (data/processed/model_performance.json), because re-running the full
    backtest (6 models x 5 years) on every API request would be far too
    slow for an endpoint that's supposed to respond instantly.

    Steps:
    1. import json, and MODEL_PERFORMANCE_PATH from ml.backtest (same
       import style as `from db.connection import get_engine` above)
    2. If the file doesn't exist yet (MODEL_PERFORMANCE_PATH.exists()),
       raise HTTPException(404, "...") with a message telling the caller
       to run ml/backtest.py first
    3. Otherwise, read and json.loads() the file, and return
       payload["models"] - already shaped to match ModelPerformanceEntry
    """
    raise NotImplementedError("See the docstring above for the steps")
