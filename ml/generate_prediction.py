"""Generate today's actual MYR/IDR prediction and store it (§12 Model 5, §14).

This is different from ml/backtest.py, which asks "how would this model
have performed historically?" by holding out past years it already knows
the answer to. This script asks "what does the model say about tomorrow,
right now?" - there's no held-out year here, because there's nothing to
hold out: tomorrow hasn't happened. So the model is retrained on ALL
available history and asked to predict the one row that has no answer
yet (today's features, predicting a next_close that doesn't exist in the
database).

THE CONFIDENCE PROBLEM (§14): "the confidence value should be based on an
actual statistical/model methodology rather than being an arbitrary
number." A model doesn't hand you a trustworthy confidence score for
free - scikit-learn/XGBoost's regression output is just a number, with no
built-in sense of "and I'm 73% sure." Two honest, backtest-derived
numbers stand in for it here, deliberately NOT invented:

  1. Prediction interval (lower_bound/upper_bound): built from the
     model's own pooled walk-forward errors (ml/backtest.py's
     collect_residuals) - the 5th and 95th percentile of every mistake
     XGBoost actually made on 5 years of real out-of-sample data, added
     to today's point prediction. This is an EMPIRICAL interval, not a
     textbook one assuming errors are bell-curve-shaped (they might not
     be - this sidesteps that assumption entirely by using the real
     distribution of past errors).

  2. Confidence (direction): the model's own historical directional
     accuracy from the same backtest (~52% for XGBoost) - i.e. "in 5
     years of walk-forward testing, this model called the direction
     correctly 52% of the time." That's an honest, low number, and it's
     reported as one - not dressed up. See docs/project_rundown.md §29:
     this is an experimental system, not a guaranteed trading signal.

Usage:
    python ml/generate_prediction.py
"""
import sys
from pathlib import Path

import numpy as np
from sqlalchemy import text

sys.path.append(str(Path(__file__).resolve().parent.parent))
from db.connection import get_engine
from ml.backtest import CURRENCY_PAIR, collect_residuals, load_data, load_raw_features, walk_forward
from ml.models import XGBoostModel

MODEL_VERSION = "xgboost_v1"
HORIZON = "1d"
LOWER_PERCENTILE = 5
UPPER_PERCENTILE = 95


def generate() -> dict:
    engine = get_engine()

    raw = load_raw_features(engine)
    today = raw.iloc[[-1]]  # the one row with no next_close yet - see module docstring
    current_close = float(today["close"].iloc[0])

    # Train on every row that DOES have a known outcome (i.e. everything
    # except the "today" row itself) - the most complete model we can
    # honestly build right now.
    train_df = load_data(engine)
    model = XGBoostModel()
    model.fit(train_df)

    predicted_close = float(model.predict(today)[0])
    predicted_return = predicted_close / current_close - 1

    # Empirical prediction interval from the model's actual walk-forward
    # track record (see module docstring, point 1).
    residuals = collect_residuals(train_df, XGBoostModel())
    lower_bound = predicted_close + float(np.percentile(residuals, LOWER_PERCENTILE))
    upper_bound = predicted_close + float(np.percentile(residuals, UPPER_PERCENTILE))

    # Historical directional accuracy from the same backtest, as the
    # confidence value (see module docstring, point 2).
    backtest_results = walk_forward(train_df, XGBoostModel())
    weights = backtest_results["n"]
    confidence = float(np.average(backtest_results["directional_accuracy"], weights=weights))

    direction = "Bullish" if predicted_return > 0 else "Bearish" if predicted_return < 0 else "Neutral"

    return {
        "currency_pair": CURRENCY_PAIR,
        "horizon": HORIZON,
        "predicted_rate": predicted_close,
        "lower_bound": min(lower_bound, upper_bound),
        "upper_bound": max(lower_bound, upper_bound),
        "direction": direction,
        "confidence": confidence,
        "model_version": MODEL_VERSION,
        "current_rate": current_close,
        "expected_movement_pct": predicted_return * 100,
        "as_of_date": today["date"].iloc[0].date().isoformat(),
    }


def save(engine, prediction: dict) -> None:
    insert_sql = text(
        """
        INSERT INTO predictions
            (currency_pair, horizon, predicted_rate, lower_bound, upper_bound,
             direction, confidence, model_version)
        VALUES
            (:currency_pair, :horizon, :predicted_rate, :lower_bound, :upper_bound,
             :direction, :confidence, :model_version)
        """
    )
    with engine.begin() as conn:
        conn.execute(insert_sql, {k: v for k, v in prediction.items() if k in {
            "currency_pair", "horizon", "predicted_rate", "lower_bound", "upper_bound",
            "direction", "confidence", "model_version",
        }})


def main():
    prediction = generate()

    print(f"As of {prediction['as_of_date']}:")
    print(f"  Current Rate:      {prediction['current_rate']:.2f}")
    print(f"  Predicted Rate:    {prediction['predicted_rate']:.2f}")
    print(f"  Expected Movement: {prediction['expected_movement_pct']:+.2f}%")
    print(f"  Direction:         {prediction['direction']}")
    print(f"  Expected Range:    {prediction['lower_bound']:.2f} - {prediction['upper_bound']:.2f}")
    print(f"  Confidence:        {prediction['confidence']:.0%} (historical directional accuracy)")

    save(get_engine(), prediction)
    print("\nSaved to predictions table.")


if __name__ == "__main__":
    main()
