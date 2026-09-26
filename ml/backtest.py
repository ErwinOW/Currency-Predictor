"""Step 3: walk-forward backtesting harness + naive baseline (§12 Model 1, §15, §16).

Task: predict the next trading day's MYR/IDR close.

Walk-forward (§15): for each test year Y, train on all data before Y and
evaluate on year Y — never a random split. Models expose fit(train_df) and
predict(test_df) -> predicted next-day close, so later models (linear,
random forest, XGBoost) plug into the same harness and are scored on
identical folds.

Each df has columns: date, close, next_close (the target). Row t's
next_close is the close at the next available date, so predict() may only
use information up to and including row t.

Usage:
    python ml/backtest.py
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

sys.path.append(str(Path(__file__).resolve().parent.parent))
from db.connection import get_engine
from ml.models import (
    LinearRegressionModel,
    MovingAverageModel,
    RandomForestModel,
    XGBoostModel,
    add_derived_features,
)

CURRENCY_PAIR = "MYR/IDR"
FIRST_TEST_YEAR = 2022
MODEL_PERFORMANCE_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "model_performance.json"


class NaiveModel:
    """Tomorrow's close = today's close (§12 Model 1)."""

    name = "naive"

    def fit(self, train: pd.DataFrame) -> None:
        pass

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        return test["close"].to_numpy()


class NaiveMomentumModel:
    """Tomorrow moves in the same direction as today's move, by the same size.

    Included because the pure naive model predicts zero change, which has no
    direction and so can't be scored on directional accuracy.
    """

    name = "naive_momentum"

    def fit(self, train: pd.DataFrame) -> None:
        pass

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        return (test["close"] + (test["close"] - test["prev_close"])).to_numpy()


def load_raw_features(engine) -> pd.DataFrame:
    """Exchange rates + every engineered feature, joined, for every date on record.

    Deliberately keeps the most recent row even though it has no next_close
    yet (nothing has happened "tomorrow" for the last row in the table) -
    that row is exactly what ml/generate_prediction.py needs: today's
    features, to predict a next_close that doesn't exist yet. load_data()
    below is the backtesting version, which correctly drops that row
    instead (you can't score a prediction you have no real answer for).
    """
    df = pd.read_sql(
        text("SELECT date, close FROM exchange_rates WHERE currency_pair = :pair ORDER BY date"),
        engine,
        params={"pair": CURRENCY_PAIR},
    )
    df["close"] = df["close"].astype(float)
    df["date"] = pd.to_datetime(df["date"])
    df["prev_close"] = df["close"].shift(1)
    df["next_close"] = df["close"].shift(-1)
    df["next_date"] = df["date"].shift(-1)

    # Attach the engineered features from the features table (one column each).
    feats = pd.read_sql(
        text("SELECT date, feature_name, feature_value FROM features WHERE currency_pair = :pair"),
        engine,
        params={"pair": CURRENCY_PAIR},
    )
    feats["date"] = pd.to_datetime(feats["date"])
    feats["feature_value"] = feats["feature_value"].astype(float)
    wide = feats.pivot(index="date", columns="feature_name", values="feature_value").reset_index()
    # prev_close already exists above; keep the one computed here, drop the duplicate.
    df = df.merge(wide.drop(columns=["prev_close"]), on="date", how="left")
    return add_derived_features(df)


def load_data(engine) -> pd.DataFrame:
    df = load_raw_features(engine)
    # Last row has no next_close (nothing to predict yet); first has no prev_close.
    return df.dropna(subset=["prev_close", "next_close"]).reset_index(drop=True)


def score(test: pd.DataFrame, pred: np.ndarray) -> dict:
    err = pred - test["next_close"].to_numpy()
    actual_move = np.sign(test["next_close"].to_numpy() - test["close"].to_numpy())
    pred_move = np.sign(pred - test["close"].to_numpy())
    # Days with no actual move can't be called up or down; exclude them.
    moved = actual_move != 0
    dir_acc = float((pred_move[moved] == actual_move[moved]).mean())
    # A model that predicts no change never gets a direction right.
    predicts_direction = bool((pred_move != 0).any())
    return {
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err**2).mean())),
        "directional_accuracy": dir_acc if predicts_direction else float("nan"),
        "n": int(len(test)),
    }


def collect_residuals(df: pd.DataFrame, model, first_test_year: int = FIRST_TEST_YEAR) -> np.ndarray:
    """Every out-of-sample (predicted - actual) error the model made across
    all walk-forward test years, pooled into one array.

    Used by ml/generate_prediction.py to build a prediction interval from
    the model's actual historical track record, instead of an assumed
    (and likely wrong) bell-curve shape or an arbitrary +/- number - see
    that script's docstring for §14's "not an arbitrary number" rule.
    """
    errors = []
    for year in range(first_test_year, int(df["date"].dt.year.max()) + 1):
        test = df[df["date"].dt.year == year]
        train = df[df["next_date"] < test["date"].min()]
        if train.empty or test.empty:
            continue
        model.fit(train)
        errors.append(model.predict(test) - test["next_close"].to_numpy())
    return np.concatenate(errors)


def walk_forward(df: pd.DataFrame, model, first_test_year: int = FIRST_TEST_YEAR) -> pd.DataFrame:
    rows = []
    for year in range(first_test_year, int(df["date"].dt.year.max()) + 1):
        test = df[df["date"].dt.year == year]
        # Train only on rows whose target (next_close) is known before the test
        # year starts. Without this, the last trading day of the prior year has
        # its target on the first day of the test year, which leaks into training.
        train = df[df["next_date"] < test["date"].min()]
        if train.empty or test.empty:
            continue
        model.fit(train)
        rows.append({"model": model.name, "test_year": year, **score(test, model.predict(test))})
    return pd.DataFrame(rows)


def main():
    df = load_data(get_engine())
    print(f"Loaded {len(df)} usable rows, {df['date'].min().date()} -> {df['date'].max().date()}")

    results = pd.concat(
        [
            walk_forward(df, m)
            for m in (
                NaiveModel(),
                NaiveMomentumModel(),
                MovingAverageModel(),
                LinearRegressionModel(),
                RandomForestModel(),
                XGBoostModel(),
            )
        ],
        ignore_index=True,
    )
    pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
    print("\nPer-year results (walk-forward, train on all prior years):")
    print(results.to_string(index=False))

    # Overall: weight each fold by its number of test days.
    def overall(g):
        w = g["n"]
        return pd.Series({
            "mae": np.average(g["mae"], weights=w),
            "rmse": np.sqrt(np.average(g["rmse"] ** 2, weights=w)),
            "directional_accuracy": np.average(g["directional_accuracy"].fillna(0), weights=w)
            if g["directional_accuracy"].notna().any() else float("nan"),
        })

    summary = results.groupby("model")[["mae", "rmse", "directional_accuracy", "n"]].apply(overall)
    print("\nOverall:")
    print(summary.to_string())

    save_model_performance(summary)


def save_model_performance(summary: pd.DataFrame) -> None:
    """Cache the overall results so the API can serve them without re-running
    the entire backtest (6 models x 5 years) on every request - it just
    reads this file. Re-run ml/backtest.py to refresh it.
    """
    MODEL_PERFORMANCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "currency_pair": CURRENCY_PAIR,
        "models": [
            {
                "model": model,
                "mae": None if pd.isna(row.mae) else round(float(row.mae), 4),
                "rmse": None if pd.isna(row.rmse) else round(float(row.rmse), 4),
                "directional_accuracy": None if pd.isna(row.directional_accuracy) else round(float(row.directional_accuracy), 4),
            }
            for model, row in summary.iterrows()
        ],
    }
    MODEL_PERFORMANCE_PATH.write_text(json.dumps(payload, indent=2))
    print(f"\nCached model performance -> {MODEL_PERFORMANCE_PATH}")


if __name__ == "__main__":
    main()
