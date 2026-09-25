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
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

sys.path.append(str(Path(__file__).resolve().parent.parent))
from db.connection import get_engine
from ml.models import LinearRegressionModel, MovingAverageModel, RandomForestModel, add_derived_features

CURRENCY_PAIR = "MYR/IDR"
FIRST_TEST_YEAR = 2022


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


def load_data(engine) -> pd.DataFrame:
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
    df = add_derived_features(df)

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


if __name__ == "__main__":
    main()
