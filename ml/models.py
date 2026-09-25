"""Models that need more than the raw close (§12 Models 2-3).

Same interface as the naive models in backtest.py: fit(train_df) and
predict(test_df) -> predicted next-day close, so the walk-forward harness
scores them on identical folds.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Scale-free inputs only. The raw price-level features (prev_close, ma_7,
# ma_30, momentum_10) drift with the exchange rate itself, so a model trained
# on 2015 levels would see unfamiliar values in 2025. Ratios stay comparable
# across years.
FEATURE_COLUMNS = [
    "return_1d",
    "return_7d",
    "return_30d",
    "volatility_7",
    "volatility_30",
    "ma_gap_7",
    "ma_gap_30",
    "momentum_10_pct",
]


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ma_gap_7"] = df["close"] / df["ma_7"] - 1
    df["ma_gap_30"] = df["close"] / df["ma_30"] - 1
    df["momentum_10_pct"] = df["momentum_10"] / df["close"]
    return df


class MovingAverageModel:
    """Tomorrow's close = the 7-day moving average (§12 Model 2)."""

    name = "moving_avg_7"

    def fit(self, train: pd.DataFrame) -> None:
        pass

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        return test["ma_7"].to_numpy()


class LinearRegressionModel:
    """Predict next-day return from the features, then convert to a price (§12 Model 3).

    Predicting the return rather than the price is the approach §13 recommends:
    the target stays in a similar range across years.
    """

    name = "linear_regression"

    def __init__(self):
        self.pipeline = make_pipeline(StandardScaler(), LinearRegression())

    def fit(self, train: pd.DataFrame) -> None:
        train = train.dropna(subset=FEATURE_COLUMNS)
        target_return = train["next_close"] / train["close"] - 1
        self.pipeline.fit(train[FEATURE_COLUMNS], target_return)

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        predicted_return = self.pipeline.predict(test[FEATURE_COLUMNS])
        return test["close"].to_numpy() * (1 + predicted_return)
