"""Models that need more than the raw close (§12 Models 2-3).

Same interface as the naive models in backtest.py: fit(train_df) and
predict(test_df) -> predicted next-day close, so the walk-forward harness
scores them on identical folds.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
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

# Random forests split on raw thresholds rather than fitting a scaled linear
# combination, so they don't care about a feature's scale or units the way
# linear regression does - the interest-rate features (in percentage points,
# e.g. 3.25) can sit alongside the price ratios (e.g. 0.007) with no scaling
# step needed. That's why this list can safely be FEATURE_COLUMNS "as is,
# plus more", while LinearRegressionModel keeps its own smaller list.
RF_FEATURE_COLUMNS = FEATURE_COLUMNS + [
    "interest_rate_malaysia",
    "interest_rate_indonesia",
    "interest_rate_differential",
    # Only the returns, not the raw brent_price/wti_price/palm_oil_price
    # levels - same reason as ma_gap_7/30 above: a raw price level drifts
    # over a decade of data (oil at $40 in 2015 vs $95 in 2026), while a
    # day-over-day return stays in a comparable range across all years.
    "brent_return_1d",
    "wti_return_1d",
    "palm_oil_return_1d",
    # sentiment_score (GDELT average tone) is already a bounded, roughly
    # stationary quantity - not article_count, which could structurally
    # grow over the years as global news coverage volume grows, the same
    # "raw level drifts over time" problem as a raw price.
    "sentiment_score",
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


class RandomForestModel:
    """Predict next-day return from the features using a random forest (§12 Model 4).

    max_depth=5 and min_samples_leaf=10 are deliberately conservative -
    an unconstrained tree can memorize individual rows in ~2,900 samples
    of noisy daily FX data. Unlike LinearRegressionModel, no scaler is
    needed: trees split on raw thresholds, so feature scale doesn't matter.
    """

    name = "random_forest"

    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=150, max_depth=5, min_samples_leaf=10, random_state=42)

    def fit(self, train: pd.DataFrame) -> None:
        train = train.dropna(subset=RF_FEATURE_COLUMNS)
        target_return = train["next_close"] / train["close"] - 1
        self.model.fit(train[RF_FEATURE_COLUMNS], target_return)

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        predicted_return = self.model.predict(test[RF_FEATURE_COLUMNS])
        return test["close"].to_numpy() * (1 + predicted_return)


class XGBoostModel:
    """Predict next-day return using gradient boosting (§12 Model 5).

    Unlike random forest's independent trees averaged together, boosting
    builds trees sequentially, each correcting the previous trees'
    errors - so max_depth=1 (a single split per tree, a "stump") is
    conservative on purpose: with ~2,900 noisy samples, deeper trees here
    would compound overfitting across rounds rather than average it out.
    """

    name = "xgboost"

    def __init__(self):
        self.model = xgb.XGBRegressor(n_estimators=150, max_depth=1, random_state=42, learning_rate=0.1)

    def fit(self, train: pd.DataFrame) -> None:
        train = train.dropna(subset=RF_FEATURE_COLUMNS)
        target_return = train["next_close"] / train["close"] - 1
        self.model.fit(train[RF_FEATURE_COLUMNS], target_return)

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        predicted_return = self.model.predict(test[RF_FEATURE_COLUMNS])
        return test["close"].to_numpy() * (1 + predicted_return)
