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

    YOUR TURN. This class follows the exact same shape as LinearRegressionModel
    above - same target (next-day return, not raw price, for the reason explained
    there), same three methods. Use it as your reference.

    What to fill in:

    1. __init__: create self.model = RandomForestRegressor(...).
       Docs: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
       Start by choosing values for:
         - n_estimators    how many trees (100-300 is a reasonable range to try)
         - max_depth       how many questions deep a tree can go. Small values
                            (e.g. 3-6) fight overfitting; None means "no limit",
                            which on ~2,900 noisy rows risks memorizing noise -
                            see the explanation in the chat before you pick this.
         - min_samples_leaf a leaf must have at least this many rows. Higher =
                            more conservative, similar effect to max_depth.
         - random_state=42  fixes the randomness so re-running gives the same
                            result - important for comparing settings fairly.
       Unlike LinearRegressionModel, there's no StandardScaler/pipeline needed -
       trees split on raw thresholds, so feature scale doesn't matter to them.

    2. fit(self, train): same 3 lines as LinearRegressionModel.fit, but:
         - use RF_FEATURE_COLUMNS instead of FEATURE_COLUMNS
         - call self.model.fit(...) instead of self.pipeline.fit(...)

    3. predict(self, test): same idea as LinearRegressionModel.predict, again
       swapping in RF_FEATURE_COLUMNS and self.model.

    Once it runs, look at self.model.feature_importances_ (one number per
    column in RF_FEATURE_COLUMNS, summing to 1) - it tells you which features
    the forest actually leaned on. Worth printing after a fit() call to see
    whether the interest-rate features earned their place.
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

    YOUR TURN, same three methods as RandomForestModel above - use it as
    your reference, and reuse RF_FEATURE_COLUMNS (same feature set, no new
    list needed - trees don't care about feature scale here either).

    The one real difference from random forest: boosting builds trees
    SEQUENTIALLY, each one correcting the previous trees' mistakes, instead
    of averaging independent trees. That makes one new hyperparameter
    relevant that random forest doesn't have:

        learning_rate   how much each new tree is allowed to correct.
                        Small (e.g. 0.05) = each tree nudges gently, needs
                        more rounds, generally overfits less. Large (e.g.
                        0.3) = corrects fast, more prone to chasing noise.
                        Try 0.05-0.1 as a starting point.

    What to fill in:

    1. __init__: create self.model = xgb.XGBRegressor(...).
       Docs: https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBRegressor
       Same parameters to choose as RandomForestModel (n_estimators,
       max_depth, random_state) PLUS learning_rate. One tuning note:
       because boosting compounds each tree's mistakes into the next,
       max_depth is usually set SHALLOWER here than for a random forest
       (e.g. 2-4 rather than 5) - a good thing to try both ways and
       compare once it's running.

    2. fit(self, train): identical shape to RandomForestModel.fit.

    3. predict(self, test): identical shape to RandomForestModel.predict.

    Compare self.model.feature_importances_ against RandomForestModel's -
    worth checking whether boosting leans on the same features or finds
    different ones useful.
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
