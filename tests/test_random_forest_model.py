"""Tests for RandomForestModel (ml/models.py).

Run with: pytest tests/test_random_forest_model.py -v

These will fail with NotImplementedError until you fill in __init__, fit,
and predict. Once implemented correctly, all three should pass.
"""
import numpy as np
import pandas as pd
import pytest

from ml.models import RF_FEATURE_COLUMNS, RandomForestModel


def make_synthetic_data(n=200, seed=0):
    """Build fake-but-structured data so we can check the model actually
    learns something, not just that it runs without crashing.

    next_close is built so that return_1d alone predicts it almost exactly
    (a small, clean signal) - every other feature is random noise. A model
    that's wired up correctly should lean on return_1d and predict well;
    a broken one (e.g. wrong column used, target computed wrong) won't.
    """
    rng = np.random.default_rng(seed)
    close = 4000 + rng.normal(0, 50, size=n).cumsum() * 0.01 + 4000
    df = pd.DataFrame({"close": close})

    for col in RF_FEATURE_COLUMNS:
        df[col] = rng.normal(0, 0.01, size=n)

    true_next_return = df["return_1d"] * 0.5  # the "signal" the model should find
    df["next_close"] = df["close"] * (1 + true_next_return)
    return df


def test_fits_without_error():
    train = make_synthetic_data(n=200, seed=0)
    model = RandomForestModel()
    model.fit(train)  # should not raise


def test_predictions_are_close_to_the_true_signal():
    train = make_synthetic_data(n=200, seed=0)
    test = make_synthetic_data(n=50, seed=1)

    model = RandomForestModel()
    model.fit(train)
    predictions = model.predict(test)

    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == len(test)

    # Predictions should correlate strongly with the true next_close, since
    # the synthetic data has a real (if small) signal to find.
    correlation = np.corrcoef(predictions, test["next_close"])[0, 1]
    assert correlation > 0.5, f"expected predictions to track the signal, got correlation={correlation:.3f}"


def test_feature_importances_has_one_value_per_feature():
    train = make_synthetic_data(n=200, seed=0)
    model = RandomForestModel()
    model.fit(train)

    importances = model.model.feature_importances_
    assert len(importances) == len(RF_FEATURE_COLUMNS)
    assert importances.sum() == pytest.approx(1.0)


def test_return_1d_is_the_most_important_feature():
    # Since return_1d is the only feature with real signal in our synthetic
    # data, a correctly-trained forest should rank it highest.
    train = make_synthetic_data(n=200, seed=0)
    model = RandomForestModel()
    model.fit(train)

    importances = dict(zip(RF_FEATURE_COLUMNS, model.model.feature_importances_))
    most_important = max(importances, key=importances.get)
    assert most_important == "return_1d", f"expected return_1d to dominate, importances={importances}"
