import numpy as np
import pandas as pd
import pytest

from ml.backtest import score


def make_test_frame(closes, next_closes):
    return pd.DataFrame({"close": closes, "next_close": next_closes})


def test_perfect_predictions_score_perfectly():
    test = make_test_frame([100.0, 101.0, 100.0], [101.0, 100.0, 102.0])
    perfect = test["next_close"].to_numpy()

    result = score(test, perfect)

    assert result["mae"] == 0.0
    assert result["rmse"] == 0.0
    assert result["directional_accuracy"] == 1.0


def test_mae_and_rmse_match_hand_calculation():
    # Errors (prediction - actual) are -1 and +4. MAE = (1 + 4) / 2 = 2.5.
    # RMSE = sqrt((1 + 16) / 2) = sqrt(8.5).
    # Both days have a real move (103 is up from 100, 92 is down from 100) so
    # directional_accuracy is defined too and this test raises no warning.
    test = make_test_frame([100.0, 100.0], [103.0, 92.0])
    predictions = np.array([102.0, 96.0])

    result = score(test, predictions)

    assert result["mae"] == pytest.approx(2.5)
    assert result["rmse"] == pytest.approx(np.sqrt(8.5))


def test_predicting_no_change_has_no_directional_accuracy():
    # A model that always says "no change" never calls a direction, so the
    # metric is undefined (NaN) rather than a misleading 0%.
    test = make_test_frame([100.0, 100.0], [101.0, 99.0])
    predictions = test["close"].to_numpy()

    result = score(test, predictions)

    assert np.isnan(result["directional_accuracy"])
