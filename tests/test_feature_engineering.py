import pandas as pd
import pytest

from etl.feature_engineering import build_features


def make_prices(closes):
    """Build the small DataFrame that build_features expects: date + close."""
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=len(closes), freq="D"),
        "close": closes,
    })


def test_return_1d_matches_hand_calculation():
    # 100 -> 110 is a 10% rise; 110 -> 99 is a 10% fall.
    feats = build_features(make_prices([100.0, 110.0, 99.0]))

    assert feats["return_1d"].iloc[1] == pytest.approx(0.10)
    assert feats["return_1d"].iloc[2] == pytest.approx(-0.10)


def test_first_row_has_no_return_or_prev_close():
    # There is no "day before" the first row, so these must be missing (NaN),
    # not a made-up number.
    feats = build_features(make_prices([100.0, 101.0, 102.0]))

    assert pd.isna(feats["return_1d"].iloc[0])
    assert pd.isna(feats["prev_close"].iloc[0])


def test_ma_7_needs_seven_days_of_history():
    feats = build_features(make_prices([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]))

    assert feats["ma_7"].iloc[:6].isna().all()          # rows 0-5: not enough history yet
    assert feats["ma_7"].iloc[6] == pytest.approx(4.0)  # mean of 1..7
    assert feats["ma_7"].iloc[7] == pytest.approx(5.0)  # mean of 2..8


def test_features_do_not_leak_the_future():
    """Changing tomorrow's price must not change any feature computed for today.

    This is the "no data leakage" rule from the spec (section 10) turned into a
    test: if a feature secretly looked forward, this would fail.
    """
    closes = [float(x) for x in range(100, 140)]
    original = build_features(make_prices(closes))

    changed = closes.copy()
    changed[-1] = 9999.0  # rewrite only the LAST day
    altered = build_features(make_prices(changed))

    # Every row except the last must be identical.
    pd.testing.assert_frame_equal(original.iloc[:-1], altered.iloc[:-1])
