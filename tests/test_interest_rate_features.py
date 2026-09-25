import pandas as pd
import pytest

from etl.interest_rate_features import forward_fill_to_daily


def make_events(dates, rates):
    return pd.DataFrame({"date": pd.to_datetime(dates), "rate": rates})


def test_each_day_gets_the_most_recent_rate_on_or_before_it():
    events = make_events(["2024-01-01", "2024-01-10"], [3.0, 3.25])
    days = pd.to_datetime(["2024-01-05", "2024-01-10", "2024-01-15"])

    result = forward_fill_to_daily(days, events)

    assert result.loc["2024-01-05"] == pytest.approx(3.0)   # before the 2nd decision: old rate
    assert result.loc["2024-01-10"] == pytest.approx(3.25)  # ON the decision day: new rate applies
    assert result.loc["2024-01-15"] == pytest.approx(3.25)  # after: still the new rate


def test_days_before_the_first_event_are_unknown_not_zero():
    # There's no rate on record before 2024-01-10, so it must come back
    # missing (NaN) - not a fabricated 0, which a model could easily mistake
    # for a real (very low) interest rate.
    events = make_events(["2024-01-10"], [3.25])
    days = pd.to_datetime(["2024-01-01", "2024-01-10"])

    result = forward_fill_to_daily(days, events)

    assert pd.isna(result.loc["2024-01-01"])
    assert result.loc["2024-01-10"] == pytest.approx(3.25)


def test_a_future_rate_change_never_affects_earlier_days():
    """The core no-leakage guarantee (section 10): a decision that hasn't
    happened yet must not change the feature value for a day before it.
    """
    days = pd.to_datetime(["2024-01-05", "2024-06-15", "2024-12-20"])

    events_before = make_events(["2024-01-01"], [3.0])
    result_before = forward_fill_to_daily(days, events_before)

    # Add a decision that happens AFTER all three test days.
    events_after = make_events(["2024-01-01", "2025-01-01"], [3.0, 5.0])
    result_after = forward_fill_to_daily(days, events_after)

    # Every day is before 2025-01-01, so nothing should have changed.
    pd.testing.assert_series_equal(result_before, result_after)
