"""Shared tool for turning sparse/event-based data into a daily feature (§10).

Originally written for interest rates (a handful of rows/year) but the same
problem shows up for any data source that doesn't publish one row per
calendar day - moved here once a second caller (commodity prices) needed
the identical logic, rather than copy-pasting it.

See etl/interest_rate_features.py's git history for the original writeup
of *why* pd.merge_asof(direction="backward") is what prevents this from
leaking future information backward - the short version: it matches each
day only to events on or before it, never after.
"""
import pandas as pd


def forward_fill_to_daily(daily_dates: pd.Series, events: pd.DataFrame, value_col: str) -> pd.Series:
    """Give every date in `daily_dates` the most recent known value from `events`.

    daily_dates: every day we need a value for (sorted ascending).
    events:      columns "date" and `value_col`, one row per real event
                 (sorted ascending) - NOT one row per calendar day.
    value_col:   name of the value column in `events` (e.g. "rate", "price").

    A day before the first event has nothing to forward-fill from, so it
    correctly comes back as NaN rather than a guessed value.
    """
    daily_dates = pd.Series(pd.to_datetime(daily_dates)).sort_values().reset_index(drop=True)
    daily_df = pd.DataFrame({"date": daily_dates})
    events_sorted = events.sort_values("date").reset_index(drop=True)
    events_sorted["date"] = pd.to_datetime(events_sorted["date"])

    merged = pd.merge_asof(daily_df, events_sorted, on="date", direction="backward")
    return merged.set_index(daily_df["date"])[value_col]
