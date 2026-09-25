"""Step: turn interest-rate EVENTS into a DAILY feature, without leaking the future (§10).

The problem this file solves:

    interest_rates table:  a handful of rows per year, one per real-world
                            event (an OPR decision, a FRED monthly reading).
    features table:        needs ONE VALUE FOR EVERY TRADING DAY, because
                            that's what exchange_rates has one row for.

The naive-looking but WRONG way to bridge that gap is to loop over every
day and ask "what was the most recent rate on or before this day?" one day
at a time - that's actually the right *idea*, just slow to write by hand
(a Python loop over ~3000 days) and easy to get subtly wrong (off-by-one
on the boundary day, or accidentally comparing "before" instead of
"on-or-before").

pandas has a built-in tool for exactly this: pd.merge_asof(). Read it as
"merge, but instead of matching on equal keys like a normal join, match
each row to the NEAREST earlier key". That's it - one call replaces the
day-by-day loop.

THE LEAKAGE RULE, made concrete: for a row dated 2024-06-15, merge_asof
with direction="backward" is only allowed to look at interest-rate events
dated 2024-06-15 or earlier. It physically cannot see a decision announced
on 2024-06-20, because that event's row lives further down a sorted list
that backward-matching never looks into. tests/test_interest_rate_features.py
checks this directly, the same way the price-feature leakage test does.

Usage:
    python etl/interest_rate_features.py
"""
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import text

sys.path.append(str(Path(__file__).resolve().parent.parent))
from db.connection import get_engine

CURRENCY_PAIR = "MYR/IDR"
COUNTRIES = {"Malaysia": "interest_rate_malaysia", "Indonesia": "interest_rate_indonesia"}


def forward_fill_to_daily(daily_dates: pd.Series, events: pd.DataFrame) -> pd.Series:
    """Give every date in `daily_dates` the most recent known value from `events`.

    daily_dates: every day we need a value for (sorted ascending).
    events:      columns "date" and "rate", one row per real event (sorted
                 ascending). NOT one row per calendar day.

    A day before the first event has nothing to forward-fill from - there
    was no OPR/FRED reading yet - so it correctly comes back as NaN rather
    than a guessed value.
    """
    daily_dates = pd.Series(pd.to_datetime(daily_dates)).sort_values().reset_index(drop=True)
    daily_df = pd.DataFrame({"date": daily_dates})
    events_sorted = events.sort_values("date").reset_index(drop=True)
    events_sorted["date"] = pd.to_datetime(events_sorted["date"])

    merged = pd.merge_asof(daily_df, events_sorted, on="date", direction="backward")
    return merged.set_index(daily_df["date"])["rate"]


def load_daily_dates(engine) -> pd.Series:
    """The trading-day calendar we need features for - every day exchange_rates has."""
    df = pd.read_sql(
        text("SELECT DISTINCT date FROM exchange_rates WHERE currency_pair = :pair ORDER BY date"),
        engine,
        params={"pair": CURRENCY_PAIR},
    )
    return df["date"]


def load_events(engine, country: str) -> pd.DataFrame:
    df = pd.read_sql(
        text("SELECT date, rate FROM interest_rates WHERE country = :country ORDER BY date"),
        engine,
        params={"country": country},
    )
    df["rate"] = df["rate"].astype(float)
    return df


def build_features(engine) -> pd.DataFrame:
    daily_dates = load_daily_dates(engine)

    aligned = {}
    for country, feature_name in COUNTRIES.items():
        events = load_events(engine, country)
        aligned[feature_name] = forward_fill_to_daily(daily_dates, events)

    wide = pd.DataFrame(aligned)
    wide["interest_rate_differential"] = (
        wide["interest_rate_malaysia"] - wide["interest_rate_indonesia"]
    )
    wide = wide.reset_index().rename(columns={"index": "date"})
    return wide


def to_long(wide: pd.DataFrame) -> pd.DataFrame:
    long_df = wide.melt(id_vars="date", var_name="feature_name", value_name="feature_value")
    return long_df.dropna(subset=["feature_value"])


def load_features(engine, long_df: pd.DataFrame) -> int:
    if long_df.empty:
        return 0
    upsert_sql = text(
        """
        INSERT INTO features (date, currency_pair, feature_name, feature_value)
        VALUES (:date, :currency_pair, :feature_name, :feature_value)
        ON CONFLICT (date, currency_pair, feature_name)
        DO UPDATE SET feature_value = EXCLUDED.feature_value
        """
    )
    records = [
        {
            "date": row.date.date(),
            "currency_pair": CURRENCY_PAIR,
            "feature_name": row.feature_name,
            "feature_value": float(row.feature_value),
        }
        for row in long_df.itertuples()
    ]
    with engine.begin() as conn:
        conn.execute(upsert_sql, records)
    return len(records)


def main():
    engine = get_engine()
    wide = build_features(engine)
    print(f"Aligned interest-rate features for {len(wide)} trading days")

    long_df = to_long(wide)
    print(f"  {len(long_df)} feature values have a real value (rest are before the first known rate)")

    n = load_features(engine, long_df)
    print(f"Upserted {n} row(s) into features")


if __name__ == "__main__":
    main()
