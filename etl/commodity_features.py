"""Turn daily-but-not-perfectly-aligned commodity prices into features (§11).

Unlike interest rates (a handful of rows/year), Brent/WTI/palm oil are
already daily series - most days will already line up with an exchange-
rate trading day. Still routed through the same forward_fill_to_daily
(etl/time_alignment.py) as interest rates, for one reason: the exact set
of market holidays differs between the MYR/IDR FX market and the CME/ICE
commodity markets, so a handful of days genuinely won't have a same-day
commodity print. merge_asof's backward-fill handles that mismatch the same
safe way, at no extra cost on days that DO line up (it just finds an exact
match).

For each commodity, two features:
    {name}_price      the aligned price level itself
    {name}_return_1d  day-over-day % change of the ALIGNED series

The return is computed after alignment (not on the raw commodity rows) so
that on the rare day a market was closed, the "return" correctly comes out
as 0% (nothing changed) rather than comparing across a gap.

Usage:
    python etl/commodity_features.py
"""
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import text

sys.path.append(str(Path(__file__).resolve().parent.parent))
from db.connection import get_engine
from etl.features_common import load_daily_dates, to_long, upsert_features
from etl.time_alignment import forward_fill_to_daily

COMMODITIES = {
    "Brent Crude": "brent",
    "WTI Crude": "wti",
    "Palm Oil": "palm_oil",
}


def load_events(engine, commodity: str) -> pd.DataFrame:
    df = pd.read_sql(
        text("SELECT date, price FROM commodities WHERE commodity = :commodity ORDER BY date"),
        engine,
        params={"commodity": commodity},
    )
    df["price"] = df["price"].astype(float)
    return df


def build_features(engine) -> pd.DataFrame:
    daily_dates = load_daily_dates(engine)

    wide = pd.DataFrame({"date": pd.to_datetime(daily_dates).sort_values().reset_index(drop=True)})
    for commodity, prefix in COMMODITIES.items():
        events = load_events(engine, commodity)
        aligned_price = forward_fill_to_daily(daily_dates, events, value_col="price")
        wide[f"{prefix}_price"] = aligned_price.to_numpy()
        wide[f"{prefix}_return_1d"] = aligned_price.pct_change().to_numpy()

    return wide


def main():
    engine = get_engine()
    wide = build_features(engine)
    print(f"Aligned commodity features for {len(wide)} trading days")

    long_df = to_long(wide)
    print(f"  {len(long_df)} feature values have a real value")

    n = upsert_features(engine, long_df)
    print(f"Upserted {n} row(s) into features")


if __name__ == "__main__":
    main()
