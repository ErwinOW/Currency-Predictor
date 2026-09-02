"""Step 2: price-based feature engineering on top of exchange_rates.

Pipeline (per docs/project_rundown.md §11):
    exchange_rates -> price-based features -> features table

Critical rule from §10 (time alignment / no data leakage): every feature
below is computed using only rows on or before its own date — shift() and
rolling() only ever look backward, with no centering — so nothing here can
leak future information into a given day's feature set. That property
matters more than the specific feature list; keep it true for every
feature added later, from any data source.

Features (§11 "Price features" + §4.1 "Derived features"):
    prev_close    previous day's close
    return_1d     1-day return
    return_7d     7-day return
    return_30d    30-day return
    ma_7          7-day moving average of close
    ma_30         30-day moving average of close
    volatility_7  7-day rolling std of daily returns
    volatility_30 30-day rolling std of daily returns
    momentum_10   close minus close 10 days ago

Stored long-form (one row per date/feature) to match the `features` table
in db/schema.sql, which is shared across every currency pair and feature
source added later.

Usage:
    python etl/feature_engineering.py
"""
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import text

sys.path.append(str(Path(__file__).resolve().parent.parent))
from db.connection import get_engine

CURRENCY_PAIR = "MYR/IDR"


def load_exchange_rates(engine) -> pd.DataFrame:
    df = pd.read_sql(
        text("SELECT date, close FROM exchange_rates WHERE currency_pair = :pair ORDER BY date"),
        engine,
        params={"pair": CURRENCY_PAIR},
    )
    df["close"] = df["close"].astype(float)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    daily_return = close.pct_change()

    feats = pd.DataFrame({"date": df["date"]})
    feats["prev_close"] = close.shift(1)
    feats["return_1d"] = daily_return
    feats["return_7d"] = close.pct_change(7)
    feats["return_30d"] = close.pct_change(30)
    feats["ma_7"] = close.rolling(7).mean()
    feats["ma_30"] = close.rolling(30).mean()
    feats["volatility_7"] = daily_return.rolling(7).std()
    feats["volatility_30"] = daily_return.rolling(30).std()
    feats["momentum_10"] = close - close.shift(10)
    return feats


def to_long(feats: pd.DataFrame) -> pd.DataFrame:
    """Wide (one column per feature) -> long (one row per date/feature).

    Rows are dropped where the feature isn't defined yet (e.g. ma_30 has
    no value for the first 29 days) rather than stored as NULL — keeps the
    features table free of placeholder rows.
    """
    long_df = feats.melt(id_vars="date", var_name="feature_name", value_name="feature_value")
    long_df = long_df.dropna(subset=["feature_value"])
    return long_df


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
            "date": row.date,
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

    df = load_exchange_rates(engine)
    print(f"Loaded {len(df)} exchange-rate rows for {CURRENCY_PAIR}")
    if df.empty:
        print("No exchange-rate data found — run ingestion/fetch_exchange_rates.py first.")
        return

    feats = build_features(df)
    long_df = to_long(feats)
    n_feature_types = feats.shape[1] - 1  # exclude the date column
    print(f"Computed {len(long_df)} feature values across {n_feature_types} feature types")

    n = load_features(engine, long_df)
    print(f"Upserted {n} row(s) into features")


if __name__ == "__main__":
    main()
