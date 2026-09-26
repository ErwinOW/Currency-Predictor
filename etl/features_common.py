"""Shared load/save helpers for anything that writes into the `features` table.

Pulled out once interest_rate_features.py and commodity_features.py both
needed the identical "load the trading-day calendar" / "wide -> long" /
"upsert into features" logic.
"""
import pandas as pd
from sqlalchemy import text

CURRENCY_PAIR = "MYR/IDR"


def load_daily_dates(engine) -> pd.Series:
    """The trading-day calendar we need features for - every day exchange_rates has."""
    df = pd.read_sql(
        text("SELECT DISTINCT date FROM exchange_rates WHERE currency_pair = :pair ORDER BY date"),
        engine,
        params={"pair": CURRENCY_PAIR},
    )
    return df["date"]


def to_long(wide: pd.DataFrame) -> pd.DataFrame:
    """Wide (one column per feature) -> long (one row per date/feature).

    Rows are dropped where a feature has no value yet, rather than stored
    as NULL, to keep the features table free of placeholder rows.
    """
    long_df = wide.melt(id_vars="date", var_name="feature_name", value_name="feature_value")
    return long_df.dropna(subset=["feature_value"])


def upsert_features(engine, long_df: pd.DataFrame) -> int:
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
            "date": row.date.date() if hasattr(row.date, "date") else row.date,
            "currency_pair": CURRENCY_PAIR,
            "feature_name": row.feature_name,
            "feature_value": float(row.feature_value),
        }
        for row in long_df.itertuples()
    ]
    with engine.begin() as conn:
        conn.execute(upsert_sql, records)
    return len(records)
