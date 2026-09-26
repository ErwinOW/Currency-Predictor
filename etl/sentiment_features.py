"""Turn weekly news-sentiment samples into daily features (§10, §11).

Same shape as etl/interest_rate_features.py and etl/commodity_features.py:
the `sentiment` table has one row per WEEKLY sample (see
ingestion/fetch_news_sentiment.py for why it's weekly, not daily), so
pandas.merge_asof(direction="backward") forward-fills each trading day
from the most recent sample - the same no-leakage guarantee as the other
two sources, tested the same way in tests/test_time_alignment.py.

Features produced: sentiment_score, article_count.
"""
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import text

sys.path.append(str(Path(__file__).resolve().parent.parent))
from db.connection import get_engine
from etl.features_common import load_daily_dates, to_long, upsert_features
from etl.time_alignment import forward_fill_to_daily


def load_events(engine) -> pd.DataFrame:
    df = pd.read_sql(
        text("SELECT date, sentiment_score, article_count FROM sentiment WHERE source = 'GDELT' ORDER BY date"),
        engine,
    )
    df["sentiment_score"] = df["sentiment_score"].astype(float)
    df["article_count"] = df["article_count"].astype(float)
    return df


def build_features(engine) -> pd.DataFrame:
    daily_dates = load_daily_dates(engine)
    events = load_events(engine)

    wide = pd.DataFrame({"date": pd.to_datetime(daily_dates).sort_values().reset_index(drop=True)})
    wide["sentiment_score"] = forward_fill_to_daily(daily_dates, events, value_col="sentiment_score").to_numpy()
    wide["article_count"] = forward_fill_to_daily(daily_dates, events, value_col="article_count").to_numpy()
    return wide


def main():
    engine = get_engine()
    wide = build_features(engine)
    print(f"Aligned sentiment features for {len(wide)} trading days")

    long_df = to_long(wide)
    print(f"  {len(long_df)} feature values have a real value (rest are before the first known sample)")

    n = upsert_features(engine, long_df)
    print(f"Upserted {n} row(s) into features")


if __name__ == "__main__":
    main()
