"""Step 1 walking-skeleton ingestion script: MYR/IDR historical exchange rates.

Pipeline (per docs/project_rundown.md §6):
    API -> validate -> save raw -> transform -> PostgreSQL

Data source: Frankfurter (https://frankfurter.dev), a free, no-API-key
service built on European Central Bank reference rates. It returns one
daily rate per day (no open/high/low/volume) — that's fine for the first
version; §4.1 lists OHLCV as "potential" fields, Close is the only one
that's required.

Usage:
    python ingestion/fetch_exchange_rates.py --start 2015-01-01 --end 2026-09-01
"""
import argparse
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import requests
from sqlalchemy import text

sys.path.append(str(Path(__file__).resolve().parent.parent))
from db.connection import get_engine

BASE_URL = "https://api.frankfurter.dev/v1/{start}..{end}"
FROM_CCY = "MYR"
TO_CCY = "IDR"
CURRENCY_PAIR = f"{FROM_CCY}/{TO_CCY}"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "exchange_rates"


def fetch(start: str, end: str) -> dict:
    url = BASE_URL.format(start=start, end=end)
    resp = requests.get(url, params={"from": FROM_CCY, "to": TO_CCY}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def save_raw(payload: dict, start: str, end: str) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / f"myridr_{start}_{end}.json"
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path


def to_dataframe(payload: dict) -> pd.DataFrame:
    rows = []
    for day, rates in payload.get("rates", {}).items():
        close = rates.get(TO_CCY)
        if close is None:
            continue
        rows.append({"date": day, "close": close})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.sort_values("date").reset_index(drop=True)
    return df


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Minimal validation per §9: drop duplicates, drop nulls, sanity-check range."""
    before = len(df)
    df = df.drop_duplicates(subset="date")
    df = df.dropna(subset=["close"])
    # MYR/IDR has historically traded roughly between 2,941 and 4,545
    # (2015-2026) — a value outside a wide sanity band is more likely a
    # data error than a real rate.
    df = df[(df["close"] > 2000) & (df["close"] < 6000)]
    after = len(df)
    if after < before:
        print(f"  validation dropped {before - after} row(s)")
    return df


def load_to_postgres(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    engine = get_engine()
    upsert_sql = text(
        """
        INSERT INTO exchange_rates (date, currency_pair, close, source)
        VALUES (:date, :currency_pair, :close, :source)
        ON CONFLICT (date, currency_pair)
        DO UPDATE SET close = EXCLUDED.close, ingested_at = now()
        """
    )
    records = [
        {"date": row.date, "currency_pair": CURRENCY_PAIR, "close": float(row.close), "source": "frankfurter"}
        for row in df.itertuples()
    ]
    with engine.begin() as conn:
        conn.execute(upsert_sql, records)
    return len(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default=date.today().isoformat())
    args = parser.parse_args()

    print(f"Fetching {CURRENCY_PAIR} from {args.start} to {args.end}...")
    payload = fetch(args.start, args.end)

    raw_path = save_raw(payload, args.start, args.end)
    print(f"  saved raw response -> {raw_path}")

    df = to_dataframe(payload)
    print(f"  parsed {len(df)} daily rate(s)")

    df = validate(df)
    n = load_to_postgres(df)
    print(f"  upserted {n} row(s) into exchange_rates")


if __name__ == "__main__":
    main()
