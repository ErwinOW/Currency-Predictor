"""Commodity-price ingestion (§4.7, §6): Brent crude, WTI crude, Malaysian palm oil.

Three different fetches, two different providers:

    Brent, WTI -> FRED (same provider as the Indonesia interest rate),
                  genuinely daily series, no gaps in principle.
    Palm oil   -> FRED only has this MONTHLY - checked its search API,
                  nothing daily exists there. Yahoo Finance's unofficial
                  chart endpoint has the real thing instead: ticker CPO=F,
                  "USD Malaysian Crude Palm Oil", traded on the CME,
                  confirmed daily back to 2014.

Two data-quality quirks worth keeping in mind (both are REAL, not bugs to
"fix away"):

  1. WTI genuinely went NEGATIVE in April 2020 (the well-known COVID
     storage-crunch event, when the futures contract's underlying spot
     proxy briefly printed below zero). A naive "price must be positive"
     filter would wrongly discard real data - see validate()'s per-
     commodity bounds below.
  2. Yahoo's chart API silently returns coarser-than-requested data for
     very long date ranges (asking for "max" history returned monthly
     candles for CPO=F, not daily) - the fix is requesting an explicit
     period1/period2 (start/end) window instead of a relative range like
     "max", which does return true daily data. See fetch_palm_oil().

Usage:
    python ingestion/fetch_commodity_prices.py --start-year 2014
"""
import argparse
import json
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from sqlalchemy import text

sys.path.append(str(Path(__file__).resolve().parent.parent))
from db.connection import get_engine

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "commodities"

FRED_SERIES = {
    "Brent Crude": "DCOILBRENTEU",
    "WTI Crude": "DCOILWTICO",
}

# Wide, commodity-specific sanity bands (§9) - deliberately not one shared
# range, since these commodities behave very differently (WTI's real 2020
# low was -36.98; Brent has never been negative; palm oil trades in the
# hundreds, not tens).
VALID_RANGES = {
    "Brent Crude": (0, 250),
    "WTI Crude": (-60, 250),
    "Palm Oil": (200, 2500),
}


def fetch_fred_commodity(commodity: str, series_id: str, start_year: int) -> pd.DataFrame:
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        raise RuntimeError("FRED_API_KEY is not set in .env")

    resp = requests.get(
        "https://api.stlouisfed.org/fred/series/observations",
        params={
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
            "observation_start": f"{start_year}-01-01",
        },
        timeout=30,
    )
    resp.raise_for_status()
    payload = resp.json()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    (RAW_DIR / f"fred_{series_id}_{start_year}.json").write_text(json.dumps(payload, indent=2))

    rows = [
        {"date": obs["date"], "price": float(obs["value"])}
        for obs in payload.get("observations", [])
        if obs["value"] != "."  # FRED's placeholder for "no reading that day"
    ]
    df = pd.DataFrame(rows)
    df["commodity"] = commodity
    return df


def fetch_palm_oil(start_year: int) -> pd.DataFrame:
    """Yahoo Finance CPO=F, using an explicit start/end window (see module
    docstring for why - a relative range like "max" silently returns
    monthly data instead of daily for this particular ticker).
    """
    period1 = int(datetime(start_year, 1, 1, tzinfo=timezone.utc).timestamp())
    period2 = int(datetime.now(timezone.utc).timestamp())

    resp = requests.get(
        "https://query1.finance.yahoo.com/v8/finance/chart/CPO=F",
        params={"period1": period1, "period2": period2, "interval": "1d"},
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=30,
    )
    resp.raise_for_status()
    payload = resp.json()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    (RAW_DIR / f"yahoo_palm_oil_{start_year}.json").write_text(json.dumps(payload, indent=2))

    result = payload["chart"]["result"][0]
    timestamps = result["timestamp"]
    closes = result["indicators"]["quote"][0]["close"]

    df = pd.DataFrame({
        "date": pd.to_datetime(timestamps, unit="s").date,
        "price": closes,
    })
    df = df.dropna(subset=["price"])  # a day the market was open in the response but didn't trade
    df["commodity"] = "Palm Oil"
    return df


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Sanity-check per §9: drop duplicates/nulls, per-commodity plausible range."""
    before = len(df)
    df = df.drop_duplicates(subset=["date", "commodity"])
    df = df.dropna(subset=["price"])

    keep = pd.Series(False, index=df.index)
    for commodity, (low, high) in VALID_RANGES.items():
        mask = df["commodity"] == commodity
        keep |= mask & df["price"].between(low, high)
    df = df[keep]

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
        INSERT INTO commodities (date, commodity, price)
        VALUES (:date, :commodity, :price)
        ON CONFLICT (date, commodity)
        DO UPDATE SET price = EXCLUDED.price
        """
    )
    records = df.to_dict("records")
    with engine.begin() as conn:
        conn.execute(upsert_sql, records)
    return len(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=2014)
    args = parser.parse_args()

    frames = []
    for commodity, series_id in FRED_SERIES.items():
        print(f"Fetching {commodity} (FRED {series_id})...")
        df = fetch_fred_commodity(commodity, series_id, args.start_year)
        print(f"  parsed {len(df)} daily reading(s)")
        frames.append(df)

    print("Fetching Palm Oil (Yahoo Finance CPO=F)...")
    palm_oil = fetch_palm_oil(args.start_year)
    print(f"  parsed {len(palm_oil)} daily reading(s)")
    frames.append(palm_oil)

    combined = pd.concat(frames, ignore_index=True)
    combined = validate(combined)

    n = load_to_postgres(combined)
    print(f"Upserted {n} row(s) into commodities")


if __name__ == "__main__":
    main()
