"""Interest-rate ingestion (§4.2, §6): Malaysia OPR + Indonesia policy-proxy rate.

Two different sources, because FRED doesn't carry a Malaysia series at all
(checked via its search API and several known ID patterns - nothing exists).
This is a normal shape for real data engineering: not every country's data
lives with the same provider.

    Indonesia -> FRED series IRSTCI01IDM156N (monthly interbank rate, tracks
                 Bank Indonesia's policy rate closely)
    Malaysia  -> Bank Negara Malaysia's own public API (api.bnm.gov.my),
                 free, no key required. Returns the actual OPR decision
                 dates - better than a monthly proxy, since it's the real
                 event data §4.5 describes.

Both sources are EVENT-based or low-frequency, not one row per calendar
day (BNM only has ~6 rows/year; FRED has 1/month). That's expected - the
`interest_rates` table stores exactly what was published, on the date it
was published. Turning that into a value for every day (without leaking
future decisions backward) is a separate step: etl/interest_rate_features.py.

Usage:
    python ingestion/fetch_interest_rates.py --start-year 2014
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

import os

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "interest_rates"
FRED_SERIES_ID = "IRSTCI01IDM156N"


def fetch_indonesia(start_year: int) -> pd.DataFrame:
    """Bank Indonesia policy-rate proxy, from FRED. One row per month."""
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        raise RuntimeError("FRED_API_KEY is not set in .env")

    resp = requests.get(
        "https://api.stlouisfed.org/fred/series/observations",
        params={
            "series_id": FRED_SERIES_ID,
            "api_key": api_key,
            "file_type": "json",
            "observation_start": f"{start_year}-01-01",
        },
        timeout=30,
    )
    resp.raise_for_status()
    payload = resp.json()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    (RAW_DIR / f"fred_indonesia_{start_year}.json").write_text(json.dumps(payload, indent=2))

    rows = [
        {"date": obs["date"], "rate": float(obs["value"])}
        for obs in payload.get("observations", [])
        if obs["value"] != "."  # FRED's placeholder for "no data that month"
    ]
    df = pd.DataFrame(rows)
    df["country"] = "Indonesia"
    return df


def fetch_malaysia(start_year: int) -> pd.DataFrame:
    """Bank Negara Malaysia OPR decisions, one row per policy meeting."""
    all_rows = []
    for year in range(start_year, date.today().year + 1):
        resp = requests.get(
            f"https://api.bnm.gov.my/public/opr/year/{year}",
            headers={"Accept": "application/vnd.BNM.API.v1+json"},
            timeout=30,
        )
        if resp.status_code == 404:
            continue  # no meetings recorded for this year (e.g. too far in the future)
        resp.raise_for_status()
        payload = resp.json()

        RAW_DIR.mkdir(parents=True, exist_ok=True)
        (RAW_DIR / f"bnm_malaysia_{year}.json").write_text(json.dumps(payload, indent=2))

        for row in payload.get("data", []):
            all_rows.append({"date": row["date"], "rate": float(row["new_opr_level"])})

    df = pd.DataFrame(all_rows)
    df["country"] = "Malaysia"
    return df


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Sanity-check per §9: drop duplicates/nulls, plausible-range check.

    Both OPR and BI-linked rates have stayed within roughly 1-13% over the
    last decade; a value outside a wide 0-20% band is more likely a parsing
    error than a real rate.
    """
    before = len(df)
    df = df.drop_duplicates(subset=["date", "country"])
    df = df.dropna(subset=["rate"])
    df = df[(df["rate"] >= 0) & (df["rate"] <= 20)]
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
        INSERT INTO interest_rates (date, country, rate)
        VALUES (:date, :country, :rate)
        ON CONFLICT (date, country)
        DO UPDATE SET rate = EXCLUDED.rate
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

    print("Fetching Indonesia rate (FRED)...")
    indonesia = fetch_indonesia(args.start_year)
    print(f"  parsed {len(indonesia)} monthly reading(s)")

    print("Fetching Malaysia OPR (Bank Negara Malaysia)...")
    malaysia = fetch_malaysia(args.start_year)
    print(f"  parsed {len(malaysia)} decision(s)")

    combined = pd.concat([indonesia, malaysia], ignore_index=True)
    combined = validate(combined)

    n = load_to_postgres(combined)
    print(f"Upserted {n} row(s) into interest_rates")


if __name__ == "__main__":
    main()
