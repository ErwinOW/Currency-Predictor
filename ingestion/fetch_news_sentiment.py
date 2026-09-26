"""News-sentiment ingestion (§4.9, §5): GDELT bulk Global Knowledge Graph files.

Why bulk files instead of GDELT's own search API (api.gdeltproject.org):
that API only accepts dates from 2017 onward and, from this environment,
returned HTTP 429 (rate limited) on every attempt regardless of how long
between requests - likely a shared/exhausted limit on its free anonymous
tier, not something fixable by waiting longer. The BULK files
(data.gdeltproject.org) are a completely separate, unthrottled path: they
are just static files, published every 15 minutes since 2015-02-18,
covering the full history this project needs.

The real cost of that reliability: each 15-minute file is a dump of ALL
global news in that window (~2,600 articles, ~10 MB compressed), not
pre-filtered for anything. Downloading and filtering EVERY 15-minute file
since 2015 (~96/day) would mean roughly 400,000 files and 4 TB - not
practical for a portfolio project. This script instead samples ONE file
per week (SAMPLE_INTERVAL_DAYS below), at a fixed time each week, and
treats that sample as representative of that week's news tone. That is a
real, explicit tradeoff: this is a WEEKLY-resolution feature (like
interest rates), not a true daily aggregate - documented here rather than
quietly presented as more precise than it is. A single sample still
typically contains 50-100+ articles mentioning Malaysia or Indonesia
(checked against a real file before committing to this approach), enough
for a meaningfully stable average, not just a couple of noisy headlines.

Each GKG row is tab-separated with NO header row - field positions are
fixed by GDELT's schema. The two fields used here (confirmed against a
real downloaded file, not assumed from documentation alone):
    field[1]  DATE            e.g. "20170103120000"
    field[9]  V1Locations     e.g. "4#Kuala Lumpur, Kuala Lumpur, Malaysia#MY..."
    field[15] V2Tone          e.g. "-0.74,1.11,1.85,2.96,14.8,..." (tone is the first number)

Usage:
    python ingestion/fetch_news_sentiment.py               # incremental (see default_start_date)
    python ingestion/fetch_news_sentiment.py --start 2015-02-19  # full historical backfill
"""
import argparse
import json
import sys
import zipfile
from datetime import date, datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests
from sqlalchemy import text

sys.path.append(str(Path(__file__).resolve().parent.parent))
from db.connection import get_engine

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "news"
BASE_URL = "http://data.gdeltproject.org/gdeltv2/{ts}.gkg.csv.zip"
SAMPLE_INTERVAL_DAYS = 7
SAMPLE_HOUR_UTC = 12  # roughly midday UTC, arbitrary but fixed for consistency
COUNTRIES = ("Malaysia", "Indonesia")
VALID_TONE_RANGE = (-20, 20)  # GDELT tone is usually within roughly -10..+10


def weekly_sample_dates(start: date, end: date) -> list[date]:
    dates = []
    current = start
    while current <= end:
        dates.append(current)
        current += timedelta(days=SAMPLE_INTERVAL_DAYS)
    return dates


def fetch_one_sample(sample_date: date) -> dict | None:
    """Download one 15-minute GKG file, filter to Malaysia/Indonesia rows,
    return the aggregate for that sample - or None if the file doesn't
    exist (GDELT has occasional gaps) or nothing matched.
    """
    ts = f"{sample_date:%Y%m%d}{SAMPLE_HOUR_UTC:02d}0000"
    url = BASE_URL.format(ts=ts)

    resp = requests.get(url, timeout=60)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()

    with zipfile.ZipFile(BytesIO(resp.content)) as z:
        raw = z.read(z.namelist()[0]).decode("utf-8", errors="replace")

    matches = []
    for line in raw.split("\n"):
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) <= 15:
            continue
        locations = fields[9]
        if any(country in locations for country in COUNTRIES):
            tone_field = fields[15]
            try:
                tone = float(tone_field.split(",")[0])
            except (ValueError, IndexError):
                continue
            matches.append(tone)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    (RAW_DIR / f"gdelt_{ts}_filtered.json").write_text(
        json.dumps({"date": sample_date.isoformat(), "tones": matches}, indent=2)
    )

    if not matches:
        return None

    return {
        "date": sample_date,
        "sentiment_score": sum(matches) / len(matches),
        "article_count": len(matches),
    }


def validate(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset=["date"])
    df = df.dropna(subset=["sentiment_score"])
    df = df[df["sentiment_score"].between(*VALID_TONE_RANGE)]
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
        INSERT INTO sentiment (date, source, sentiment_score, article_count)
        VALUES (:date, :source, :sentiment_score, :article_count)
        ON CONFLICT (date, source)
        DO UPDATE SET sentiment_score = EXCLUDED.sentiment_score, article_count = EXCLUDED.article_count
        """
    )
    records = [
        {
            "date": row.date,
            "source": "GDELT",
            "sentiment_score": float(row.sentiment_score),
            "article_count": int(row.article_count),
        }
        for row in df.itertuples()
    ]
    with engine.begin() as conn:
        conn.execute(upsert_sql, records)
    return len(records)


EARLIEST_AVAILABLE = date(2015, 2, 19)  # GDELT GKG 2.0's own launch date


def default_start_date(engine) -> date:
    """Incremental by default: resume from the week after the most recent
    sample already stored, instead of re-fetching the full history every
    run. Re-fetching from scratch (~25-30 min) is fine for a one-time
    backfill but not for a scheduled job that might run daily - see
    automation/pipeline.py, which calls this script on a weekly cadence
    specifically because that matches how often new samples actually
    appear.
    """
    with engine.connect() as conn:
        latest = conn.execute(
            text("SELECT MAX(date) FROM sentiment WHERE source = 'GDELT'")
        ).scalar()
    if latest is None:
        return EARLIEST_AVAILABLE
    return latest + timedelta(days=SAMPLE_INTERVAL_DAYS)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=None, help="Defaults to incremental: the week after the latest stored sample")
    parser.add_argument("--end", default=date.today().isoformat())
    args = parser.parse_args(argv)

    start = date.fromisoformat(args.start) if args.start else default_start_date(get_engine())
    end = date.fromisoformat(args.end)

    if start > end:
        print(f"Already up to date (latest sample would start {start}, nothing to fetch)")
        return
    sample_dates = weekly_sample_dates(start, end)
    print(f"Sampling {len(sample_dates)} weekly snapshots from {start} to {end}...")

    rows = []
    no_data = 0
    for i, sample_date in enumerate(sample_dates, 1):
        result = fetch_one_sample(sample_date)
        if result is None:
            no_data += 1
        else:
            rows.append(result)
        if i % 25 == 0 or i == len(sample_dates):
            print(f"  {i}/{len(sample_dates)} samples fetched ({len(rows)} with matches, {no_data} empty/missing)")

    df = pd.DataFrame(rows)
    print(f"Parsed {len(df)} weekly sentiment readings")

    df = validate(df)
    n = load_to_postgres(df)
    print(f"Upserted {n} row(s) into sentiment")


if __name__ == "__main__":
    main()
