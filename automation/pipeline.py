"""Automated pipeline jobs (§21) with basic monitoring (§22).

Two jobs, on different cadences, because the data sources don't all
change at the same rate:

    daily_job()   exchange rates -> price features -> commodities ->
                  commodity features -> a fresh live prediction.
                  Everything here genuinely changes day to day.

    weekly_job()  interest rates (change a handful of times a YEAR),
                  news sentiment (this project samples it weekly on
                  purpose - see ingestion/fetch_news_sentiment.py), and
                  a full backtest (§21: "retraining can occur weekly").
                  Running these daily would just re-fetch data that
                  hasn't changed and re-run an expensive backtest for
                  no new information.

Each step runs in isolation: one source failing (a dead API, a network
blip) is logged and skipped, not allowed to take down the rest of the
day's pipeline - see docs/project_rundown.md §22's example log format,
which this mirrors (component name + SUCCESS/FAILED per line).

Every ingestion script here is called with argv=[] specifically so it
uses its own built-in defaults (regardless of what this process's own
sys.argv happens to contain) - for the daily-changing sources that means
the full default range (cheap: one API call covers the whole range
either way), and for news sentiment specifically it means incremental
(resuming from the last stored sample, not a full 2015- backfill every
time - see ingestion/fetch_news_sentiment.py's default_start_date()).
"""
import logging
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "pipeline.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("pipeline")


def run_step(name: str, func, *args, **kwargs) -> bool:
    """Run one pipeline step, logging SUCCESS/FAILED either way, and
    never let one step's exception stop the rest of the pipeline.
    """
    start = time.monotonic()
    try:
        func(*args, **kwargs)
        logger.info("%s: SUCCESS (%.1fs)", name, time.monotonic() - start)
        return True
    except Exception:
        logger.exception("%s: FAILED (%.1fs)", name, time.monotonic() - start)
        return False


def daily_job():
    logger.info("=== daily pipeline starting ===")

    from ingestion.fetch_exchange_rates import main as fetch_exchange_rates
    from etl.feature_engineering import main as build_price_features
    from ingestion.fetch_commodity_prices import main as fetch_commodities
    from etl.commodity_features import main as build_commodity_features
    from ml.generate_prediction import main as generate_prediction

    ok = run_step("fetch_exchange_rates", fetch_exchange_rates, argv=[])
    if ok:
        run_step("build_price_features", build_price_features)

    ok = run_step("fetch_commodity_prices", fetch_commodities, argv=[])
    if ok:
        run_step("build_commodity_features", build_commodity_features)

    # Always attempt a fresh prediction, even if a data refresh above
    # failed - it'll just train on slightly older data rather than none.
    run_step("generate_prediction", generate_prediction)

    logger.info("=== daily pipeline finished ===")


def weekly_job():
    logger.info("=== weekly pipeline starting ===")

    from ingestion.fetch_interest_rates import main as fetch_interest_rates
    from etl.interest_rate_features import main as build_interest_rate_features
    from ingestion.fetch_news_sentiment import main as fetch_news_sentiment
    from etl.sentiment_features import main as build_sentiment_features
    from ml.backtest import main as run_backtest

    ok = run_step("fetch_interest_rates", fetch_interest_rates, argv=[])
    if ok:
        run_step("build_interest_rate_features", build_interest_rate_features)

    ok = run_step("fetch_news_sentiment", fetch_news_sentiment, argv=[])
    if ok:
        run_step("build_sentiment_features", build_sentiment_features)

    run_step("run_backtest", run_backtest)

    logger.info("=== weekly pipeline finished ===")


if __name__ == "__main__":
    # Manual one-off run, e.g. `python -m automation.pipeline daily`
    target = sys.argv[1] if len(sys.argv) > 1 else "daily"
    {"daily": daily_job, "weekly": weekly_job}[target]()
