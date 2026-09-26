"""Long-running scheduler entrypoint (§21).

Runs the daily and weekly pipeline jobs (automation/pipeline.py) on a
fixed schedule, forever, in this process. This is deliberately the
simple option the spec recommends starting with (APScheduler) rather
than Airflow - see docs/project_rundown.md §21 and §24.

Usage:
    python -m automation.scheduler

Runs in the foreground - use a process manager (systemd, Task Scheduler,
pm2, etc.) or `nohup ... &` to keep it running unattended. Ctrl+C to stop.
"""
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from automation.pipeline import daily_job, logger, weekly_job

# Times are somewhat arbitrary but deliberately AFTER both data sources'
# own daily update times (Frankfurter/FRED/BNM all publish once daily,
# typically early), so a run doesn't race a source that hasn't updated yet.
DAILY_HOUR_UTC = 6
WEEKLY_DAY_OF_WEEK = "mon"
WEEKLY_HOUR_UTC = 7


def main():
    scheduler = BlockingScheduler(timezone="UTC")

    scheduler.add_job(
        daily_job,
        trigger=CronTrigger(hour=DAILY_HOUR_UTC, minute=0),
        id="daily_pipeline",
        name="Daily: exchange rates, commodities, live prediction",
    )
    scheduler.add_job(
        weekly_job,
        trigger=CronTrigger(day_of_week=WEEKLY_DAY_OF_WEEK, hour=WEEKLY_HOUR_UTC, minute=0),
        id="weekly_pipeline",
        name="Weekly: interest rates, news sentiment, backtest",
    )

    logger.info(
        "Scheduler started. Daily job at %02d:00 UTC, weekly job %s at %02d:00 UTC.",
        DAILY_HOUR_UTC, WEEKLY_DAY_OF_WEEK, WEEKLY_HOUR_UTC,
    )
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped.")


if __name__ == "__main__":
    main()
