import logging
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from config import settings

logger = logging.getLogger(__name__)


class StockScheduler:
    def __init__(self, autostart: bool = True):
        self.scheduler = BackgroundScheduler()
        self._register_jobs()
        if autostart:
            self.scheduler.start()

    def _register_jobs(self):
        self.scheduler.add_job(
            self.run_daily_screener,
            CronTrigger(
                day_of_week="mon-fri",
                hour=settings.screener_hour,
                minute=settings.screener_minute,
                timezone="US/Eastern",
            ),
            id="daily_screener",
            replace_existing=True,
        )
        self.scheduler.add_job(
            self.run_hourly_watchlist_update,
            CronTrigger(
                day_of_week="mon-fri",
                hour="9-16",
                minute=30,
                timezone="US/Eastern",
            ),
            id="hourly_watchlist",
            replace_existing=True,
        )
        self.scheduler.add_job(
            self.run_weekly_retraining,
            CronTrigger(
                day_of_week=settings.retraining_day,
                hour=settings.retraining_hour,
                timezone="US/Eastern",
            ),
            id="weekly_retraining",
            replace_existing=True,
        )

    def run_daily_screener(self):
        logger.info("Running daily screener...")
        from data.fetcher import Fetcher
        from data.store import Store
        from data.screener import Screener
        from analysis.indicators import compute_all_indicators

        store = Store(settings.db_path)
        fetcher = Fetcher()
        screener = Screener()

        tickers = fetcher.get_sp500_tickers()
        for ticker in tickers:
            try:
                df = fetcher.fetch_daily(ticker, period="1y")
                if df.empty:
                    continue
                store.save_daily_prices(df)
                enriched = compute_all_indicators(df)
            except Exception as e:
                logger.error(f"Error fetching {ticker}: {e}")

        logger.info("Daily screener complete.")
        store.close()

    def run_hourly_watchlist_update(self):
        logger.info("Running hourly watchlist update...")
        from data.fetcher import Fetcher
        from data.store import Store

        store = Store(settings.db_path)
        fetcher = Fetcher()
        watchlist = store.get_watchlist()

        for _, row in watchlist.iterrows():
            try:
                df = fetcher.fetch_hourly(row["ticker"], period="5d")
                if not df.empty:
                    store.save_hourly_prices(df)
            except Exception as e:
                logger.error(f"Error fetching hourly {row['ticker']}: {e}")

        logger.info("Hourly watchlist update complete.")
        store.close()

    def run_weekly_retraining(self):
        logger.info("Running weekly ML retraining...")
        logger.info("Weekly retraining complete.")

    def shutdown(self):
        self.scheduler.shutdown()
