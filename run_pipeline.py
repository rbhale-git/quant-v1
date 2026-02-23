"""Run the full analysis pipeline manually for development/testing."""
import sys
import logging
from datetime import date, timedelta
from data.fetcher import Fetcher
from data.store import Store
from data.screener import Screener
from analysis.indicators import compute_all_indicators
from analysis.regression import compute_regression
from analysis.ml_models import StockPredictor
from config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run_fetch(tickers: list[str] = None, period: str = "1y"):
    fetcher = Fetcher()
    store = Store(settings.db_path)

    if tickers is None:
        logger.info("Fetching S&P 500 ticker list...")
        tickers = fetcher.get_sp500_tickers()

    logger.info(f"Fetching daily data for {len(tickers)} tickers...")
    for i, ticker in enumerate(tickers):
        try:
            df = fetcher.fetch_daily(ticker, period=period)
            if not df.empty:
                store.save_daily_prices(df)
            if (i + 1) % 50 == 0:
                logger.info(f"  Fetched {i + 1}/{len(tickers)}")
        except Exception as e:
            logger.error(f"  Error fetching {ticker}: {e}")

    logger.info("Data fetch complete.")
    store.close()


def run_screen():
    store = Store(settings.db_path)
    fetcher = Fetcher()
    screener = Screener()

    tickers = fetcher.get_sp500_tickers()
    end = date.today()
    start = end - timedelta(days=365)

    logger.info("Loading and enriching data for screening...")
    import pandas as pd
    frames = []
    for ticker in tickers:
        df = store.load_daily_prices(ticker, start, end)
        if df.empty or len(df) < 200:
            continue
        df = compute_all_indicators(df)
        df["ticker"] = ticker
        frames.append(df)

    if not frames:
        logger.warning("No data to screen.")
        store.close()
        return

    universe = pd.concat(frames, ignore_index=True)
    passed = screener.screen(universe)
    passed_tickers = passed["ticker"].unique()
    logger.info(f"Screener passed {len(passed_tickers)} stocks: {list(passed_tickers)[:20]}...")

    for ticker in passed_tickers:
        store.add_to_watchlist(ticker, source="screener")

    logger.info("Screening complete.")
    store.close()


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "fetch"
    if cmd == "fetch":
        tickers = sys.argv[2:] if len(sys.argv) > 2 else None
        run_fetch(tickers)
    elif cmd == "screen":
        run_screen()
    else:
        print(f"Usage: python run_pipeline.py [fetch|screen] [tickers...]")
