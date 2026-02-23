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


def run_signals(tickers: list[str] = None):
    store = Store(settings.db_path)
    screener = Screener()

    if tickers is None:
        wl = store.get_watchlist()
        if wl.empty:
            logger.warning("Watchlist is empty. Run screener first or provide tickers.")
            store.close()
            return
        tickers = wl["ticker"].tolist()

    end = date.today()
    start = end - timedelta(days=365)

    # Try loading ML model if it exists
    import os
    model_path = os.path.join(settings.ml_model_dir, "stock_predictor.joblib")
    predictor = None
    if os.path.exists(model_path):
        predictor = StockPredictor()
        predictor.load(model_path)
        logger.info("Loaded ML model for predictions.")

    import pandas as pd
    signal_rows = []
    for ticker in tickers:
        try:
            df = store.load_daily_prices(ticker, start, end)
            if df.empty or len(df) < 30:
                continue
            df = compute_all_indicators(df)
            df = compute_regression(df)

            latest = df.iloc[-1]

            # ML prediction
            ml_prediction = "HOLD"
            ml_confidence = 0.5
            if predictor is not None:
                avail_cols = [c for c in predictor._feature_cols if c in df.columns]
                if avail_cols:
                    row_data = df[avail_cols].iloc[[-1]].dropna(axis=1)
                    if not row_data.empty:
                        pred = predictor.predict(row_data)
                        if pred:
                            ml_prediction = pred[0]["signal"]
                            ml_confidence = pred[0]["confidence"]

            # Indicator alignment: fraction of bullish signals
            bullish_count = 0
            total = 0
            if "rsi" in df.columns:
                total += 1
                if latest.get("rsi", 50) < 40:
                    bullish_count += 1
            if "macd_histogram" in df.columns:
                total += 1
                if latest.get("macd_histogram", 0) > 0:
                    bullish_count += 1
            if "sma_20" in df.columns and "sma_50" in df.columns:
                total += 1
                if latest.get("sma_20", 0) > latest.get("sma_50", 0):
                    bullish_count += 1
            if "reg_slope_30" in df.columns:
                total += 1
                if latest.get("reg_slope_30", 0) > 0:
                    bullish_count += 1
            indicator_alignment = (bullish_count / total * 2 - 1) if total > 0 else 0.0

            # Sentiment defaults to 0.0
            sentiment_score = 0.0

            composite = screener.compute_composite_signal(
                ml_prediction, ml_confidence, indicator_alignment, sentiment_score,
            )

            signal_rows.append({
                "date": end,
                "ticker": ticker,
                "signal": composite["signal"],
                "confidence": composite["confidence"],
                "ml_prediction": ml_prediction,
                "ml_confidence": round(ml_confidence, 3),
                "indicator_alignment": round(indicator_alignment, 3),
                "sentiment_score": sentiment_score,
            })
            logger.info(f"  {ticker}: {composite['signal']} (confidence={composite['confidence']})")
        except Exception as e:
            logger.error(f"  Error generating signal for {ticker}: {e}")

    if signal_rows:
        signals_df = pd.DataFrame(signal_rows)
        store.save_signals(signals_df)
        logger.info(f"Saved {len(signal_rows)} signals.")
    else:
        logger.warning("No signals generated.")

    store.close()


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "fetch"
    tickers = sys.argv[2:] if len(sys.argv) > 2 else None
    if cmd == "fetch":
        run_fetch(tickers)
    elif cmd == "screen":
        run_screen()
    elif cmd == "signals":
        run_signals(tickers)
    elif cmd == "all":
        run_fetch(tickers)
        run_screen()
        run_signals(tickers)
    else:
        print(f"Usage: python run_pipeline.py [fetch|screen|signals|all] [tickers...]")
