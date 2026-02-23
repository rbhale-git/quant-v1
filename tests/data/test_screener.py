import pytest
import pandas as pd
import numpy as np
from data.screener import Screener


@pytest.fixture
def sample_universe():
    """Simulate indicator-enriched data for multiple stocks."""
    np.random.seed(42)
    frames = []
    for ticker in ["AAPL", "MSFT", "LOW_VOL", "LOW_ATR"]:
        n = 250
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        volume = np.random.randint(500_000, 5_000_000, n)
        if ticker == "LOW_VOL":
            volume = np.random.randint(1_000, 100_000, n)  # Below threshold
        atr = np.abs(np.random.randn(n)) + 1.0
        if ticker == "LOW_ATR":
            atr = np.full(n, 0.1)  # Below threshold
        df = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=n, freq="B"),
            "ticker": ticker,
            "close": close,
            "volume": volume,
            "sma_200": pd.Series(close).rolling(200).mean(),
            "rsi": np.random.rand(n) * 100,
            "atr": atr,
        })
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def test_screen_filters_by_volume(sample_universe):
    screener = Screener(min_avg_volume=200_000)
    result = screener.screen(sample_universe)
    assert "LOW_VOL" not in result["ticker"].unique()


def test_screen_filters_by_atr(sample_universe):
    screener = Screener(atr_threshold=0.5)
    result = screener.screen(sample_universe)
    assert "LOW_ATR" not in result["ticker"].unique()


def test_screen_returns_passing_stocks(sample_universe):
    screener = Screener(min_avg_volume=200_000, atr_threshold=0.5)
    result = screener.screen(sample_universe)
    passing = result["ticker"].unique()
    assert "AAPL" in passing or "MSFT" in passing


def test_generate_composite_signal():
    screener = Screener()
    signal = screener.compute_composite_signal(
        ml_prediction="BUY", ml_confidence=0.8,
        indicator_alignment=0.7,
        sentiment_score=0.5,
    )
    assert signal["signal"] in ["BUY", "SELL", "HOLD"]
    assert 0 <= signal["confidence"] <= 100
