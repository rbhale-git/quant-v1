import pytest
import pandas as pd
import numpy as np
from analysis.indicators import (
    compute_sma, compute_ema, compute_bollinger_bands,
    compute_rsi, compute_macd, compute_vwap, compute_atr,
    compute_all_indicators,
)


@pytest.fixture
def sample_data():
    np.random.seed(42)
    n = 250
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n))
    low = close - np.abs(np.random.randn(n))
    open_ = close + np.random.randn(n) * 0.3
    volume = np.random.randint(1_000_000, 10_000_000, size=n)
    return pd.DataFrame({
        "date": dates, "open": open_, "high": high,
        "low": low, "close": close, "volume": volume,
    })


def test_sma(sample_data):
    result = compute_sma(sample_data, window=20)
    assert f"sma_20" in result.columns
    assert result["sma_20"].isna().sum() == 19
    assert not result["sma_20"].iloc[-1:].isna().any()


def test_ema(sample_data):
    result = compute_ema(sample_data, window=12)
    assert "ema_12" in result.columns
    assert not result["ema_12"].iloc[-1:].isna().any()


def test_bollinger_bands(sample_data):
    result = compute_bollinger_bands(sample_data, window=20, num_std=2.0)
    assert "bb_upper" in result.columns
    assert "bb_middle" in result.columns
    assert "bb_lower" in result.columns
    assert "bb_pct" in result.columns
    last = result.iloc[-1]
    assert last["bb_upper"] > last["bb_middle"] > last["bb_lower"]


def test_rsi(sample_data):
    result = compute_rsi(sample_data, window=14)
    assert "rsi" in result.columns
    valid = result["rsi"].dropna()
    assert (valid >= 0).all() and (valid <= 100).all()


def test_macd(sample_data):
    result = compute_macd(sample_data, fast=12, slow=26, signal=9)
    assert "macd" in result.columns
    assert "macd_signal" in result.columns
    assert "macd_histogram" in result.columns


def test_vwap(sample_data):
    result = compute_vwap(sample_data)
    assert "vwap" in result.columns
    valid = result["vwap"].dropna()
    assert len(valid) > 0


def test_atr(sample_data):
    result = compute_atr(sample_data, window=14)
    assert "atr" in result.columns
    valid = result["atr"].dropna()
    assert (valid > 0).all()


def test_compute_all_indicators(sample_data):
    result = compute_all_indicators(sample_data)
    expected_cols = [
        "sma_20", "sma_50", "sma_200",
        "ema_12", "ema_26",
        "bb_upper", "bb_lower", "bb_pct",
        "rsi", "macd", "macd_signal", "macd_histogram",
        "vwap", "atr",
    ]
    for col in expected_cols:
        assert col in result.columns, f"Missing column: {col}"
