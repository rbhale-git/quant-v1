import pytest
import pandas as pd
import numpy as np
from backtest.strategies import SmaCrossover, RsiBollinger, CompositeStrategy


@pytest.fixture
def data_with_indicators():
    np.random.seed(42)
    n = 100
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    df = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=n, freq="B"),
        "close": close,
        "sma_20": pd.Series(close).rolling(20).mean(),
        "sma_50": pd.Series(close).rolling(50).mean(),
        "rsi": np.random.rand(n) * 100,
        "bb_lower": close - 2,
        "bb_upper": close + 2,
        "bb_pct": np.random.rand(n),
    })
    return df


def test_sma_crossover_generates_signals(data_with_indicators):
    strategy = SmaCrossover()
    signals = strategy.generate_signals(data_with_indicators)
    assert len(signals) == len(data_with_indicators)
    assert set(signals.unique()).issubset({"BUY", "SELL", "HOLD"})


def test_rsi_bollinger_generates_signals(data_with_indicators):
    strategy = RsiBollinger()
    signals = strategy.generate_signals(data_with_indicators)
    assert len(signals) == len(data_with_indicators)
    assert set(signals.unique()).issubset({"BUY", "SELL", "HOLD"})


def test_composite_strategy(data_with_indicators):
    strategies = [SmaCrossover(), RsiBollinger()]
    composite = CompositeStrategy(strategies, weights=[0.5, 0.5])
    signals = composite.generate_signals(data_with_indicators)
    assert len(signals) == len(data_with_indicators)
    assert set(signals.unique()).issubset({"BUY", "SELL", "HOLD"})
