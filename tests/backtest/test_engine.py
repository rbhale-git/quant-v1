import pytest
import pandas as pd
import numpy as np
from backtest.engine import BacktestEngine
from backtest.strategies import SmaCrossover


@pytest.fixture
def price_data():
    np.random.seed(42)
    n = 200
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    df = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=n, freq="B"),
        "open": close + np.random.randn(n) * 0.1,
        "high": close + abs(np.random.randn(n)),
        "low": close - abs(np.random.randn(n)),
        "close": close,
        "volume": np.random.randint(1_000_000, 10_000_000, n),
        "sma_20": pd.Series(close).rolling(20).mean(),
        "sma_50": pd.Series(close).rolling(50).mean(),
        "rsi": np.random.rand(n) * 100,
        "bb_lower": close - 2,
        "bb_upper": close + 2,
        "bb_pct": np.random.rand(n),
        "atr": np.abs(np.random.randn(n)) + 0.5,
    })
    return df


def test_backtest_runs(price_data):
    engine = BacktestEngine(starting_capital=10_000)
    strategy = SmaCrossover()
    result = engine.run(price_data, strategy)
    assert "total_return" in result
    assert "sharpe_ratio" in result
    assert "max_drawdown" in result
    assert "win_rate" in result
    assert "trade_count" in result
    assert "equity_curve" in result
    assert "trades" in result


def test_backtest_equity_curve(price_data):
    engine = BacktestEngine(starting_capital=10_000)
    strategy = SmaCrossover()
    result = engine.run(price_data, strategy)
    curve = result["equity_curve"]
    assert len(curve) == len(price_data)
    assert curve.iloc[0] == 10_000


def test_backtest_no_negative_cash(price_data):
    engine = BacktestEngine(starting_capital=10_000)
    strategy = SmaCrossover()
    result = engine.run(price_data, strategy)
    assert all(v >= 0 for v in result["equity_curve"])
