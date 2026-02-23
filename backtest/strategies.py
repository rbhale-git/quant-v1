from abc import ABC, abstractmethod
import pandas as pd
import numpy as np


class Strategy(ABC):
    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Return a Series of 'BUY', 'SELL', or 'HOLD' for each row."""
        pass


class SmaCrossover(Strategy):
    def __init__(self, fast_col: str = "sma_20", slow_col: str = "sma_50"):
        self.fast_col = fast_col
        self.slow_col = slow_col

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series("HOLD", index=df.index)
        if self.fast_col not in df.columns or self.slow_col not in df.columns:
            return signals
        fast = df[self.fast_col]
        slow = df[self.slow_col]
        prev_fast = fast.shift(1)
        prev_slow = slow.shift(1)
        signals[(prev_fast <= prev_slow) & (fast > slow)] = "BUY"
        signals[(prev_fast >= prev_slow) & (fast < slow)] = "SELL"
        return signals


class RsiBollinger(Strategy):
    def __init__(self, rsi_buy: float = 30, rsi_sell: float = 70):
        self.rsi_buy = rsi_buy
        self.rsi_sell = rsi_sell

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series("HOLD", index=df.index)
        if "rsi" not in df.columns:
            return signals
        buy_mask = (df["rsi"] < self.rsi_buy)
        if "bb_pct" in df.columns:
            buy_mask = buy_mask & (df["bb_pct"] < 0.1)
        sell_mask = (df["rsi"] > self.rsi_sell)
        if "bb_pct" in df.columns:
            sell_mask = sell_mask & (df["bb_pct"] > 0.9)
        signals[buy_mask] = "BUY"
        signals[sell_mask] = "SELL"
        return signals


class MlSignalStrategy(Strategy):
    def __init__(self, predictions: list[dict]):
        self._predictions = predictions

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        if len(self._predictions) != len(df):
            return pd.Series("HOLD", index=df.index)
        return pd.Series([p["signal"] for p in self._predictions], index=df.index)


class CompositeStrategy(Strategy):
    SIGNAL_MAP = {"BUY": 1, "HOLD": 0, "SELL": -1}

    def __init__(self, strategies: list[Strategy], weights: list[float] = None):
        self.strategies = strategies
        self.weights = weights or [1.0 / len(strategies)] * len(strategies)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        combined = pd.Series(0.0, index=df.index)
        for strategy, weight in zip(self.strategies, self.weights):
            signals = strategy.generate_signals(df)
            numeric = signals.map(self.SIGNAL_MAP).fillna(0)
            combined += numeric * weight

        result = pd.Series("HOLD", index=df.index)
        result[combined > 0.3] = "BUY"
        result[combined < -0.3] = "SELL"
        return result


STRATEGIES = {
    "SMA Crossover": SmaCrossover,
    "RSI + Bollinger": RsiBollinger,
    "Composite": lambda: CompositeStrategy([SmaCrossover(), RsiBollinger()]),
}
