import pandas as pd
import numpy as np
from config import settings


class Screener:
    def __init__(
        self,
        min_avg_volume: int = None,
        atr_threshold: float = None,
        rsi_buy_threshold: float = None,
        rsi_sell_threshold: float = None,
        sma_trend_filter: bool = None,
    ):
        self.min_avg_volume = min_avg_volume if min_avg_volume is not None else settings.min_avg_volume
        self.atr_threshold = atr_threshold if atr_threshold is not None else settings.atr_threshold
        self.rsi_buy_threshold = rsi_buy_threshold or settings.rsi_buy_threshold
        self.rsi_sell_threshold = rsi_sell_threshold or settings.rsi_sell_threshold
        self.sma_trend_filter = sma_trend_filter if sma_trend_filter is not None else settings.sma_trend_filter

    def screen(self, df: pd.DataFrame) -> pd.DataFrame:
        latest = df.groupby("ticker").tail(20)

        # Volume filter
        avg_volume = latest.groupby("ticker")["volume"].mean()
        passing_volume = avg_volume[avg_volume >= self.min_avg_volume].index

        # ATR filter
        latest_atr = df.groupby("ticker").last()
        if "atr" in latest_atr.columns:
            passing_atr = latest_atr[latest_atr["atr"] >= self.atr_threshold].index
        else:
            passing_atr = latest_atr.index

        # RSI filter: in actionable range
        if "rsi" in latest_atr.columns:
            passing_rsi = latest_atr[
                (latest_atr["rsi"] <= self.rsi_buy_threshold) |
                (latest_atr["rsi"] >= self.rsi_sell_threshold)
            ].index
        else:
            passing_rsi = latest_atr.index

        # SMA trend filter
        if self.sma_trend_filter and "sma_200" in latest_atr.columns:
            passing_trend = latest_atr[
                latest_atr["close"] > latest_atr["sma_200"]
            ].index.dropna()
        else:
            passing_trend = latest_atr.index

        passing = set(passing_volume) & set(passing_atr) & set(passing_rsi) & set(passing_trend)
        return df[df["ticker"].isin(passing)]

    def compute_composite_signal(
        self,
        ml_prediction: str,
        ml_confidence: float,
        indicator_alignment: float,
        sentiment_score: float,
        ml_weight: float = 0.5,
        indicator_weight: float = 0.3,
        sentiment_weight: float = 0.2,
    ) -> dict:
        signal_map = {"BUY": 1, "HOLD": 0, "SELL": -1}
        ml_numeric = signal_map.get(ml_prediction, 0) * ml_confidence

        composite = (
            ml_numeric * ml_weight +
            indicator_alignment * indicator_weight +
            sentiment_score * sentiment_weight
        )

        if composite > 0.2:
            signal = "BUY"
        elif composite < -0.2:
            signal = "SELL"
        else:
            signal = "HOLD"

        confidence = min(abs(composite) * 100, 100)
        return {"signal": signal, "confidence": round(confidence, 1)}
