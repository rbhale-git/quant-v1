import pandas as pd
import numpy as np
from config import settings


def compute_sma(df: pd.DataFrame, window: int) -> pd.DataFrame:
    df = df.copy()
    df[f"sma_{window}"] = df["close"].rolling(window=window).mean()
    return df


def compute_ema(df: pd.DataFrame, window: int) -> pd.DataFrame:
    df = df.copy()
    df[f"ema_{window}"] = df["close"].ewm(span=window, adjust=False).mean()
    return df


def compute_bollinger_bands(df: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    df = df.copy()
    df["bb_middle"] = df["close"].rolling(window=window).mean()
    rolling_std = df["close"].rolling(window=window).std()
    df["bb_upper"] = df["bb_middle"] + num_std * rolling_std
    df["bb_lower"] = df["bb_middle"] - num_std * rolling_std
    df["bb_pct"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
    return df


def compute_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    df = df.copy()
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    df["rsi"] = 100 - (100 / (1 + rs))
    return df


def compute_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    df = df.copy()
    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
    df["macd"] = ema_fast - ema_slow
    df["macd_signal"] = df["macd"].ewm(span=signal, adjust=False).mean()
    df["macd_histogram"] = df["macd"] - df["macd_signal"]
    return df


def compute_vwap(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    cumulative_tp_vol = (typical_price * df["volume"]).cumsum()
    cumulative_vol = df["volume"].cumsum()
    df["vwap"] = cumulative_tp_vol / cumulative_vol
    return df


def compute_atr(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    df = df.copy()
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = true_range.rolling(window=window).mean()
    return df


def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for w in settings.sma_windows:
        result = compute_sma(result, w)
    for w in settings.ema_windows:
        result = compute_ema(result, w)
    result = compute_bollinger_bands(result, settings.bollinger_window, settings.bollinger_std)
    result = compute_rsi(result, settings.rsi_window)
    result = compute_macd(result, settings.macd_fast, settings.macd_slow, settings.macd_signal)
    result = compute_vwap(result)
    result = compute_atr(result, settings.atr_window)
    return result
