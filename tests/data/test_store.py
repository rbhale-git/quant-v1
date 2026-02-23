import os
import pytest
import pandas as pd
from datetime import date, datetime
from data.store import Store


@pytest.fixture
def store(tmp_path):
    db_path = str(tmp_path / "test.db")
    s = Store(db_path)
    yield s
    s.close()


def test_init_creates_tables(store):
    tables = store.list_tables()
    assert "daily_prices" in tables
    assert "hourly_prices" in tables
    assert "watchlist" in tables
    assert "sentiment_scores" in tables
    assert "signals" in tables
    assert "backtest_results" in tables
    assert "model_runs" in tables


def test_save_and_load_daily_prices(store):
    df = pd.DataFrame({
        "date": [date(2024, 1, 2), date(2024, 1, 3)],
        "ticker": ["AAPL", "AAPL"],
        "open": [185.0, 186.0],
        "high": [187.0, 188.0],
        "low": [184.0, 185.0],
        "close": [186.5, 187.5],
        "volume": [50000000, 48000000],
    })
    store.save_daily_prices(df)
    result = store.load_daily_prices("AAPL", date(2024, 1, 1), date(2024, 1, 5))
    assert len(result) == 2
    assert result.iloc[0]["close"] == 186.5


def test_save_daily_prices_upsert(store):
    df1 = pd.DataFrame({
        "date": [date(2024, 1, 2)],
        "ticker": ["AAPL"],
        "open": [185.0], "high": [187.0], "low": [184.0],
        "close": [186.5], "volume": [50000000],
    })
    store.save_daily_prices(df1)
    df2 = pd.DataFrame({
        "date": [date(2024, 1, 2)],
        "ticker": ["AAPL"],
        "open": [185.0], "high": [187.0], "low": [184.0],
        "close": [190.0], "volume": [55000000],
    })
    store.save_daily_prices(df2)
    result = store.load_daily_prices("AAPL", date(2024, 1, 1), date(2024, 1, 5))
    assert len(result) == 1
    assert result.iloc[0]["close"] == 190.0


def test_save_and_load_hourly_prices(store):
    df = pd.DataFrame({
        "datetime": [datetime(2024, 1, 2, 10, 0), datetime(2024, 1, 2, 11, 0)],
        "ticker": ["AAPL", "AAPL"],
        "open": [185.0, 186.0],
        "high": [187.0, 188.0],
        "low": [184.0, 185.0],
        "close": [186.5, 187.5],
        "volume": [5000000, 4800000],
    })
    store.save_hourly_prices(df)
    result = store.load_hourly_prices("AAPL", datetime(2024, 1, 1), datetime(2024, 1, 3))
    assert len(result) == 2


def test_watchlist_add_remove(store):
    store.add_to_watchlist("AAPL", source="screener")
    store.add_to_watchlist("TSLA", source="manual")
    wl = store.get_watchlist()
    assert set(wl["ticker"]) == {"AAPL", "TSLA"}
    store.remove_from_watchlist("AAPL")
    wl = store.get_watchlist()
    assert list(wl["ticker"]) == ["TSLA"]


def test_save_and_load_signals(store):
    df = pd.DataFrame({
        "date": [date(2024, 1, 2)],
        "ticker": ["AAPL"],
        "signal": ["BUY"],
        "confidence": [85.5],
        "ml_prediction": ["BUY"],
        "ml_confidence": [0.78],
        "indicator_alignment": [0.8],
        "sentiment_score": [0.45],
    })
    store.save_signals(df)
    result = store.load_signals(date(2024, 1, 1), date(2024, 1, 5))
    assert len(result) == 1
    assert result.iloc[0]["signal"] == "BUY"


def test_save_and_load_sentiment(store):
    df = pd.DataFrame({
        "date": [date(2024, 1, 2)],
        "ticker": ["AAPL"],
        "sentiment_score": [0.65],
        "mention_count": [150],
        "sentiment_trend": [0.1],
    })
    store.save_sentiment_scores(df)
    result = store.load_sentiment("AAPL", date(2024, 1, 1), date(2024, 1, 5))
    assert len(result) == 1
    assert result.iloc[0]["sentiment_score"] == 0.65
