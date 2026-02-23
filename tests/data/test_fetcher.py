import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from data.fetcher import Fetcher


@pytest.fixture
def fetcher():
    return Fetcher()


def test_get_sp500_tickers(fetcher):
    tickers = fetcher.get_sp500_tickers()
    assert isinstance(tickers, list)
    assert len(tickers) > 400
    assert "AAPL" in tickers
    assert "MSFT" in tickers


def test_fetch_daily_returns_dataframe(fetcher):
    df = fetcher.fetch_daily("AAPL", period="5d")
    assert isinstance(df, pd.DataFrame)
    assert "ticker" in df.columns
    assert "open" in df.columns
    assert "high" in df.columns
    assert "low" in df.columns
    assert "close" in df.columns
    assert "volume" in df.columns
    assert "date" in df.columns
    assert len(df) > 0
    assert df.iloc[0]["ticker"] == "AAPL"


def test_fetch_hourly_returns_dataframe(fetcher):
    df = fetcher.fetch_hourly("AAPL", period="5d")
    assert isinstance(df, pd.DataFrame)
    assert "ticker" in df.columns
    assert "datetime" in df.columns
    assert "close" in df.columns
    assert len(df) > 0


def test_fetch_daily_multiple_tickers(fetcher):
    df = fetcher.fetch_daily_multiple(["AAPL", "MSFT"], period="5d")
    assert isinstance(df, pd.DataFrame)
    tickers = df["ticker"].unique()
    assert "AAPL" in tickers
    assert "MSFT" in tickers
