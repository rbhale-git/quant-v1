import io
import urllib.request

import pandas as pd
import yfinance as yf


class Fetcher:
    INDEX_SOURCES = {
        "S&P 500": {
            "url": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            "table_index": 0,
            "column": "Symbol",
        },
        "NASDAQ-100": {
            "url": "https://en.wikipedia.org/wiki/Nasdaq-100",
            "table_index": 4,
            "column": "Ticker",
        },
        "Dow 30": {
            "url": "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average",
            "table_index": 2,
            "column": "Symbol",
        },
    }

    def get_index_tickers(self, index_name: str = "S&P 500") -> list[str]:
        source = self.INDEX_SOURCES[index_name]
        req = urllib.request.Request(source["url"], headers={"User-Agent": "stock-analyzer/1.0"})
        with urllib.request.urlopen(req) as resp:
            html = resp.read().decode("utf-8")
        table = pd.read_html(io.StringIO(html))[source["table_index"]]
        tickers = table[source["column"]].str.replace(".", "-", regex=False).tolist()
        return sorted(tickers)

    def get_sp500_tickers(self) -> list[str]:
        return self.get_index_tickers("S&P 500")

    def fetch_daily(self, ticker: str, period: str = "2y", start: str = None, end: str = None) -> pd.DataFrame:
        t = yf.Ticker(ticker)
        if start and end:
            hist = t.history(start=start, end=end, interval="1d")
        else:
            hist = t.history(period=period, interval="1d")
        if hist.empty:
            return pd.DataFrame()
        df = hist.reset_index()
        df = df.rename(columns={
            "Date": "date", "Open": "open", "High": "high",
            "Low": "low", "Close": "close", "Volume": "volume",
        })
        df["ticker"] = ticker
        df["date"] = pd.to_datetime(df["date"]).dt.date
        return df[["date", "ticker", "open", "high", "low", "close", "volume"]]

    def fetch_hourly(self, ticker: str, period: str = "30d") -> pd.DataFrame:
        t = yf.Ticker(ticker)
        hist = t.history(period=period, interval="1h")
        if hist.empty:
            return pd.DataFrame()
        df = hist.reset_index()
        df = df.rename(columns={
            "Datetime": "datetime", "Open": "open", "High": "high",
            "Low": "low", "Close": "close", "Volume": "volume",
        })
        if "datetime" not in df.columns and "Date" in df.columns:
            df = df.rename(columns={"Date": "datetime"})
        df["ticker"] = ticker
        df["datetime"] = pd.to_datetime(df["datetime"])
        return df[["datetime", "ticker", "open", "high", "low", "close", "volume"]]

    def fetch_daily_multiple(self, tickers: list[str], period: str = "2y") -> pd.DataFrame:
        frames = []
        for ticker in tickers:
            df = self.fetch_daily(ticker, period=period)
            if not df.empty:
                frames.append(df)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)
