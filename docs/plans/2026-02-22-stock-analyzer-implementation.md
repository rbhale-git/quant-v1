# Stock Analyzer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a Python + Dash stock analysis tool with automated S&P 500 screening, ML predictions, Reddit sentiment, and backtesting.

**Architecture:** Single-process Python monolith with clear module boundaries. SQLite for storage, APScheduler for periodic jobs, Dash for interactive UI. All analysis in pandas/scikit-learn.

**Tech Stack:** Python 3.11+, Dash, Plotly, pandas, numpy, yfinance, scikit-learn, PRAW, VADER, APScheduler, pydantic-settings, SQLite, pytest

**Design doc:** `docs/plans/2026-02-22-stock-analyzer-design.md`

---

### Task 1: Project Scaffolding

**Files:**
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `.env.example`
- Create: `config.py`
- Create: `data/__init__.py`
- Create: `analysis/__init__.py`
- Create: `sentiment/__init__.py`
- Create: `backtest/__init__.py`
- Create: `trading/__init__.py`
- Create: `ui/__init__.py`
- Create: `tests/__init__.py`
- Test: `tests/test_config.py`

**Step 1: Create requirements.txt**

```txt
dash==2.18.2
plotly==5.24.1
pandas==2.2.3
numpy==1.26.4
yfinance==0.2.48
scikit-learn==1.5.2
joblib==1.4.2
praw==7.8.1
vaderSentiment==3.3.2
APScheduler==3.10.4
pydantic-settings==2.6.1
python-dotenv==1.0.1
pytest==8.3.4
pytest-cov==6.0.0
```

**Step 2: Create .gitignore**

```
__pycache__/
*.pyc
.env
*.db
models/
.pytest_cache/
*.egg-info/
dist/
build/
.venv/
venv/
```

**Step 3: Create .env.example**

```
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=stock-analyzer/1.0
```

**Step 4: Create config.py**

```python
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List


class Settings(BaseSettings):
    # Database
    db_path: str = "stock_analyzer.db"

    # Reddit API
    reddit_client_id: str = ""
    reddit_client_secret: str = ""
    reddit_user_agent: str = "stock-analyzer/1.0"
    reddit_subreddits: List[str] = ["wallstreetbets", "stocks", "investing"]

    # Screener
    min_avg_volume: int = 1_000_000
    atr_threshold: float = 1.0
    rsi_buy_threshold: float = 40.0
    rsi_sell_threshold: float = 60.0
    sma_trend_filter: bool = True

    # Indicators
    sma_windows: List[int] = [20, 50, 200]
    ema_windows: List[int] = [12, 26]
    bollinger_window: int = 20
    bollinger_std: float = 2.0
    rsi_window: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    atr_window: int = 14

    # ML
    ml_lookback_years: int = 2
    ml_validation_months: int = 3
    ml_buy_threshold: float = 0.03
    ml_sell_threshold: float = -0.03
    ml_forward_days: int = 7
    ml_model_dir: str = "models"

    # Regression
    regression_windows: List[int] = [30, 60, 90]

    # Backtest
    backtest_starting_capital: float = 10_000.0
    backtest_commission: float = 0.0
    backtest_slippage: float = 0.001

    # Scheduler
    screener_hour: int = 16
    screener_minute: int = 30
    retraining_day: str = "sun"
    retraining_hour: int = 20

    model_config = {"env_file": ".env", "env_prefix": "SA_"}


settings = Settings()
```

**Step 5: Create all __init__.py files**

Empty files for: `data/`, `analysis/`, `sentiment/`, `backtest/`, `trading/`, `ui/`, `tests/`

**Step 6: Write the config test**

```python
# tests/test_config.py
from config import Settings


def test_default_settings():
    s = Settings()
    assert s.db_path == "stock_analyzer.db"
    assert s.sma_windows == [20, 50, 200]
    assert s.ema_windows == [12, 26]
    assert s.rsi_window == 14
    assert s.backtest_starting_capital == 10_000.0
    assert s.ml_forward_days == 7


def test_custom_settings():
    s = Settings(db_path="custom.db", rsi_window=21)
    assert s.db_path == "custom.db"
    assert s.rsi_window == 21
```

**Step 7: Install dependencies and run test**

Run: `pip install -r requirements.txt`
Run: `cd C:/Users/ronak/Projects/stock-analyzer && python -m pytest tests/test_config.py -v`
Expected: 2 PASS

**Step 8: Commit**

```bash
git add -A
git commit -m "feat: project scaffolding with config, dependencies, and structure"
```

---

### Task 2: SQLite Storage Layer

**Files:**
- Create: `data/store.py`
- Test: `tests/data/__init__.py`
- Test: `tests/data/test_store.py`

**Step 1: Write the failing tests**

```python
# tests/data/test_store.py
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
```

**Step 2: Run tests to verify they fail**

Run: `cd C:/Users/ronak/Projects/stock-analyzer && python -m pytest tests/data/test_store.py -v`
Expected: FAIL (ModuleNotFoundError)

**Step 3: Implement Store**

```python
# data/store.py
import sqlite3
from datetime import date, datetime
from typing import Optional

import pandas as pd


class Store:
    def __init__(self, db_path: str = "stock_analyzer.db"):
        self.conn = sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()

    def _create_tables(self):
        cur = self.conn.cursor()
        cur.executescript("""
            CREATE TABLE IF NOT EXISTS daily_prices (
                date DATE NOT NULL,
                ticker TEXT NOT NULL,
                open REAL, high REAL, low REAL, close REAL,
                volume INTEGER,
                PRIMARY KEY (date, ticker)
            );
            CREATE TABLE IF NOT EXISTS hourly_prices (
                datetime TIMESTAMP NOT NULL,
                ticker TEXT NOT NULL,
                open REAL, high REAL, low REAL, close REAL,
                volume INTEGER,
                PRIMARY KEY (datetime, ticker)
            );
            CREATE TABLE IF NOT EXISTS watchlist (
                ticker TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS sentiment_scores (
                date DATE NOT NULL,
                ticker TEXT NOT NULL,
                sentiment_score REAL,
                mention_count INTEGER,
                sentiment_trend REAL,
                PRIMARY KEY (date, ticker)
            );
            CREATE TABLE IF NOT EXISTS signals (
                date DATE NOT NULL,
                ticker TEXT NOT NULL,
                signal TEXT,
                confidence REAL,
                ml_prediction TEXT,
                ml_confidence REAL,
                indicator_alignment REAL,
                sentiment_score REAL,
                PRIMARY KEY (date, ticker)
            );
            CREATE TABLE IF NOT EXISTS backtest_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                strategy TEXT,
                ticker TEXT,
                start_date DATE,
                end_date DATE,
                total_return REAL,
                annualized_return REAL,
                sharpe_ratio REAL,
                max_drawdown REAL,
                win_rate REAL,
                trade_count INTEGER,
                results_json TEXT
            );
            CREATE TABLE IF NOT EXISTS model_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trained_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                model_path TEXT,
                accuracy REAL,
                f1_score REAL,
                feature_importances TEXT
            );
        """)
        self.conn.commit()

    def list_tables(self) -> list[str]:
        cur = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        return [row[0] for row in cur.fetchall()]

    def save_daily_prices(self, df: pd.DataFrame):
        df.to_sql("daily_prices", self.conn, if_exists="append", index=False,
                   method=self._upsert_method("daily_prices"))
        self.conn.commit()

    def load_daily_prices(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        return pd.read_sql_query(
            "SELECT * FROM daily_prices WHERE ticker = ? AND date BETWEEN ? AND ? ORDER BY date",
            self.conn, params=(ticker, start, end),
            parse_dates=["date"],
        )

    def save_hourly_prices(self, df: pd.DataFrame):
        df.to_sql("hourly_prices", self.conn, if_exists="append", index=False,
                   method=self._upsert_method("hourly_prices"))
        self.conn.commit()

    def load_hourly_prices(self, ticker: str, start: datetime, end: datetime) -> pd.DataFrame:
        return pd.read_sql_query(
            "SELECT * FROM hourly_prices WHERE ticker = ? AND datetime BETWEEN ? AND ? ORDER BY datetime",
            self.conn, params=(ticker, start, end),
            parse_dates=["datetime"],
        )

    def add_to_watchlist(self, ticker: str, source: str = "manual"):
        self.conn.execute(
            "INSERT OR REPLACE INTO watchlist (ticker, source) VALUES (?, ?)",
            (ticker, source),
        )
        self.conn.commit()

    def remove_from_watchlist(self, ticker: str):
        self.conn.execute("DELETE FROM watchlist WHERE ticker = ?", (ticker,))
        self.conn.commit()

    def get_watchlist(self) -> pd.DataFrame:
        return pd.read_sql_query("SELECT * FROM watchlist ORDER BY ticker", self.conn)

    def save_signals(self, df: pd.DataFrame):
        df.to_sql("signals", self.conn, if_exists="append", index=False,
                   method=self._upsert_method("signals"))
        self.conn.commit()

    def load_signals(self, start: date, end: date, ticker: Optional[str] = None) -> pd.DataFrame:
        if ticker:
            return pd.read_sql_query(
                "SELECT * FROM signals WHERE ticker = ? AND date BETWEEN ? AND ? ORDER BY date",
                self.conn, params=(ticker, start, end),
            )
        return pd.read_sql_query(
            "SELECT * FROM signals WHERE date BETWEEN ? AND ? ORDER BY confidence DESC",
            self.conn, params=(start, end),
        )

    def save_sentiment_scores(self, df: pd.DataFrame):
        df.to_sql("sentiment_scores", self.conn, if_exists="append", index=False,
                   method=self._upsert_method("sentiment_scores"))
        self.conn.commit()

    def load_sentiment(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        return pd.read_sql_query(
            "SELECT * FROM sentiment_scores WHERE ticker = ? AND date BETWEEN ? AND ? ORDER BY date",
            self.conn, params=(ticker, start, end),
        )

    def save_backtest_result(self, result: dict):
        df = pd.DataFrame([result])
        df.to_sql("backtest_results", self.conn, if_exists="append", index=False)
        self.conn.commit()

    def save_model_run(self, run: dict):
        df = pd.DataFrame([run])
        df.to_sql("model_runs", self.conn, if_exists="append", index=False)
        self.conn.commit()

    @staticmethod
    def _upsert_method(table_name: str):
        def method(pd_table, conn, keys, data_iter):
            cols = [f'"{k}"' for k in keys]
            s_cols = ", ".join(cols)
            s_placeholders = ", ".join(["?"] * len(cols))
            sql = f"INSERT OR REPLACE INTO {table_name} ({s_cols}) VALUES ({s_placeholders})"
            data = [list(row) for row in data_iter]
            conn.executemany(sql, data)
        return method

    def close(self):
        self.conn.close()
```

**Step 4: Run tests to verify they pass**

Run: `cd C:/Users/ronak/Projects/stock-analyzer && python -m pytest tests/data/test_store.py -v`
Expected: 7 PASS

**Step 5: Commit**

```bash
git add data/store.py tests/data/
git commit -m "feat: SQLite storage layer with OHLCV, watchlist, signals, sentiment tables"
```

---

### Task 3: Data Fetcher (Yahoo Finance)

**Files:**
- Create: `data/fetcher.py`
- Test: `tests/data/test_fetcher.py`

**Step 1: Write the failing tests**

```python
# tests/data/test_fetcher.py
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
```

**Step 2: Run tests to verify they fail**

Run: `cd C:/Users/ronak/Projects/stock-analyzer && python -m pytest tests/data/test_fetcher.py -v`
Expected: FAIL

**Step 3: Implement Fetcher**

```python
# data/fetcher.py
import pandas as pd
import yfinance as yf


class Fetcher:
    def get_sp500_tickers(self) -> list[str]:
        table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
        tickers = table["Symbol"].str.replace(".", "-", regex=False).tolist()
        return sorted(tickers)

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
```

**Step 4: Run tests**

Run: `cd C:/Users/ronak/Projects/stock-analyzer && python -m pytest tests/data/test_fetcher.py -v`
Expected: 4 PASS (requires internet connection)

**Step 5: Commit**

```bash
git add data/fetcher.py tests/data/test_fetcher.py
git commit -m "feat: Yahoo Finance data fetcher with daily, hourly, and multi-ticker support"
```

---

### Task 4: Technical Indicators

**Files:**
- Create: `analysis/indicators.py`
- Test: `tests/analysis/__init__.py`
- Test: `tests/analysis/test_indicators.py`

**Step 1: Write the failing tests**

```python
# tests/analysis/test_indicators.py
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
```

**Step 2: Run tests to verify they fail**

Run: `cd C:/Users/ronak/Projects/stock-analyzer && python -m pytest tests/analysis/test_indicators.py -v`
Expected: FAIL

**Step 3: Implement indicators**

```python
# analysis/indicators.py
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
```

**Step 4: Run tests**

Run: `cd C:/Users/ronak/Projects/stock-analyzer && python -m pytest tests/analysis/test_indicators.py -v`
Expected: 8 PASS

**Step 5: Commit**

```bash
git add analysis/indicators.py tests/analysis/
git commit -m "feat: technical indicators - SMA, EMA, Bollinger, RSI, MACD, VWAP, ATR"
```

---

### Task 5: Linear Regression

**Files:**
- Create: `analysis/regression.py`
- Test: `tests/analysis/test_regression.py`

**Step 1: Write the failing tests**

```python
# tests/analysis/test_regression.py
import pytest
import pandas as pd
import numpy as np
from analysis.regression import compute_regression


@pytest.fixture
def trending_data():
    n = 100
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    close = 100 + np.arange(n) * 0.5 + np.random.randn(n) * 0.1
    return pd.DataFrame({"date": dates, "close": close})


@pytest.fixture
def flat_data():
    n = 100
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    close = 100 + np.random.randn(n) * 0.1
    return pd.DataFrame({"date": dates, "close": close})


def test_regression_uptrend(trending_data):
    result = compute_regression(trending_data, window=30)
    assert "reg_slope_30" in result.columns
    assert "reg_r2_30" in result.columns
    assert "reg_predicted_30" in result.columns
    last = result.iloc[-1]
    assert last["reg_slope_30"] > 0
    assert last["reg_r2_30"] > 0.8


def test_regression_flat(flat_data):
    result = compute_regression(flat_data, window=30)
    last = result.iloc[-1]
    assert abs(last["reg_slope_30"]) < 0.1


def test_regression_multiple_windows(trending_data):
    result = compute_regression(trending_data, window=30)
    result = compute_regression(result, window=60)
    assert "reg_slope_30" in result.columns
    assert "reg_slope_60" in result.columns
```

**Step 2: Run tests to verify they fail**

Run: `cd C:/Users/ronak/Projects/stock-analyzer && python -m pytest tests/analysis/test_regression.py -v`
Expected: FAIL

**Step 3: Implement regression**

```python
# analysis/regression.py
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def compute_regression(df: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    df = df.copy()
    slopes = []
    r2s = []
    predictions = []

    for i in range(len(df)):
        if i < window - 1:
            slopes.append(np.nan)
            r2s.append(np.nan)
            predictions.append(np.nan)
            continue
        segment = df["close"].iloc[i - window + 1: i + 1].values
        X = np.arange(window).reshape(-1, 1)
        y = segment
        model = LinearRegression()
        model.fit(X, y)
        slopes.append(model.coef_[0])
        r2s.append(model.score(X, y))
        next_X = np.array([[window]])
        predictions.append(model.predict(next_X)[0])

    df[f"reg_slope_{window}"] = slopes
    df[f"reg_r2_{window}"] = r2s
    df[f"reg_predicted_{window}"] = predictions
    return df
```

**Step 4: Run tests**

Run: `cd C:/Users/ronak/Projects/stock-analyzer && python -m pytest tests/analysis/test_regression.py -v`
Expected: 3 PASS

**Step 5: Commit**

```bash
git add analysis/regression.py tests/analysis/test_regression.py
git commit -m "feat: linear regression for price trend analysis"
```

---

### Task 6: ML Model (Random Forest Classifier)

**Files:**
- Create: `analysis/ml_models.py`
- Test: `tests/analysis/test_ml_models.py`

**Step 1: Write the failing tests**

```python
# tests/analysis/test_ml_models.py
import pytest
import os
import pandas as pd
import numpy as np
from analysis.ml_models import StockPredictor


@pytest.fixture
def sample_features():
    np.random.seed(42)
    n = 500
    df = pd.DataFrame({
        "date": pd.date_range("2022-01-01", periods=n, freq="B"),
        "close": 100 + np.cumsum(np.random.randn(n) * 0.5),
        "sma_20": np.random.randn(n),
        "sma_50": np.random.randn(n),
        "ema_12": np.random.randn(n),
        "ema_26": np.random.randn(n),
        "bb_pct": np.random.rand(n),
        "rsi": np.random.rand(n) * 100,
        "macd_histogram": np.random.randn(n),
        "vwap": 100 + np.random.randn(n),
        "atr": np.abs(np.random.randn(n)) + 0.5,
        "reg_slope_30": np.random.randn(n),
        "reg_r2_30": np.random.rand(n),
        "volume": np.random.randint(1_000_000, 10_000_000, n),
        "sentiment_score": np.random.randn(n) * 0.5,
    })
    return df


def test_prepare_features(sample_features):
    predictor = StockPredictor()
    X, y = predictor.prepare_features(sample_features, forward_days=7)
    assert len(X) < len(sample_features)
    assert len(X) == len(y)
    assert set(y.unique()).issubset({"BUY", "HOLD", "SELL"})


def test_train_and_predict(sample_features):
    predictor = StockPredictor()
    X, y = predictor.prepare_features(sample_features, forward_days=7)
    metrics = predictor.train(X, y)
    assert "accuracy" in metrics
    assert "f1_score" in metrics
    assert 0 <= metrics["accuracy"] <= 1

    preds = predictor.predict(X.iloc[-10:])
    assert len(preds) == 10
    assert all(p["signal"] in ["BUY", "HOLD", "SELL"] for p in preds)
    assert all(0 <= p["confidence"] <= 1 for p in preds)


def test_save_and_load(sample_features, tmp_path):
    predictor = StockPredictor()
    X, y = predictor.prepare_features(sample_features, forward_days=7)
    predictor.train(X, y)

    model_path = str(tmp_path / "model.joblib")
    predictor.save(model_path)
    assert os.path.exists(model_path)

    predictor2 = StockPredictor()
    predictor2.load(model_path)
    preds = predictor2.predict(X.iloc[-5:])
    assert len(preds) == 5


def test_feature_importances(sample_features):
    predictor = StockPredictor()
    X, y = predictor.prepare_features(sample_features, forward_days=7)
    predictor.train(X, y)
    importances = predictor.feature_importances()
    assert isinstance(importances, dict)
    assert len(importances) > 0
```

**Step 2: Run tests to verify they fail**

Run: `cd C:/Users/ronak/Projects/stock-analyzer && python -m pytest tests/analysis/test_ml_models.py -v`
Expected: FAIL

**Step 3: Implement StockPredictor**

```python
# analysis/ml_models.py
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from config import settings


FEATURE_COLUMNS = [
    "sma_20", "sma_50", "ema_12", "ema_26",
    "bb_pct", "rsi", "macd_histogram", "atr",
    "reg_slope_30", "reg_r2_30", "sentiment_score",
    "volume_change_pct", "sma_20_ratio", "sma_50_ratio", "vwap_ratio",
]


class StockPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_split=10,
            random_state=42, n_jobs=-1,
        )
        self._feature_cols = []

    def prepare_features(
        self, df: pd.DataFrame, forward_days: int = None,
        buy_threshold: float = None, sell_threshold: float = None,
    ) -> tuple[pd.DataFrame, pd.Series]:
        forward_days = forward_days or settings.ml_forward_days
        buy_threshold = buy_threshold or settings.ml_buy_threshold
        sell_threshold = sell_threshold or settings.ml_sell_threshold

        df = df.copy()

        # Derived features
        df["volume_change_pct"] = df["volume"].pct_change()
        df["sma_20_ratio"] = df["close"] / df["sma_20"] if "sma_20" in df.columns else 1.0
        df["sma_50_ratio"] = df["close"] / df["sma_50"] if "sma_50" in df.columns else 1.0
        df["vwap_ratio"] = df["close"] / df["vwap"] if "vwap" in df.columns else 1.0

        # Target: forward return
        df["forward_return"] = df["close"].shift(-forward_days) / df["close"] - 1

        def label(ret):
            if pd.isna(ret):
                return np.nan
            if ret > buy_threshold:
                return "BUY"
            elif ret < sell_threshold:
                return "SELL"
            return "HOLD"

        df["target"] = df["forward_return"].apply(label)

        # Select available feature columns
        self._feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]

        # Drop rows with NaN in features or target
        subset = df[self._feature_cols + ["target"]].dropna()
        X = subset[self._feature_cols]
        y = subset["target"]
        return X, y

    def train(self, X: pd.DataFrame, y: pd.Series) -> dict:
        self._feature_cols = list(X.columns)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False,
        )
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred, average="weighted"),
        }

    def predict(self, X: pd.DataFrame) -> list[dict]:
        probas = self.model.predict_proba(X)
        preds = self.model.predict(X)
        results = []
        for pred, proba in zip(preds, probas):
            confidence = float(max(proba))
            results.append({"signal": pred, "confidence": confidence})
        return results

    def feature_importances(self) -> dict:
        importances = self.model.feature_importances_
        return dict(zip(self._feature_cols, [float(x) for x in importances]))

    def save(self, path: str):
        joblib.dump({"model": self.model, "feature_cols": self._feature_cols}, path)

    def load(self, path: str):
        data = joblib.load(path)
        self.model = data["model"]
        self._feature_cols = data["feature_cols"]
```

**Step 4: Run tests**

Run: `cd C:/Users/ronak/Projects/stock-analyzer && python -m pytest tests/analysis/test_ml_models.py -v`
Expected: 4 PASS

**Step 5: Commit**

```bash
git add analysis/ml_models.py tests/analysis/test_ml_models.py
git commit -m "feat: Random Forest classifier for buy/hold/sell predictions"
```

---

### Task 7: Sentiment Analysis (Reddit + VADER)

**Files:**
- Create: `sentiment/reddit.py`
- Create: `sentiment/analyzer.py`
- Test: `tests/sentiment/__init__.py`
- Test: `tests/sentiment/test_analyzer.py`

**Step 1: Write the failing tests**

```python
# tests/sentiment/test_analyzer.py
import pytest
import pandas as pd
from datetime import date
from sentiment.analyzer import SentimentAnalyzer


@pytest.fixture
def analyzer():
    return SentimentAnalyzer()


def test_score_text_positive(analyzer):
    score = analyzer.score_text("AAPL is going to the moon! Great earnings!")
    assert score > 0


def test_score_text_negative(analyzer):
    score = analyzer.score_text("This stock is crashing. Terrible outlook. Sell now.")
    assert score < 0


def test_score_text_neutral(analyzer):
    score = analyzer.score_text("The stock closed at 150.")
    assert -0.3 <= score <= 0.3


def test_aggregate_scores(analyzer):
    posts = [
        {"text": "AAPL to the moon!", "upvotes": 100, "ticker": "AAPL"},
        {"text": "AAPL great earnings beat", "upvotes": 50, "ticker": "AAPL"},
        {"text": "AAPL might drop tomorrow", "upvotes": 30, "ticker": "AAPL"},
    ]
    result = analyzer.aggregate_sentiment(posts, "AAPL", date(2024, 1, 2))
    assert result["ticker"] == "AAPL"
    assert "sentiment_score" in result
    assert "mention_count" in result
    assert result["mention_count"] == 3
    assert -1 <= result["sentiment_score"] <= 1


def test_aggregate_empty(analyzer):
    result = analyzer.aggregate_sentiment([], "AAPL", date(2024, 1, 2))
    assert result["mention_count"] == 0
    assert result["sentiment_score"] == 0.0


def test_extract_tickers(analyzer):
    text = "I'm bullish on $AAPL and TSLA. MSFT is looking good too."
    tickers = analyzer.extract_tickers(text)
    assert "AAPL" in tickers
    assert "TSLA" in tickers
    assert "MSFT" in tickers
```

**Step 2: Run tests to verify they fail**

Run: `cd C:/Users/ronak/Projects/stock-analyzer && python -m pytest tests/sentiment/test_analyzer.py -v`
Expected: FAIL

**Step 3: Implement SentimentAnalyzer**

```python
# sentiment/analyzer.py
import re
from datetime import date
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class SentimentAnalyzer:
    # Common words that look like tickers but aren't
    TICKER_BLACKLIST = {
        "A", "I", "AM", "PM", "CEO", "CFO", "IPO", "ETF", "ATH",
        "DD", "FD", "GDP", "LLC", "USA", "USD", "SEC", "FDA",
        "IMO", "YOLO", "FOMO", "FYI", "TBH", "LOL", "OMG",
        "THE", "FOR", "AND", "BUT", "NOT", "ARE", "ALL", "HAS",
        "NOW", "NEW", "OLD", "BIG", "RUN", "PUT", "CALL",
    }

    def __init__(self):
        self._vader = SentimentIntensityAnalyzer()

    def score_text(self, text: str) -> float:
        scores = self._vader.polarity_scores(text)
        return scores["compound"]

    def extract_tickers(self, text: str) -> list[str]:
        cashtag = re.findall(r"\$([A-Z]{1,5})\b", text)
        allcaps = re.findall(r"\b([A-Z]{2,5})\b", text)
        combined = set(cashtag + allcaps) - self.TICKER_BLACKLIST
        return sorted(combined)

    def aggregate_sentiment(self, posts: list[dict], ticker: str, as_of: date) -> dict:
        if not posts:
            return {
                "date": as_of,
                "ticker": ticker,
                "sentiment_score": 0.0,
                "mention_count": 0,
                "sentiment_trend": 0.0,
            }
        scores = []
        weights = []
        for post in posts:
            score = self.score_text(post["text"])
            weight = max(post.get("upvotes", 1), 1)
            scores.append(score)
            weights.append(weight)

        total_weight = sum(weights)
        weighted_score = sum(s * w for s, w in zip(scores, weights)) / total_weight

        return {
            "date": as_of,
            "ticker": ticker,
            "sentiment_score": round(weighted_score, 4),
            "mention_count": len(posts),
            "sentiment_trend": 0.0,  # Computed when historical data available
        }
```

**Step 4: Implement Reddit scraper**

```python
# sentiment/reddit.py
import praw
from config import settings


class RedditScraper:
    def __init__(self):
        self.reddit = praw.Reddit(
            client_id=settings.reddit_client_id,
            client_secret=settings.reddit_client_secret,
            user_agent=settings.reddit_user_agent,
        )

    def fetch_posts(self, subreddits: list[str] = None, limit: int = 100) -> list[dict]:
        subreddits = subreddits or settings.reddit_subreddits
        posts = []
        for sub_name in subreddits:
            subreddit = self.reddit.subreddit(sub_name)
            for post in subreddit.hot(limit=limit):
                posts.append({
                    "title": post.title,
                    "text": f"{post.title} {post.selftext}",
                    "upvotes": post.score,
                    "subreddit": sub_name,
                    "created_utc": post.created_utc,
                    "num_comments": post.num_comments,
                })
                # Also grab top comments
                post.comments.replace_more(limit=0)
                for comment in post.comments[:5]:
                    posts.append({
                        "text": comment.body,
                        "upvotes": comment.score,
                        "subreddit": sub_name,
                        "created_utc": comment.created_utc,
                    })
        return posts

    def fetch_posts_for_ticker(self, ticker: str, subreddits: list[str] = None, limit: int = 50) -> list[dict]:
        subreddits = subreddits or settings.reddit_subreddits
        posts = []
        for sub_name in subreddits:
            subreddit = self.reddit.subreddit(sub_name)
            for post in subreddit.search(ticker, limit=limit, time_filter="day"):
                posts.append({
                    "text": f"{post.title} {post.selftext}",
                    "upvotes": post.score,
                    "ticker": ticker,
                    "subreddit": sub_name,
                    "created_utc": post.created_utc,
                })
        return posts
```

**Step 5: Run tests**

Run: `cd C:/Users/ronak/Projects/stock-analyzer && python -m pytest tests/sentiment/test_analyzer.py -v`
Expected: 6 PASS

**Step 6: Commit**

```bash
git add sentiment/ tests/sentiment/
git commit -m "feat: Reddit scraping with PRAW and VADER sentiment scoring"
```

---

### Task 8: Backtesting Engine

**Files:**
- Create: `backtest/engine.py`
- Create: `backtest/strategies.py`
- Test: `tests/backtest/__init__.py`
- Test: `tests/backtest/test_engine.py`
- Test: `tests/backtest/test_strategies.py`

**Step 1: Write the failing tests for strategies**

```python
# tests/backtest/test_strategies.py
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
```

**Step 2: Write the failing tests for engine**

```python
# tests/backtest/test_engine.py
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
```

**Step 3: Run tests to verify they fail**

Run: `cd C:/Users/ronak/Projects/stock-analyzer && python -m pytest tests/backtest/ -v`
Expected: FAIL

**Step 4: Implement strategies**

```python
# backtest/strategies.py
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
```

**Step 5: Implement backtest engine**

```python
# backtest/engine.py
import pandas as pd
import numpy as np
from backtest.strategies import Strategy
from config import settings


class BacktestEngine:
    def __init__(
        self,
        starting_capital: float = None,
        commission: float = None,
        slippage: float = None,
    ):
        self.starting_capital = starting_capital or settings.backtest_starting_capital
        self.commission = commission or settings.backtest_commission
        self.slippage = slippage or settings.backtest_slippage

    def run(self, df: pd.DataFrame, strategy: Strategy) -> dict:
        signals = strategy.generate_signals(df)
        cash = self.starting_capital
        shares = 0
        equity_curve = []
        trades = []
        entry_price = 0.0

        for i in range(len(df)):
            price = df["close"].iloc[i]
            signal = signals.iloc[i]
            date = df["date"].iloc[i] if "date" in df.columns else i

            if signal == "BUY" and shares == 0:
                exec_price = price * (1 + self.slippage)
                shares = int(cash / exec_price)
                if shares > 0:
                    cost = shares * exec_price + self.commission
                    cash -= cost
                    entry_price = exec_price
                    trades.append({
                        "date": date, "action": "BUY",
                        "price": exec_price, "shares": shares,
                    })

            elif signal == "SELL" and shares > 0:
                exec_price = price * (1 - self.slippage)
                revenue = shares * exec_price - self.commission
                cash += revenue
                trades.append({
                    "date": date, "action": "SELL",
                    "price": exec_price, "shares": shares,
                    "pnl": (exec_price - entry_price) * shares,
                })
                shares = 0

            portfolio_value = cash + shares * price
            equity_curve.append(portfolio_value)

        equity = pd.Series(equity_curve)
        metrics = self._compute_metrics(equity, trades)
        metrics["equity_curve"] = equity
        metrics["trades"] = trades
        return metrics

    def _compute_metrics(self, equity: pd.Series, trades: list[dict]) -> dict:
        total_return = (equity.iloc[-1] / equity.iloc[0]) - 1
        n_days = len(equity)
        annualized_return = (1 + total_return) ** (252 / max(n_days, 1)) - 1

        daily_returns = equity.pct_change().dropna()
        sharpe = 0.0
        if daily_returns.std() > 0:
            sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)

        running_max = equity.cummax()
        drawdown = (equity - running_max) / running_max
        max_drawdown = drawdown.min()

        sell_trades = [t for t in trades if t["action"] == "SELL"]
        wins = [t for t in sell_trades if t.get("pnl", 0) > 0]
        win_rate = len(wins) / max(len(sell_trades), 1)

        avg_win = np.mean([t["pnl"] for t in wins]) if wins else 0
        losses = [t for t in sell_trades if t.get("pnl", 0) <= 0]
        avg_loss = abs(np.mean([t["pnl"] for t in losses])) if losses else 0
        win_loss_ratio = avg_win / max(avg_loss, 0.01)

        return {
            "total_return": round(total_return, 4),
            "annualized_return": round(annualized_return, 4),
            "sharpe_ratio": round(sharpe, 4),
            "max_drawdown": round(max_drawdown, 4),
            "win_rate": round(win_rate, 4),
            "win_loss_ratio": round(win_loss_ratio, 4),
            "trade_count": len(sell_trades),
        }
```

**Step 6: Run tests**

Run: `cd C:/Users/ronak/Projects/stock-analyzer && python -m pytest tests/backtest/ -v`
Expected: 6 PASS

**Step 7: Commit**

```bash
git add backtest/ tests/backtest/
git commit -m "feat: backtesting engine with SMA, RSI+Bollinger, ML, and composite strategies"
```

---

### Task 9: Screener & Signal Generation

**Files:**
- Create: `data/screener.py`
- Test: `tests/data/test_screener.py`

**Step 1: Write the failing tests**

```python
# tests/data/test_screener.py
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
```

**Step 2: Run tests to verify they fail**

Run: `cd C:/Users/ronak/Projects/stock-analyzer && python -m pytest tests/data/test_screener.py -v`
Expected: FAIL

**Step 3: Implement Screener**

```python
# data/screener.py
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
```

**Step 4: Run tests**

Run: `cd C:/Users/ronak/Projects/stock-analyzer && python -m pytest tests/data/test_screener.py -v`
Expected: 4 PASS

**Step 5: Commit**

```bash
git add data/screener.py tests/data/test_screener.py
git commit -m "feat: stock screener with volume, ATR, RSI, and trend filters"
```

---

### Task 10: Scheduler

**Files:**
- Create: `scheduler.py`
- Test: `tests/test_scheduler.py`

**Step 1: Write the failing tests**

```python
# tests/test_scheduler.py
import pytest
from unittest.mock import patch, MagicMock
from scheduler import StockScheduler


def test_scheduler_creates_jobs():
    scheduler = StockScheduler(autostart=False)
    job_ids = [job.id for job in scheduler.scheduler.get_jobs()]
    assert "daily_screener" in job_ids
    assert "hourly_watchlist" in job_ids
    assert "weekly_retraining" in job_ids


def test_scheduler_does_not_start_when_autostart_false():
    scheduler = StockScheduler(autostart=False)
    assert not scheduler.scheduler.running
```

**Step 2: Run tests to verify they fail**

Run: `cd C:/Users/ronak/Projects/stock-analyzer && python -m pytest tests/test_scheduler.py -v`
Expected: FAIL

**Step 3: Implement scheduler**

```python
# scheduler.py
import logging
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from config import settings

logger = logging.getLogger(__name__)


class StockScheduler:
    def __init__(self, autostart: bool = True):
        self.scheduler = BackgroundScheduler()
        self._register_jobs()
        if autostart:
            self.scheduler.start()

    def _register_jobs(self):
        self.scheduler.add_job(
            self.run_daily_screener,
            CronTrigger(
                day_of_week="mon-fri",
                hour=settings.screener_hour,
                minute=settings.screener_minute,
                timezone="US/Eastern",
            ),
            id="daily_screener",
            replace_existing=True,
        )
        self.scheduler.add_job(
            self.run_hourly_watchlist_update,
            CronTrigger(
                day_of_week="mon-fri",
                hour="9-16",
                minute=30,
                timezone="US/Eastern",
            ),
            id="hourly_watchlist",
            replace_existing=True,
        )
        self.scheduler.add_job(
            self.run_weekly_retraining,
            CronTrigger(
                day_of_week=settings.retraining_day,
                hour=settings.retraining_hour,
                timezone="US/Eastern",
            ),
            id="weekly_retraining",
            replace_existing=True,
        )

    def run_daily_screener(self):
        logger.info("Running daily screener...")
        from data.fetcher import Fetcher
        from data.store import Store
        from data.screener import Screener
        from analysis.indicators import compute_all_indicators

        store = Store(settings.db_path)
        fetcher = Fetcher()
        screener = Screener()

        tickers = fetcher.get_sp500_tickers()
        for ticker in tickers:
            try:
                df = fetcher.fetch_daily(ticker, period="1y")
                if df.empty:
                    continue
                store.save_daily_prices(df)
                enriched = compute_all_indicators(df)
                # Screen happens on full universe after all data fetched
            except Exception as e:
                logger.error(f"Error fetching {ticker}: {e}")

        # Load all data and screen
        # Signal generation happens here
        logger.info("Daily screener complete.")
        store.close()

    def run_hourly_watchlist_update(self):
        logger.info("Running hourly watchlist update...")
        from data.fetcher import Fetcher
        from data.store import Store

        store = Store(settings.db_path)
        fetcher = Fetcher()
        watchlist = store.get_watchlist()

        for _, row in watchlist.iterrows():
            try:
                df = fetcher.fetch_hourly(row["ticker"], period="5d")
                if not df.empty:
                    store.save_hourly_prices(df)
            except Exception as e:
                logger.error(f"Error fetching hourly {row['ticker']}: {e}")

        logger.info("Hourly watchlist update complete.")
        store.close()

    def run_weekly_retraining(self):
        logger.info("Running weekly ML retraining...")
        # Full retraining pipeline will be wired in integration
        logger.info("Weekly retraining complete.")

    def shutdown(self):
        self.scheduler.shutdown()
```

**Step 4: Run tests**

Run: `cd C:/Users/ronak/Projects/stock-analyzer && python -m pytest tests/test_scheduler.py -v`
Expected: 2 PASS

**Step 5: Commit**

```bash
git add scheduler.py tests/test_scheduler.py
git commit -m "feat: APScheduler with daily screener, hourly watchlist, and weekly retraining jobs"
```

---

### Task 11: Dash UI — App Shell & Dashboard

**Files:**
- Create: `app.py`
- Create: `ui/layouts.py`
- Create: `ui/callbacks.py`

**Step 1: Create Dash app entry point**

```python
# app.py
import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

app = dash.Dash(
    __name__,
    use_pages=True,
    external_stylesheets=[dbc.themes.DARKLY],
    suppress_callback_exceptions=True,
)

sidebar = dbc.Nav(
    [
        dbc.NavLink("Dashboard", href="/", active="exact"),
        dbc.NavLink("Stock Detail", href="/stock", active="exact"),
        dbc.NavLink("Backtesting", href="/backtest", active="exact"),
        dbc.NavLink("Settings", href="/settings", active="exact"),
    ],
    vertical=True,
    pills=True,
    className="bg-dark",
)

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(
            html.Div([
                html.H4("Stock Analyzer", className="text-light p-3"),
                sidebar,
            ]),
            width=2,
            className="bg-dark vh-100 position-fixed",
        ),
        dbc.Col(
            dash.page_container,
            width=10,
            className="ms-auto p-4",
        ),
    ]),
], fluid=True, className="bg-dark text-light")

if __name__ == "__main__":
    app.run(debug=True)
```

**Step 2: Add dash-bootstrap-components to requirements.txt**

Add `dash-bootstrap-components==1.6.0` to `requirements.txt`.

**Step 3: Create page files**

```python
# ui/pages/__init__.py
# empty

# ui/pages/dashboard.py
import dash
from dash import html, dcc, dash_table, callback, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
from data.store import Store
from config import settings
from datetime import date, timedelta

dash.register_page(__name__, path="/", name="Dashboard")

layout = dbc.Container([
    html.H2("Dashboard"),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Today's Top Signals"),
                dbc.CardBody(id="signals-table"),
            ]),
        ], width=8),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Watchlist"),
                dbc.CardBody(id="watchlist-summary"),
            ]),
            dbc.Card([
                dbc.CardHeader("Last Scan"),
                dbc.CardBody(id="scan-status"),
            ], className="mt-3"),
        ], width=4),
    ]),
    dcc.Interval(id="dashboard-refresh", interval=60_000, n_intervals=0),
], fluid=True)


@callback(
    Output("signals-table", "children"),
    Input("dashboard-refresh", "n_intervals"),
)
def update_signals_table(n):
    try:
        store = Store(settings.db_path)
        today = date.today()
        signals = store.load_signals(today - timedelta(days=1), today)
        store.close()
        if signals.empty:
            return html.P("No signals yet. Run the screener first.")
        signals = signals.sort_values("confidence", ascending=False).head(10)
        return dash_table.DataTable(
            data=signals.to_dict("records"),
            columns=[{"name": c, "id": c} for c in signals.columns],
            style_table={"overflowX": "auto"},
            style_cell={"backgroundColor": "#303030", "color": "white"},
            style_header={"backgroundColor": "#404040", "fontWeight": "bold"},
        )
    except Exception:
        return html.P("No data available.")


@callback(
    Output("watchlist-summary", "children"),
    Input("dashboard-refresh", "n_intervals"),
)
def update_watchlist(n):
    try:
        store = Store(settings.db_path)
        wl = store.get_watchlist()
        store.close()
        if wl.empty:
            return html.P("Watchlist is empty.")
        return html.Ul([html.Li(t) for t in wl["ticker"]])
    except Exception:
        return html.P("No data available.")
```

```python
# ui/pages/stock_detail.py
import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import date, timedelta
from data.store import Store
from analysis.indicators import compute_all_indicators
from config import settings

dash.register_page(__name__, path="/stock", name="Stock Detail")

layout = dbc.Container([
    html.H2("Stock Detail"),
    dbc.Row([
        dbc.Col([
            dbc.Input(id="ticker-input", placeholder="Enter ticker (e.g. AAPL)", type="text"),
        ], width=3),
        dbc.Col([
            dbc.Button("Analyze", id="analyze-btn", color="primary"),
        ], width=2),
    ], className="mb-3"),
    dcc.Loading([
        html.Div(id="stock-chart"),
        html.Div(id="stock-indicators"),
        html.Div(id="stock-sentiment"),
    ]),
], fluid=True)


@callback(
    Output("stock-chart", "children"),
    Input("analyze-btn", "n_clicks"),
    State("ticker-input", "value"),
    prevent_initial_call=True,
)
def update_stock_chart(n_clicks, ticker):
    if not ticker:
        return html.P("Enter a ticker symbol.")
    ticker = ticker.upper().strip()
    try:
        store = Store(settings.db_path)
        end = date.today()
        start = end - timedelta(days=365)
        df = store.load_daily_prices(ticker, start, end)
        store.close()

        if df.empty:
            return html.P(f"No data for {ticker}. Fetch data first.")

        df = compute_all_indicators(df)

        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True, row_heights=[0.6, 0.2, 0.2],
            vertical_spacing=0.03,
        )

        fig.add_trace(go.Candlestick(
            x=df["date"], open=df["open"], high=df["high"],
            low=df["low"], close=df["close"], name="Price",
        ), row=1, col=1)

        if "sma_20" in df.columns:
            fig.add_trace(go.Scatter(x=df["date"], y=df["sma_20"], name="SMA 20", line=dict(width=1)), row=1, col=1)
        if "sma_50" in df.columns:
            fig.add_trace(go.Scatter(x=df["date"], y=df["sma_50"], name="SMA 50", line=dict(width=1)), row=1, col=1)
        if "bb_upper" in df.columns:
            fig.add_trace(go.Scatter(x=df["date"], y=df["bb_upper"], name="BB Upper", line=dict(width=1, dash="dot")), row=1, col=1)
            fig.add_trace(go.Scatter(x=df["date"], y=df["bb_lower"], name="BB Lower", line=dict(width=1, dash="dot")), row=1, col=1)

        if "rsi" in df.columns:
            fig.add_trace(go.Scatter(x=df["date"], y=df["rsi"], name="RSI"), row=2, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

        if "macd" in df.columns:
            fig.add_trace(go.Scatter(x=df["date"], y=df["macd"], name="MACD"), row=3, col=1)
            fig.add_trace(go.Scatter(x=df["date"], y=df["macd_signal"], name="Signal"), row=3, col=1)
            fig.add_trace(go.Bar(x=df["date"], y=df["macd_histogram"], name="Histogram"), row=3, col=1)

        fig.update_layout(
            template="plotly_dark", height=800,
            xaxis_rangeslider_visible=False,
            title=f"{ticker} Analysis",
        )
        return dcc.Graph(figure=fig)
    except Exception as e:
        return html.P(f"Error: {e}")
```

```python
# ui/pages/backtesting.py
import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import date, timedelta
from data.store import Store
from analysis.indicators import compute_all_indicators
from backtest.engine import BacktestEngine
from backtest.strategies import SmaCrossover, RsiBollinger, CompositeStrategy
from config import settings

dash.register_page(__name__, path="/backtest", name="Backtesting")

STRATEGIES = {
    "SMA Crossover": SmaCrossover,
    "RSI + Bollinger": RsiBollinger,
    "Composite": lambda: CompositeStrategy([SmaCrossover(), RsiBollinger()]),
}

layout = dbc.Container([
    html.H2("Backtesting"),
    dbc.Row([
        dbc.Col([
            dbc.Label("Ticker"),
            dbc.Input(id="bt-ticker", placeholder="AAPL", type="text"),
        ], width=2),
        dbc.Col([
            dbc.Label("Strategy"),
            dcc.Dropdown(
                id="bt-strategy",
                options=[{"label": k, "value": k} for k in STRATEGIES],
                value="SMA Crossover",
            ),
        ], width=3),
        dbc.Col([
            dbc.Label("Start Date"),
            dcc.DatePickerSingle(id="bt-start", date=date.today() - timedelta(days=365)),
        ], width=2),
        dbc.Col([
            dbc.Label("End Date"),
            dcc.DatePickerSingle(id="bt-end", date=date.today()),
        ], width=2),
        dbc.Col([
            dbc.Label("\u00a0"),
            html.Br(),
            dbc.Button("Run Backtest", id="bt-run", color="primary"),
        ], width=2),
    ], className="mb-3"),
    dcc.Loading([
        html.Div(id="bt-results"),
        html.Div(id="bt-chart"),
    ]),
], fluid=True)


@callback(
    [Output("bt-results", "children"), Output("bt-chart", "children")],
    Input("bt-run", "n_clicks"),
    [State("bt-ticker", "value"), State("bt-strategy", "value"),
     State("bt-start", "date"), State("bt-end", "date")],
    prevent_initial_call=True,
)
def run_backtest(n_clicks, ticker, strategy_name, start_date, end_date):
    if not ticker:
        return html.P("Enter a ticker."), ""
    ticker = ticker.upper().strip()

    try:
        store = Store(settings.db_path)
        df = store.load_daily_prices(ticker, start_date, end_date)
        store.close()

        if df.empty:
            return html.P(f"No data for {ticker}."), ""

        df = compute_all_indicators(df)
        strategy_cls = STRATEGIES.get(strategy_name, SmaCrossover)
        strategy = strategy_cls() if callable(strategy_cls) else strategy_cls

        engine = BacktestEngine()
        result = engine.run(df, strategy)

        # Metrics card
        metrics_card = dbc.Card([
            dbc.CardHeader("Performance Metrics"),
            dbc.CardBody([
                html.P(f"Total Return: {result['total_return']:.2%}"),
                html.P(f"Annualized Return: {result['annualized_return']:.2%}"),
                html.P(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}"),
                html.P(f"Max Drawdown: {result['max_drawdown']:.2%}"),
                html.P(f"Win Rate: {result['win_rate']:.2%}"),
                html.P(f"Trade Count: {result['trade_count']}"),
            ]),
        ])

        # Equity curve chart
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.5, 0.5])
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["close"], name="Price",
        ), row=1, col=1)

        # Buy/sell markers
        for trade in result["trades"]:
            color = "green" if trade["action"] == "BUY" else "red"
            marker = "triangle-up" if trade["action"] == "BUY" else "triangle-down"
            fig.add_trace(go.Scatter(
                x=[trade["date"]], y=[trade["price"]],
                mode="markers", name=trade["action"],
                marker=dict(color=color, size=12, symbol=marker),
                showlegend=False,
            ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=df["date"], y=result["equity_curve"], name="Portfolio",
        ), row=2, col=1)
        fig.update_layout(template="plotly_dark", height=600, title=f"Backtest: {strategy_name} on {ticker}")

        return metrics_card, dcc.Graph(figure=fig)
    except Exception as e:
        return html.P(f"Error: {e}"), ""
```

```python
# ui/pages/settings_page.py
import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
from data.store import Store
from config import settings

dash.register_page(__name__, path="/settings", name="Settings")

layout = dbc.Container([
    html.H2("Settings"),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Screener Thresholds"),
                dbc.CardBody([
                    dbc.Label("Min Avg Volume"),
                    dbc.Input(id="set-min-volume", type="number", value=settings.min_avg_volume),
                    dbc.Label("ATR Threshold"),
                    dbc.Input(id="set-atr", type="number", value=settings.atr_threshold, step=0.1),
                    dbc.Label("RSI Buy Threshold"),
                    dbc.Input(id="set-rsi-buy", type="number", value=settings.rsi_buy_threshold),
                    dbc.Label("RSI Sell Threshold"),
                    dbc.Input(id="set-rsi-sell", type="number", value=settings.rsi_sell_threshold),
                ]),
            ]),
        ], width=4),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Watchlist Management"),
                dbc.CardBody([
                    dbc.Input(id="wl-ticker-input", placeholder="Add ticker..."),
                    dbc.Button("Add", id="wl-add-btn", color="success", className="mt-2"),
                    html.Hr(),
                    html.Div(id="wl-current-list"),
                ]),
            ]),
        ], width=4),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("ML Model Status"),
                dbc.CardBody(id="ml-status"),
            ]),
        ], width=4),
    ]),
], fluid=True)


@callback(
    Output("wl-current-list", "children"),
    Input("wl-add-btn", "n_clicks"),
    State("wl-ticker-input", "value"),
    prevent_initial_call=False,
)
def update_watchlist(n_clicks, ticker):
    store = Store(settings.db_path)
    if n_clicks and ticker:
        store.add_to_watchlist(ticker.upper().strip(), source="manual")
    wl = store.get_watchlist()
    store.close()
    if wl.empty:
        return html.P("No stocks in watchlist.")
    return html.Ul([html.Li(t) for t in wl["ticker"]])
```

**Step 4: Move page files into proper Dash pages directory**

The files above should be placed in `ui/pages/` directory. Also update `app.py` to point `pages_folder` to `ui/pages`.

Update `app.py`:
```python
app = dash.Dash(
    __name__,
    use_pages=True,
    pages_folder="ui/pages",
    external_stylesheets=[dbc.themes.DARKLY],
    suppress_callback_exceptions=True,
)
```

**Step 5: Install new dependency and test manually**

Run: `pip install dash-bootstrap-components`
Run: `cd C:/Users/ronak/Projects/stock-analyzer && python app.py`
Expected: Dash app starts at http://127.0.0.1:8050

**Step 6: Commit**

```bash
git add app.py ui/ requirements.txt
git commit -m "feat: Dash UI with dashboard, stock detail, backtesting, and settings pages"
```

---

### Task 12: Integration — Wire Everything Together

**Files:**
- Modify: `scheduler.py` — complete the daily screener pipeline
- Create: `run_pipeline.py` — manual pipeline runner for development

**Step 1: Create manual pipeline runner**

```python
# run_pipeline.py
"""Run the full analysis pipeline manually for development/testing."""
import sys
import logging
from datetime import date, timedelta
from data.fetcher import Fetcher
from data.store import Store
from data.screener import Screener
from analysis.indicators import compute_all_indicators
from analysis.regression import compute_regression
from analysis.ml_models import StockPredictor
from config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run_fetch(tickers: list[str] = None, period: str = "1y"):
    fetcher = Fetcher()
    store = Store(settings.db_path)

    if tickers is None:
        logger.info("Fetching S&P 500 ticker list...")
        tickers = fetcher.get_sp500_tickers()

    logger.info(f"Fetching daily data for {len(tickers)} tickers...")
    for i, ticker in enumerate(tickers):
        try:
            df = fetcher.fetch_daily(ticker, period=period)
            if not df.empty:
                store.save_daily_prices(df)
            if (i + 1) % 50 == 0:
                logger.info(f"  Fetched {i + 1}/{len(tickers)}")
        except Exception as e:
            logger.error(f"  Error fetching {ticker}: {e}")

    logger.info("Data fetch complete.")
    store.close()


def run_screen():
    store = Store(settings.db_path)
    fetcher = Fetcher()
    screener = Screener()

    tickers = fetcher.get_sp500_tickers()
    end = date.today()
    start = end - timedelta(days=365)

    logger.info("Loading and enriching data for screening...")
    import pandas as pd
    frames = []
    for ticker in tickers:
        df = store.load_daily_prices(ticker, start, end)
        if df.empty or len(df) < 200:
            continue
        df = compute_all_indicators(df)
        df["ticker"] = ticker
        frames.append(df)

    if not frames:
        logger.warning("No data to screen.")
        store.close()
        return

    universe = pd.concat(frames, ignore_index=True)
    passed = screener.screen(universe)
    passed_tickers = passed["ticker"].unique()
    logger.info(f"Screener passed {len(passed_tickers)} stocks: {list(passed_tickers)[:20]}...")

    for ticker in passed_tickers:
        store.add_to_watchlist(ticker, source="screener")

    logger.info("Screening complete.")
    store.close()


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "fetch"
    if cmd == "fetch":
        tickers = sys.argv[2:] if len(sys.argv) > 2 else None
        run_fetch(tickers)
    elif cmd == "screen":
        run_screen()
    else:
        print(f"Usage: python run_pipeline.py [fetch|screen] [tickers...]")
```

**Step 2: Test the pipeline manually**

Run: `cd C:/Users/ronak/Projects/stock-analyzer && python run_pipeline.py fetch AAPL MSFT TSLA`
Expected: Fetches and stores data for 3 tickers

Run: `cd C:/Users/ronak/Projects/stock-analyzer && python run_pipeline.py screen`
Expected: Runs screener (may not find results with only 3 stocks)

**Step 3: Commit**

```bash
git add run_pipeline.py
git commit -m "feat: manual pipeline runner for fetch and screen operations"
```

---

### Task 13: Full Test Suite & Final Verification

**Step 1: Run the complete test suite**

Run: `cd C:/Users/ronak/Projects/stock-analyzer && python -m pytest tests/ -v --tb=short`
Expected: All tests pass

**Step 2: Run the Dash app and verify all pages load**

Run: `cd C:/Users/ronak/Projects/stock-analyzer && python app.py`
Verify:
- Dashboard page loads at `/`
- Stock Detail page loads at `/stock`
- Backtesting page loads at `/backtest`
- Settings page loads at `/settings`

**Step 3: Final commit**

```bash
git add -A
git commit -m "chore: final verification — all tests passing, all pages loading"
```

---

## Task Dependency Map

```
Task 1 (Scaffolding)
├── Task 2 (Store) ──────────────┐
├── Task 3 (Fetcher) ────────────┤
├── Task 4 (Indicators) ─────────┤
│   └── Task 5 (Regression) ─────┤
│       └── Task 6 (ML Models) ──┤
├── Task 7 (Sentiment) ──────────┤
├── Task 8 (Backtesting) ────────┤
├── Task 9 (Screener) ───────────┤
├── Task 10 (Scheduler) ─────────┤
├── Task 11 (Dash UI) ───────────┘
│       All above ─────→ Task 12 (Integration)
│                             └──→ Task 13 (Verification)
```

Tasks 2-10 can be parallelized after Task 1. Task 11 depends on 2, 4, 8. Task 12 depends on all. Task 13 is final.
