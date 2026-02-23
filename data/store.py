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
