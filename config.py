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
    atr_threshold: float = 0.5
    rsi_buy_threshold: float = 45.0
    rsi_sell_threshold: float = 55.0
    sma_trend_filter: bool = False

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
