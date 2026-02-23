# Stock Analysis & Trading Tool — Design Document

**Date:** 2026-02-22
**Status:** Approved

## Overview

An analytical tool that imports historical stock data, runs statistical and ML analyses, scrubs Reddit for sentiment, and provides buy/sell recommendations. Includes a backtesting platform. Robinhood API integration planned for Phase 2.

## Decisions

| Decision | Choice |
|----------|--------|
| Tech stack | Python + Dash |
| Architecture | Monolith with clear module boundaries |
| Build order | Analysis/backtesting first, trading later |
| Deployment | Local first, cloud later |
| Data granularity | Daily OHLCV for S&P 500, hourly for watchlist |
| Storage | SQLite |
| ML approach | Random Forest, weekly periodic retraining |
| Sentiment source | Reddit (PRAW + VADER) |

## Project Structure

```
stock-analyzer/
├── app.py                  # Dash app entry point
├── config.py               # Settings, API keys, schedules
├── data/
│   ├── fetcher.py          # Yahoo Finance data pull
│   ├── store.py            # SQLite storage
│   └── screener.py         # S&P 500 universe scanning
├── analysis/
│   ├── indicators.py       # SMA, EMA, Bollinger, RSI, MACD, VWAP, ATR
│   ├── regression.py       # Linear regression
│   └── ml_models.py        # Scikit-learn Random Forest + training
├── sentiment/
│   ├── reddit.py           # PRAW scraping
│   └── analyzer.py         # VADER sentiment scoring
├── backtest/
│   ├── engine.py           # Backtest runner
│   └── strategies.py       # Strategy definitions
├── trading/                # Phase 2: Robinhood
│   └── executor.py
├── ui/
│   ├── layouts.py          # Dash page layouts
│   └── callbacks.py        # Dash interactivity
├── scheduler.py            # APScheduler for periodic tasks
├── tests/                  # pytest test suite
├── requirements.txt
├── .env.example
└── .gitignore
```

## Data Layer

### Fetching
- `yfinance` library for OHLCV data
- S&P 500 ticker list from Wikipedia or maintained CSV
- Configurable date ranges and intervals

### Granularity
- **Full S&P 500:** Daily OHLCV, fetched after market close
- **Watchlist stocks:** Hourly data (last 30 days), auto-promoted from screener hits

### Storage
- SQLite database
- Tables: `daily_prices`, `hourly_prices`, `watchlist`, `sentiment_scores`, `model_runs`, `signals`, `backtest_results`

## Analysis & Indicators

### Technical Indicators (`analysis/indicators.py`)

| Indicator | Config | Purpose | ML Feature |
|-----------|--------|---------|------------|
| SMA | 20, 50, 200 day | Trend identification, crossovers | Yes (ratios) |
| EMA | 12, 26 day | Responsive trend, feeds MACD | Yes |
| Bollinger Bands | 20-day, 2 std dev | Overbought/oversold, squeezes | Yes (% position) |
| RSI | 14-day | Momentum, reversals | Yes |
| MACD | 12/26 EMA, 9-day signal | Trend momentum, crossovers | Yes (histogram) |
| VWAP | Intraday | Fair value, entry timing | Yes (price vs VWAP) |
| ATR | 14-day | Stop-loss, position sizing, screener filter | Yes |

### Regression (`analysis/regression.py`)
- Linear regression on closing prices, configurable lookback (30, 60, 90 days)
- Outputs: slope, R-squared, predicted next-day price
- Used as ML feature input, not standalone signal

### ML Models (`analysis/ml_models.py`)
- **Model:** Random Forest classifier (buy/hold/sell)
- **Features:** SMA ratios, EMA crossover distance, Bollinger Band position, RSI, MACD histogram, VWAP ratio, ATR, regression slope, R-squared, volume change %, sentiment score
- **Target:** Forward 5-10 day return bucketed into buy (>3%), hold (-3% to 3%), sell (<-3%) — thresholds configurable
- **Training:** Rolling window — train on 1-2 years, validate on next 3 months
- **Retraining:** Weekly via APScheduler
- **Persistence:** `joblib`, versioned by date

## Sentiment Analysis

### Reddit Scraping (`sentiment/reddit.py`)
- PRAW (Python Reddit API Wrapper)
- Subreddits: `r/wallstreetbets`, `r/stocks`, `r/investing`
- Pulls posts and top comments mentioning tracked tickers
- Runs daily after market close
- Rate-limited per Reddit API terms

### Scoring (`sentiment/analyzer.py`)
- VADER sentiment scoring (compound score -1 to +1)
- Per-ticker daily aggregate: weighted average (upvotes as weight), mention count, sentiment trend
- Output: `sentiment_score`, `mention_count`, `sentiment_trend`
- Upgrade path: FinBERT for finance-specific sentiment if VADER proves too noisy

## Backtesting Engine

### Engine (`backtest/engine.py`)
- Inputs: strategy, ticker(s), date range
- Configurable: starting capital ($10K default), position sizing, commission/slippage, stop-loss/take-profit (ATR-based)
- Tracks: portfolio value, individual trades, cash balance

### Built-in Strategies (`backtest/strategies.py`)
- Standard interface: `generate_signals(data) -> buy/sell/hold` per day
- **SMA Crossover:** 20-day crosses 50-day
- **RSI + Bollinger:** RSI < 30 + lower band touch = buy, RSI > 70 + upper band = sell
- **ML Signal:** Random Forest predictions directly
- **Composite:** Weighted combination of all signals

### Performance Metrics
- Total return, annualized return
- Sharpe ratio, max drawdown
- Win rate, average win/loss ratio
- Trade count, average holding period
- Benchmark comparison vs. SPY buy-and-hold

## Screener & Signal Generation

### Screener (`data/screener.py`)
- Daily scan 30 minutes after market close
- Pipeline: fetch OHLCV → compute indicators → apply filters → promote survivors to watchlist → ML scoring
- Filters: minimum volume (1M/day), ATR threshold, RSI in actionable range, price above 200-day SMA
- All thresholds configurable via settings page

### Signal Generation
- Composite signal per watchlist stock combining:
  - ML prediction + confidence
  - Technical indicator alignment
  - Sentiment score and trend
- Output: ranked list with BUY/SELL/HOLD label and confidence score (0-100)
- Stored in `signals` table for historical tracking

## Dash UI

### Pages
1. **Dashboard (Home):** Top signals table, watchlist summary, market overview (SPY + overall sentiment)
2. **Stock Detail:** Candlestick chart with toggleable overlays, RSI/MACD subplots, sentiment timeline, ML prediction breakdown, "Run Backtest" button
3. **Backtesting:** Strategy selector, date range, equity curve with buy/sell markers, performance metrics, strategy comparison mode
4. **Settings:** Screener thresholds, watchlist management, retraining schedule/status, subreddit config, API keys

### Theme
- Dark theme for better chart contrast and reduced eye strain

## Scheduling

| Job | Schedule | Description |
|-----|----------|-------------|
| Daily screener | 4:30 PM ET (weekdays) | Full S&P 500 scan, filter, score |
| Hourly watchlist update | Every hour 9:30 AM - 4 PM ET | Fetch hourly data for watchlist stocks |
| Weekly ML retraining | Sunday evening | Retrain model, save new version, log performance |

## Configuration
- `pydantic-settings` for validation and env var support
- `.env` for secrets (API keys)
- `.env.example` committed as template

## Dependencies
- `dash`, `plotly` — UI and charting
- `pandas`, `numpy` — data manipulation
- `yfinance` — stock data
- `scikit-learn`, `joblib` — ML models
- `praw` — Reddit API
- `vaderSentiment` — sentiment scoring
- `APScheduler` — job scheduling
- `pydantic-settings` — config validation
- `pytest` — testing

## Phase 2: Trading (Future)
- Robinhood API via `robin_stocks` library
- Low-frequency swing trades
- Interface stubs in `trading/executor.py` during Phase 1
