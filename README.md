# Stock Analyzer Terminal

A Python-based stock analysis platform with automated screening, ML predictions, Reddit sentiment analysis, backtesting, and an interactive Dash web UI.

## Features

- **Multi-Index Screener** — Screen S&P 500, NASDAQ-100, and Dow 30 with configurable volume, ATR, RSI, and SMA filters
- **Technical Indicators** — SMA, EMA, Bollinger Bands, RSI, MACD, VWAP, ATR
- **ML Predictions** — Random Forest classifier for buy/hold/sell signals with confidence scoring
- **Sentiment Analysis** — Reddit scraping (r/wallstreetbets, r/stocks, r/investing) with VADER sentiment scoring
- **Backtesting** — 4 strategies (SMA Crossover, RSI+Bollinger, ML, Composite) with trade tracking and performance metrics
- **Portfolio Management** — Group stocks into portfolios with per-stock strategy overrides
- **Scheduled Jobs** — Daily screening, hourly watchlist updates, weekly ML retraining via APScheduler
- **Terminal-Style UI** — Dark Bloomberg-inspired theme with interactive Plotly charts

## Tech Stack

Python 3.11+ | Dash | Plotly | pandas | scikit-learn | yfinance | SQLite | APScheduler | PRAW + VADER

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# (Optional) Set up Reddit API for sentiment analysis
cp .env.example .env
# Edit .env with your REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET

# Run the app
python app.py
# Open http://localhost:8050
```

## Usage

### Web UI

The app has 5 pages accessible from the sidebar:

- **Dashboard** — Select an index, fetch data, run screener, view top signals and watchlist with live price metrics
- **Stock Detail** — Single-stock analysis with candlestick charts and overlaid indicators
- **Backtesting** — Run strategies against historical data with performance reports
- **Portfolios** — Create stock groups with strategy overrides and run batch backtests
- **Settings** — Configure screener thresholds, manage watchlist, view ML model status

### CLI Pipeline

```bash
# Fetch 1 year of daily data for all S&P 500 stocks
python run_pipeline.py fetch

# Fetch specific tickers
python run_pipeline.py fetch AAPL MSFT TSLA

# Run screener on all data in DB
python run_pipeline.py screen

# Generate buy/sell/hold signals for watchlist
python run_pipeline.py signals

# Run full pipeline: fetch -> screen -> signals
python run_pipeline.py all
```

## Project Structure

```
stock-analyzer/
├── app.py                  # Dash entry point
├── config.py               # Pydantic settings
├── run_pipeline.py         # CLI pipeline runner
├── scheduler.py            # APScheduler jobs
├── data/
│   ├── fetcher.py          # Yahoo Finance + index ticker scraping
│   ├── store.py            # SQLite storage layer
│   └── screener.py         # Stock screening + composite signals
├── analysis/
│   ├── indicators.py       # Technical indicators
│   ├── regression.py       # Linear regression trends
│   └── ml_models.py        # Random Forest predictor
├── sentiment/
│   ├── reddit.py           # PRAW Reddit scraper
│   └── analyzer.py         # VADER sentiment scoring
├── backtest/
│   ├── engine.py           # Backtesting engine
│   └── strategies.py       # Trading strategies
├── ui/pages/               # Dash UI pages
├── assets/                 # CSS theme + Plotly config
└── tests/                  # pytest suite (46 tests)
```

## Deployment

The app is configured for Railway deployment:

```bash
# Deployed via GitHub integration on Railway
# railway.app -> New Project -> Deploy from GitHub Repo
```

Includes `Procfile` (gunicorn), `runtime.txt` (Python 3.11), and auto-binds to the `PORT` environment variable.

## Running Tests

```bash
pytest
pytest --cov    # With coverage
```

## License

MIT
