# Dashboard Documentation Design

**Date:** 2026-02-26
**Goal:** Add user-facing documentation to the dashboard so portfolio visitors understand what the app does and how to use it.

## Approach

Dash Bootstrap Modal + Collapsible inline hints. No new dependencies — uses existing `dbc.Modal`, `dbc.Collapse`, `dbc.Card`, `dbc.Tabs`, and `dcc.Store`.

## Welcome Modal

A large `dbc.Modal` that appears on first visit. Dismissal state persisted via `dcc.Store(storage_type='local')` so it doesn't reappear on refresh.

### Tab 1 — Overview

Intro paragraph + description of each page:

- **Dashboard:** Fetch stock data for major indices, run the screener to find trading opportunities, and monitor your watchlist in real time.
- **Stock Detail:** Analyze any stock with interactive candlestick charts and technical indicators (SMA, RSI, MACD, Bollinger Bands).
- **Backtesting:** Test 4 trading strategies against historical data and see performance metrics like Sharpe ratio, max drawdown, and win rate.
- **Portfolios:** Group stocks into portfolios, assign strategy overrides, and compare performance across holdings.
- **Settings:** Configure screener thresholds, manage your watchlist, and check ML model training status.

### Tab 2 — Quick Start

1. Go to **Dashboard** → select an index (S&P 500, NASDAQ-100, or Dow 30)
2. Click **Fetch Data** to download the latest stock prices
3. Click **Run Scan** to screen for stocks matching signal criteria
4. Check the **Top Signals** table for buy/sell recommendations with confidence scores
5. Go to **Stock Detail** → enter a ticker → click **Analyze** for the full chart
6. Try **Backtesting** → pick a strategy and date range to see historical performance
7. Use **Portfolios** to group stocks and compare strategies side by side

## Inline Page Hints

Each page gets a "? Guide" toggle button at the top that expands a collapsible card with page-specific help. Collapsed by default.

### Dashboard Page
- What the screener does (filters by volume, volatility, momentum)
- Parameter explanations (Min Volume, ATR Threshold, RSI thresholds)
- How to read the Signals table (BUY/SELL/HOLD + confidence 0-1)
- Watchlist color coding (green = up, red = down)

### Stock Detail Page
- SMA (20/50): Short and medium-term trend lines
- Bollinger Bands: Volatility envelope — lower band touch may signal oversold
- RSI: Momentum — below 30 oversold, above 70 overbought
- MACD: Trend change detector — crossovers signal momentum shifts

### Backtesting Page
- What backtesting is ("tests how a strategy would have performed historically")
- Brief description of each of the 4 strategies
- Plain-English explanation of each metric (Total Return, Alpha, Sharpe Ratio, Max Drawdown, Win Rate, Trade Count)

### Portfolios Page
- How to create/manage portfolios
- What strategy overrides do
- How to read comparison results and equity curves

### Settings Page
- What each screener threshold controls
- How the watchlist works
- What the ML model status means

## Implementation Scope

### Files to modify:
- `app.py` — Add welcome modal + dcc.Store + callback
- `ui/pages/dashboard.py` — Add inline guide section
- `ui/pages/stock_detail.py` — Add inline guide section
- `ui/pages/backtesting.py` — Add inline guide section
- `ui/pages/portfolios.py` — Add inline guide section
- `ui/pages/settings_page.py` — Add inline guide section

### No new files or dependencies required.
