# Dashboard Documentation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a welcome modal (with Overview + Quick Start tabs) and collapsible inline guide sections to every page, so portfolio visitors immediately understand what the app does and how to use it.

**Architecture:** Welcome modal lives in `app.py` layout. Each page gets a reusable guide toggle (button + collapsible card) inserted before its main content. First-visit detection uses `dcc.Store(storage_type='local')`. All components use existing `dbc` library — zero new dependencies.

**Tech Stack:** Dash, dash-bootstrap-components, dcc.Store (localStorage)

---

### Task 1: Welcome Modal in app.py

**Files:**
- Modify: `app.py:1-56`

**Step 1: Add the welcome modal + store to app.py layout**

Add imports and the modal component. Insert `dcc.Store` and `dbc.Modal` into `app.layout` right before the `dbc.Row`:

```python
import dash
from dash import html, dcc, DiskcacheManager, callback, Input, Output, State
import dash_bootstrap_components as dbc
import diskcache

cache = diskcache.Cache("./.cache")
background_callback_manager = DiskcacheManager(cache)

app = dash.Dash(
    __name__,
    use_pages=True,
    pages_folder="ui/pages",
    external_stylesheets=[dbc.themes.DARKLY],
    suppress_callback_exceptions=True,
    background_callback_manager=background_callback_manager,
)
server = app.server

sidebar = dbc.Nav(
    [
        dbc.NavLink("Dashboard", href="/", active="exact"),
        dbc.NavLink("Stock Detail", href="/stock", active="exact"),
        dbc.NavLink("Backtesting", href="/backtest", active="exact"),
        dbc.NavLink("Portfolios", href="/portfolios", active="exact"),
        dbc.NavLink("Settings", href="/settings", active="exact"),
    ],
    vertical=True,
    pills=True,
)

welcome_modal = dbc.Modal([
    dbc.ModalHeader(dbc.ModalTitle("Welcome to Stock Analyzer Terminal")),
    dbc.ModalBody([
        dbc.Tabs([
            dbc.Tab(label="Overview", tab_id="tab-overview", children=html.Div([
                html.P(
                    "A full-stack stock analysis platform that combines real-time market data, "
                    "technical indicators, machine learning signals, and strategy backtesting — "
                    "all in a single terminal-style dashboard.",
                    className="mb-3",
                ),
                html.H6("Pages", className="mb-2"),
                html.Ul([
                    html.Li([html.Strong("Dashboard"), html.Span(" — Fetch stock data for major indices (S&P 500, NASDAQ-100, Dow 30), run the screener to find trading opportunities, and monitor your watchlist in real time.")]),
                    html.Li([html.Strong("Stock Detail"), html.Span(" — Analyze any stock with interactive candlestick charts and technical indicators including SMA, RSI, MACD, and Bollinger Bands.")]),
                    html.Li([html.Strong("Backtesting"), html.Span(" — Test trading strategies against historical data and see performance metrics like Sharpe ratio, max drawdown, and win rate.")]),
                    html.Li([html.Strong("Portfolios"), html.Span(" — Group stocks into portfolios, assign per-stock strategy overrides, and compare performance across all holdings.")]),
                    html.Li([html.Strong("Settings"), html.Span(" — Configure screener thresholds, manage your watchlist, and check ML model training status.")]),
                ]),
            ], className="p-2")),
            dbc.Tab(label="Quick Start", tab_id="tab-quickstart", children=html.Div([
                html.P("Follow these steps to get started:", className="mb-3"),
                html.Ol([
                    html.Li(["Go to ", html.Strong("Dashboard"), " and select an index (S&P 500, NASDAQ-100, or Dow 30)."]),
                    html.Li(["Click ", html.Strong("Fetch Data"), " to download the latest stock prices. This runs in the background and may take a minute."]),
                    html.Li(["Click ", html.Strong("Run Scan"), " to screen for stocks that match your signal criteria (volume, volatility, momentum)."]),
                    html.Li(["Check the ", html.Strong("Top Signals"), " table for buy/sell recommendations ranked by confidence score."]),
                    html.Li(["Go to ", html.Strong("Stock Detail"), ", enter a ticker symbol, and click ", html.Strong("Analyze"), " to view its full technical chart."]),
                    html.Li(["Try ", html.Strong("Backtesting"), " — pick a strategy and date range to see how it would have performed on historical data."]),
                    html.Li(["Use ", html.Strong("Portfolios"), " to group stocks together and compare strategy results side by side."]),
                ]),
            ], className="p-2")),
        ], active_tab="tab-overview"),
    ]),
    dbc.ModalFooter(
        dbc.Button("Get Started", id="welcome-close-btn", color="primary"),
    ),
], id="welcome-modal", size="lg", is_open=False, centered=True)

app.layout = dbc.Container([
    dcc.Store(id="welcome-seen", storage_type="local"),
    welcome_modal,
    dbc.Row([
        dbc.Col(
            html.Div([
                html.Div([
                    html.Span("Stock Analyzer"),
                    html.Span("Terminal", className="brand-sub"),
                ], className="sidebar-brand"),
                sidebar,
            ]),
            width=2,
            className="sidebar-container vh-100 position-fixed",
        ),
        dbc.Col(
            html.Div(dash.page_container, style={"padding": "28px 36px"}),
            width=10,
            className="offset-2",
        ),
    ]),
], fluid=True)


@callback(
    [Output("welcome-modal", "is_open"),
     Output("welcome-seen", "data")],
    [Input("welcome-seen", "data"),
     Input("welcome-close-btn", "n_clicks")],
)
def toggle_welcome_modal(seen_data, close_clicks):
    from dash import ctx
    if ctx.triggered_id == "welcome-close-btn":
        return False, True
    if not seen_data:
        return True, seen_data
    return False, seen_data


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8050))
    debug = os.environ.get("RAILWAY_ENVIRONMENT") is None
    app.run(host="0.0.0.0", port=port, debug=debug)
```

**Step 2: Verify the app starts without errors**

Run: `cd C:/Users/ronak/Projects/stock-analyzer && python -c "from app import app; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add app.py
git commit -m "feat: add welcome modal with Overview and Quick Start tabs"
```

---

### Task 2: Inline Guide for Dashboard Page

**Files:**
- Modify: `ui/pages/dashboard.py:1-84` (layout section)

**Step 1: Add the guide toggle and collapsible card to the dashboard layout**

Insert after `html.H2("Dashboard")` (line 21) and before the Screener card (line 24). Add necessary imports.

Add to imports at line 2 — extend the existing import:
```python
from dash import html, dcc, dash_table, callback, Input, Output, State, no_update, ctx
```

Insert after `html.H2("Dashboard"),` (line 21):
```python
    dbc.Button(
        "? Guide",
        id="dash-guide-toggle",
        color="link",
        size="sm",
        className="mb-2 p-0",
        style={"fontSize": "0.85rem", "textDecoration": "none", "color": "#999"},
    ),
    dbc.Collapse(
        dbc.Card(dbc.CardBody([
            html.P([
                html.Strong("How the screener works: "),
                "The screener filters stocks from the selected index by volume, volatility (ATR), and momentum (RSI). "
                "Stocks that pass all filters are added to your watchlist automatically.",
            ], className="mb-2"),
            html.P([
                html.Strong("Parameters: "),
            ], className="mb-1"),
            html.Ul([
                html.Li([html.Strong("Min Avg Volume"), " — Minimum average daily trading volume. Higher values focus on liquid, actively-traded stocks."]),
                html.Li([html.Strong("ATR Threshold"), " — Average True Range measures daily price volatility. Higher values find more volatile stocks."]),
                html.Li([html.Strong("RSI Buy/Sell"), " — Relative Strength Index thresholds. RSI below the buy level suggests oversold (potential buy); above the sell level suggests overbought (potential sell)."]),
            ], className="mb-2"),
            html.P([
                html.Strong("Top Signals: "),
                "Shows the latest BUY, SELL, and HOLD recommendations ranked by confidence (0 to 1). "
                "Signals are generated by running technical analysis and ML predictions on screened stocks.",
            ], className="mb-2"),
            html.P([
                html.Strong("Watchlist: "),
                "Displays live prices for tracked stocks. ",
                html.Span("Green", style={"color": "#00d632"}),
                " = price up, ",
                html.Span("Red", style={"color": "#ff3b30"}),
                " = price down vs. previous close.",
            ]),
        ]), className="mb-3", style={"borderColor": "#2a2a2a"}),
        id="dash-guide-collapse",
        is_open=False,
    ),
```

Add the callback at the end of the file:
```python
@callback(
    Output("dash-guide-collapse", "is_open"),
    Input("dash-guide-toggle", "n_clicks"),
    State("dash-guide-collapse", "is_open"),
    prevent_initial_call=True,
)
def toggle_dash_guide(n_clicks, is_open):
    return not is_open
```

**Step 2: Verify app starts**

Run: `cd C:/Users/ronak/Projects/stock-analyzer && python -c "from ui.pages.dashboard import layout; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add ui/pages/dashboard.py
git commit -m "feat: add inline guide section to Dashboard page"
```

---

### Task 3: Inline Guide for Stock Detail Page

**Files:**
- Modify: `ui/pages/stock_detail.py:16-31` (layout section)

**Step 1: Add guide toggle and collapsible card to stock_detail layout**

Insert after `html.H2("Stock Detail"),` (line 17) and before the `dbc.Row` (line 18):

```python
    dbc.Button(
        "? Guide",
        id="stock-guide-toggle",
        color="link",
        size="sm",
        className="mb-2 p-0",
        style={"fontSize": "0.85rem", "textDecoration": "none", "color": "#999"},
    ),
    dbc.Collapse(
        dbc.Card(dbc.CardBody([
            html.P(
                "Enter a ticker symbol and click Analyze to view a full technical analysis chart. "
                "If the stock isn't in the database yet, it will be fetched automatically.",
                className="mb-2",
            ),
            html.P(html.Strong("Chart indicators:"), className="mb-1"),
            html.Ul([
                html.Li([html.Strong("SMA 20 / SMA 50"), " — Simple Moving Averages smooth out price noise to show the short-term (20-day) and medium-term (50-day) trend direction. When the faster SMA crosses above the slower one, it may signal an uptrend."]),
                html.Li([html.Strong("Bollinger Bands"), " — An envelope around the price based on volatility. When price touches the lower band, the stock may be oversold; touching the upper band may indicate overbought conditions."]),
                html.Li([html.Strong("RSI (Relative Strength Index)"), " — A momentum oscillator ranging from 0 to 100. Below 30 is considered oversold (potential buying opportunity), above 70 is overbought (potential selling opportunity)."]),
                html.Li([html.Strong("MACD (Moving Average Convergence Divergence)"), " — Tracks the relationship between two moving averages. When the MACD line crosses above the signal line, it suggests upward momentum; crossing below suggests downward momentum."]),
            ]),
        ]), className="mb-3", style={"borderColor": "#2a2a2a"}),
        id="stock-guide-collapse",
        is_open=False,
    ),
```

Add the callback at the end of the file:
```python


@callback(
    Output("stock-guide-collapse", "is_open"),
    Input("stock-guide-toggle", "n_clicks"),
    State("stock-guide-collapse", "is_open"),
    prevent_initial_call=True,
)
def toggle_stock_guide(n_clicks, is_open):
    return not is_open
```

**Step 2: Verify**

Run: `cd C:/Users/ronak/Projects/stock-analyzer && python -c "from ui.pages.stock_detail import layout; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add ui/pages/stock_detail.py
git commit -m "feat: add inline guide section to Stock Detail page"
```

---

### Task 4: Inline Guide for Backtesting Page

**Files:**
- Modify: `ui/pages/backtesting.py:18-59` (layout section)

**Step 1: Add guide toggle and collapsible card to backtesting layout**

Insert after `html.H2("Backtesting"),` (line 19) and before the `dbc.Card(dbc.CardBody(` (line 20):

```python
    dbc.Button(
        "? Guide",
        id="bt-guide-toggle",
        color="link",
        size="sm",
        className="mb-2 p-0",
        style={"fontSize": "0.85rem", "textDecoration": "none", "color": "#999"},
    ),
    dbc.Collapse(
        dbc.Card(dbc.CardBody([
            html.P(
                "Backtesting simulates how a trading strategy would have performed on historical data. "
                "Select a stock, pick a strategy, choose a date range, and see the results.",
                className="mb-2",
            ),
            html.P(html.Strong("Available strategies:"), className="mb-1"),
            html.Ul([
                html.Li([html.Strong("SMA Crossover"), " — Buys when the 20-day moving average crosses above the 50-day average (uptrend signal), and sells when it crosses below (downtrend signal)."]),
                html.Li([html.Strong("RSI + Bollinger"), " — Buys when RSI is below 30 and price is near the lower Bollinger Band (oversold), sells when RSI is above 70 and price is near the upper band (overbought)."]),
                html.Li([html.Strong("Composite"), " — Combines SMA Crossover and RSI + Bollinger signals with equal weighting for a more balanced approach."]),
            ], className="mb-2"),
            html.P(html.Strong("Performance metrics:"), className="mb-1"),
            html.Ul([
                html.Li([html.Strong("Total Return"), " — Overall percentage gain or loss of the strategy."]),
                html.Li([html.Strong("Asset Change"), " — What you'd have earned by simply buying and holding the stock (the benchmark)."]),
                html.Li([html.Strong("Alpha"), " — The excess return over buy-and-hold. Positive alpha means the strategy beat the market."]),
                html.Li([html.Strong("Sharpe Ratio"), " — Risk-adjusted return. Above 1.0 is good, above 2.0 is excellent. Measures return per unit of risk."]),
                html.Li([html.Strong("Max Drawdown"), " — The worst peak-to-trough decline. Shows the largest loss you would have experienced."]),
                html.Li([html.Strong("Win Rate"), " — Percentage of trades that were profitable."]),
            ]),
        ]), className="mb-3", style={"borderColor": "#2a2a2a"}),
        id="bt-guide-collapse",
        is_open=False,
    ),
```

Add the callback at the end of the file:
```python


@callback(
    Output("bt-guide-collapse", "is_open"),
    Input("bt-guide-toggle", "n_clicks"),
    State("bt-guide-collapse", "is_open"),
    prevent_initial_call=True,
)
def toggle_bt_guide(n_clicks, is_open):
    return not is_open
```

**Step 2: Verify**

Run: `cd C:/Users/ronak/Projects/stock-analyzer && python -c "from ui.pages.backtesting import layout; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add ui/pages/backtesting.py
git commit -m "feat: add inline guide section to Backtesting page"
```

---

### Task 5: Inline Guide for Portfolios Page

**Files:**
- Modify: `ui/pages/portfolios.py:27-135` (layout function)

**Step 1: Add guide toggle and collapsible card to portfolios layout**

Inside the `layout()` function, insert after `html.H2("Portfolio Backtesting"),` (line 29) and before the Portfolio selector card (line 32):

```python
    dbc.Button(
        "? Guide",
        id="pf-guide-toggle",
        color="link",
        size="sm",
        className="mb-2 p-0",
        style={"fontSize": "0.85rem", "textDecoration": "none", "color": "#999"},
    ),
    dbc.Collapse(
        dbc.Card(dbc.CardBody([
            html.P(
                "Portfolios let you group stocks together and backtest them as a collection. "
                "You can assign different trading strategies to individual stocks to compare performance.",
                className="mb-2",
            ),
            html.P(html.Strong("Getting started:"), className="mb-1"),
            html.Ol([
                html.Li("Click New to create a portfolio with a name and default strategy."),
                html.Li("Add stocks by typing ticker symbols and clicking Add Stock."),
                html.Li("Optionally set a Strategy Override for individual stocks using the dropdown in the table."),
                html.Li("Click Save Overrides to persist your changes."),
                html.Li("Set a date range and click Run Portfolio Backtest to see results."),
            ], className="mb-2"),
            html.P([
                html.Strong("Results: "),
                "The comparison table shows performance metrics for each stock. "
                "The equity curves chart displays normalized percentage returns so you can compare stocks regardless of their price.",
            ]),
        ]), className="mb-3", style={"borderColor": "#2a2a2a"}),
        id="pf-guide-collapse",
        is_open=False,
    ),
```

Add the callback at the end of the file:
```python


@callback(
    Output("pf-guide-collapse", "is_open"),
    Input("pf-guide-toggle", "n_clicks"),
    State("pf-guide-collapse", "is_open"),
    prevent_initial_call=True,
)
def toggle_pf_guide(n_clicks, is_open):
    return not is_open
```

**Step 2: Verify**

Run: `cd C:/Users/ronak/Projects/stock-analyzer && python -c "from ui.pages.portfolios import layout; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add ui/pages/portfolios.py
git commit -m "feat: add inline guide section to Portfolios page"
```

---

### Task 6: Inline Guide for Settings Page

**Files:**
- Modify: `ui/pages/settings_page.py:10-46` (layout section)

**Step 1: Add guide toggle and collapsible card to settings layout**

Insert after `html.H2("Settings"),` (line 11) and before the `dbc.Row` (line 12):

```python
    dbc.Button(
        "? Guide",
        id="set-guide-toggle",
        color="link",
        size="sm",
        className="mb-2 p-0",
        style={"fontSize": "0.85rem", "textDecoration": "none", "color": "#999"},
    ),
    dbc.Collapse(
        dbc.Card(dbc.CardBody([
            html.P(html.Strong("Screener Thresholds:"), className="mb-1"),
            html.Ul([
                html.Li([html.Strong("Min Avg Volume"), " — Minimum average daily volume a stock must have to pass screening. Filters out thinly-traded stocks."]),
                html.Li([html.Strong("ATR Threshold"), " — Minimum Average True Range (daily volatility). Higher values select more volatile stocks with larger price swings."]),
                html.Li([html.Strong("RSI Buy Threshold"), " — Stocks with RSI below this level are considered oversold and flagged as potential buys."]),
                html.Li([html.Strong("RSI Sell Threshold"), " — Stocks with RSI above this level are considered overbought and flagged as potential sells."]),
            ], className="mb-2"),
            html.P([
                html.Strong("Watchlist: "),
                "Manually add ticker symbols to track. Stocks passing the screener are also added automatically. "
                "Watchlist tickers appear on the Dashboard with live price data.",
            ], className="mb-2"),
            html.P([
                html.Strong("ML Model Status: "),
                "Shows whether the Random Forest prediction model has been trained. "
                "The model uses technical indicators and sentiment data to generate BUY/SELL/HOLD signals with confidence scores. "
                "Run the training pipeline (", html.Code("python run_pipeline.py"), ") to train or retrain it.",
            ]),
        ]), className="mb-3", style={"borderColor": "#2a2a2a"}),
        id="set-guide-collapse",
        is_open=False,
    ),
```

Add the callback at the end of the file:
```python


@callback(
    Output("set-guide-collapse", "is_open"),
    Input("set-guide-toggle", "n_clicks"),
    State("set-guide-collapse", "is_open"),
    prevent_initial_call=True,
)
def toggle_set_guide(n_clicks, is_open):
    return not is_open
```

**Step 2: Verify**

Run: `cd C:/Users/ronak/Projects/stock-analyzer && python -c "from ui.pages.settings_page import layout; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add ui/pages/settings_page.py
git commit -m "feat: add inline guide section to Settings page"
```

---

### Task 7: Run Existing Tests

**Files:** None (verification only)

**Step 1: Run full test suite to verify nothing is broken**

Run: `cd C:/Users/ronak/Projects/stock-analyzer && python -m pytest tests/ -v --tb=short`
Expected: All existing tests pass.

**Step 2: Final commit if any fixups needed**

```bash
git add -A
git commit -m "fix: address test failures from documentation changes"
```
