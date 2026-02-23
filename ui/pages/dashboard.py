import dash
from dash import html, dcc, dash_table, callback, Input, Output, State, no_update
import dash_bootstrap_components as dbc
import pandas as pd
from datetime import date, timedelta
from data.store import Store
from data.fetcher import Fetcher
from data.screener import Screener
from analysis.indicators import compute_all_indicators
from config import settings

dash.register_page(__name__, path="/", name="Dashboard")

layout = dbc.Container([
    html.H2("Dashboard"),

    # Screener controls + Run Scan + Last Scan status
    dbc.Card([
        dbc.CardHeader("Screener"),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    dbc.Label("Min Avg Volume"),
                    dbc.Input(id="dash-min-volume", type="number", value=settings.min_avg_volume),
                ], width=3),
                dbc.Col([
                    dbc.Label("ATR Threshold"),
                    dbc.Input(id="dash-atr", type="number", value=settings.atr_threshold, step=0.1),
                ], width=2),
                dbc.Col([
                    dbc.Label("RSI Buy"),
                    dbc.Input(id="dash-rsi-buy", type="number", value=settings.rsi_buy_threshold),
                ], width=2),
                dbc.Col([
                    dbc.Label("RSI Sell"),
                    dbc.Input(id="dash-rsi-sell", type="number", value=settings.rsi_sell_threshold),
                ], width=2),
                dbc.Col([
                    dbc.Label("\u00a0"),
                    html.Br(),
                    html.Div([
                        dbc.Button("Run Scan", id="dash-run-scan", color="primary", className="me-2"),
                        dbc.Button("Fetch S&P 500", id="dash-fetch-data", color="success", size="sm"),
                    ], className="d-flex align-items-center"),
                ], width=3),
            ], align="end"),
            dcc.Loading(
                html.Div(id="dash-scan-output", className="mt-3"),
                type="default",
                color="#00d632",
            ),
            html.Div(id="dash-fetch-progress", className="mt-3"),
            html.Div(id="scan-status", className="mt-2"),
        ]),
    ], className="mb-3"),

    # Top Signals — full width
    dbc.Card([
        dbc.CardHeader("Today's Top Signals"),
        dbc.CardBody(id="signals-table"),
    ], className="mb-3"),

    # Watchlist — full width
    dbc.Card([
        dbc.CardHeader("Watchlist"),
        dbc.CardBody(id="watchlist-summary"),
    ], className="mb-3"),
    dcc.Interval(id="dashboard-refresh", interval=60_000, n_intervals=0),
], fluid=True)


@callback(
    [Output("dash-scan-output", "children"),
     Output("dashboard-refresh", "n_intervals")],
    Input("dash-run-scan", "n_clicks"),
    [State("dash-min-volume", "value"),
     State("dash-atr", "value"),
     State("dash-rsi-buy", "value"),
     State("dash-rsi-sell", "value")],
    prevent_initial_call=True,
)
def run_scan(n_clicks, min_vol, atr_thresh, rsi_buy, rsi_sell):
    """Screen all tickers already in the database — no network calls."""
    try:
        store = Store(settings.db_path)
        screener = Screener(
            min_avg_volume=int(min_vol) if min_vol else None,
            atr_threshold=float(atr_thresh) if atr_thresh else None,
            rsi_buy_threshold=float(rsi_buy) if rsi_buy else None,
            rsi_sell_threshold=float(rsi_sell) if rsi_sell else None,
        )

        end = date.today()
        start = end - timedelta(days=365)

        # Get all tickers that exist in the database
        all_tickers = pd.read_sql_query(
            "SELECT DISTINCT ticker FROM daily_prices", store.conn
        )["ticker"].tolist()

        if not all_tickers:
            store.close()
            return dbc.Alert(
                "No data in database. Click 'Fetch S&P 500' first to download price data.",
                color="warning",
            ), no_update

        frames = []
        for ticker in all_tickers:
            df = store.load_daily_prices(ticker, start, end)
            if df.empty or len(df) < 30:
                continue
            df = compute_all_indicators(df)
            df["ticker"] = ticker
            frames.append(df)

        if not frames:
            store.close()
            return dbc.Alert("Not enough price history to screen.", color="warning"), no_update

        universe = pd.concat(frames, ignore_index=True)
        passed = screener.screen(universe)
        passed_tickers = list(passed["ticker"].unique())

        for t in passed_tickers:
            store.add_to_watchlist(t, source="screener")

        store.close()

        msg = f"Scan complete — {len(passed_tickers)}/{len(all_tickers)} stocks passed: {', '.join(passed_tickers[:20])}"
        if len(passed_tickers) > 20:
            msg += f" (+{len(passed_tickers) - 20} more)"

        return dbc.Alert(msg, color="success", dismissable=True), 0
    except Exception as e:
        return dbc.Alert(f"Scan error: {e}", color="danger", dismissable=True), no_update


@callback(
    Output("dash-scan-output", "children", allow_duplicate=True),
    Input("dash-fetch-data", "n_clicks"),
    State("dash-fetch-progress", "id"),
    prevent_initial_call=True,
    background=True,
    running=[
        (Output("dash-fetch-data", "disabled"), True, False),
        (Output("dash-run-scan", "disabled"), True, False),
    ],
    progress=[Output("dash-fetch-progress", "children")],
)
def fetch_sp500_data(set_progress, n_clicks, _progress_id):
    """Fetch 1 year of daily data for all S&P 500 tickers into the database."""
    try:
        set_progress(dbc.Progress(value=0, label="Starting...", color="success",
                                  className="mb-2", style={"height": "24px"}))

        fetcher = Fetcher()
        store = Store(settings.db_path)
        tickers = fetcher.get_sp500_tickers()

        # Check which tickers already have recent data
        existing = pd.read_sql_query(
            "SELECT ticker, MAX(date) as last_date FROM daily_prices GROUP BY ticker",
            store.conn,
        )
        recent_cutoff = str(date.today() - timedelta(days=3))
        up_to_date = set(
            existing[existing["last_date"] >= recent_cutoff]["ticker"].tolist()
        )
        to_fetch = [t for t in tickers if t not in up_to_date]

        if not to_fetch:
            store.close()
            set_progress(html.Div())
            return dbc.Alert(
                f"All {len(tickers)} S&P 500 tickers already up to date.",
                color="info", dismissable=True,
            )

        fetched = 0
        errors = 0
        total = len(to_fetch)
        for i, ticker in enumerate(to_fetch):
            try:
                df = fetcher.fetch_daily(ticker, period="1y")
                if not df.empty:
                    store.save_daily_prices(df)
                    fetched += 1
            except Exception:
                errors += 1

            pct = int((i + 1) / total * 100)
            set_progress(dbc.Progress(
                value=pct,
                label=f"Fetching {i + 1}/{total} ({ticker})",
                color="success",
                className="mb-2",
                style={"height": "24px"},
            ))

        store.close()
        set_progress(html.Div())

        msg = f"Fetched {fetched} tickers ({len(up_to_date)} already cached"
        if errors:
            msg += f", {errors} errors"
        msg += "). Click 'Run Scan' to screen them."

        return dbc.Alert(msg, color="success", dismissable=True)
    except Exception as e:
        set_progress(html.Div())
        return dbc.Alert(f"Fetch error: {e}", color="danger", dismissable=True)


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
            return html.P("No signals yet. Run a scan first.")
        signals = signals.sort_values("confidence", ascending=False).head(10)
        return dash_table.DataTable(
            data=signals.to_dict("records"),
            columns=[{"name": c, "id": c} for c in signals.columns],
            style_table={"overflowX": "auto"},
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
        if wl.empty:
            store.close()
            return html.P("Watchlist is empty.")

        today = date.today()
        start = today - timedelta(days=10)
        rows = []
        for ticker in wl["ticker"]:
            df = store.load_daily_prices(ticker, start, today)
            if df.empty or len(df) < 2:
                continue
            latest = df.iloc[-1]
            prev_close = df.iloc[-2]["close"]
            day_change = latest["close"] - prev_close
            day_change_pct = (day_change / prev_close) * 100 if prev_close else 0
            rows.append({
                "Ticker": ticker,
                "Open": round(latest["open"], 2),
                "High": round(latest["high"], 2),
                "Low": round(latest["low"], 2),
                "Close": round(latest["close"], 2),
                "Change": round(day_change, 2),
                "Change %": round(day_change_pct, 2),
                "Volume": int(latest["volume"]),
            })

        store.close()

        if not rows:
            return html.P("No price data available for watchlist tickers.")

        return dash_table.DataTable(
            data=rows,
            columns=[{"name": c, "id": c} for c in rows[0].keys()],
            sort_action="native",
            style_table={"overflowX": "auto"},
            style_data_conditional=[
                {"if": {"filter_query": "{Change %} > 0", "column_id": "Change %"},
                 "color": "#00d632"},
                {"if": {"filter_query": "{Change %} < 0", "column_id": "Change %"},
                 "color": "#ff3b30"},
                {"if": {"filter_query": "{Change} > 0", "column_id": "Change"},
                 "color": "#00d632"},
                {"if": {"filter_query": "{Change} < 0", "column_id": "Change"},
                 "color": "#ff3b30"},
            ],
        )
    except Exception:
        return html.P("No data available.")


@callback(
    Output("scan-status", "children"),
    Input("dashboard-refresh", "n_intervals"),
)
def update_scan_status(n):
    try:
        store = Store(settings.db_path)
        signals = store.load_signals(date.today() - timedelta(days=7), date.today())
        store.close()
        if signals.empty:
            return html.P("No scans yet.")
        last_date = signals["date"].max()
        count = len(signals[signals["date"] == last_date])
        return html.Div([
            html.P(f"Last scan: {last_date}"),
            html.P(f"Signals generated: {count}"),
        ])
    except Exception:
        return html.P("No scan data available.")
