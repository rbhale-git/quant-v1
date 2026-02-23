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

    # Screener controls + Run Scan
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
                    dbc.Button("Run Scan", id="dash-run-scan", color="primary"),
                ], width=3),
            ], align="end"),
            html.Div(id="dash-scan-output", className="mt-3"),
        ]),
    ], className="mb-3"),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Today's Top Signals"),
                dbc.CardBody(id="signals-table"),
            ]),
        ], width=7),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Watchlist"),
                dbc.CardBody(id="watchlist-summary"),
            ]),
            dbc.Card([
                dbc.CardHeader("Last Scan"),
                dbc.CardBody(id="scan-status"),
            ], className="mt-3"),
        ], width=5),
    ]),
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
    try:
        store = Store(settings.db_path)
        fetcher = Fetcher()
        screener = Screener(
            min_avg_volume=int(min_vol) if min_vol else None,
            atr_threshold=float(atr_thresh) if atr_thresh else None,
            rsi_buy_threshold=float(rsi_buy) if rsi_buy else None,
            rsi_sell_threshold=float(rsi_sell) if rsi_sell else None,
        )

        # Load watchlist tickers; fall back to fetching S&P 500 subset
        wl = store.get_watchlist()
        if not wl.empty:
            tickers = wl["ticker"].tolist()
        else:
            tickers = fetcher.get_sp500_tickers()[:50]

        end = date.today()
        start = end - timedelta(days=365)

        frames = []
        for ticker in tickers:
            df = store.load_daily_prices(ticker, start, end)
            if df.empty:
                df = fetcher.fetch_daily(ticker, period="1y")
                if not df.empty:
                    store.save_daily_prices(df)
            if df.empty or len(df) < 30:
                continue
            df = compute_all_indicators(df)
            df["ticker"] = ticker
            frames.append(df)

        if not frames:
            store.close()
            return html.P("No data available to screen."), 0

        universe = pd.concat(frames, ignore_index=True)
        passed = screener.screen(universe)
        passed_tickers = list(passed["ticker"].unique())

        for t in passed_tickers:
            store.add_to_watchlist(t, source="screener")

        store.close()

        msg = f"Scan complete — {len(passed_tickers)} stocks passed: {', '.join(passed_tickers[:15])}"
        if len(passed_tickers) > 15:
            msg += f" (+{len(passed_tickers) - 15} more)"

        return dbc.Alert(msg, color="success", dismissable=True), 0
    except Exception as e:
        return dbc.Alert(f"Scan error: {e}", color="danger", dismissable=True), no_update


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
        store.close()
        if wl.empty:
            return html.P("Watchlist is empty.")
        return html.Ul([html.Li(t) for t in wl["ticker"]])
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
