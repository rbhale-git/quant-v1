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
