import os
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
                    dbc.Label("Min Avg Volume", className="mt-2"),
                    dbc.Input(id="set-min-volume", type="number", value=settings.min_avg_volume),
                    dbc.Label("ATR Threshold", className="mt-3"),
                    dbc.Input(id="set-atr", type="number", value=settings.atr_threshold, step=0.1),
                    dbc.Label("RSI Buy Threshold", className="mt-3"),
                    dbc.Input(id="set-rsi-buy", type="number", value=settings.rsi_buy_threshold),
                    dbc.Label("RSI Sell Threshold", className="mt-3"),
                    dbc.Input(id="set-rsi-sell", type="number", value=settings.rsi_sell_threshold),
                ]),
            ]),
        ], width=4),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Watchlist Management"),
                dbc.CardBody([
                    dbc.Input(id="wl-ticker-input", placeholder="Add ticker..."),
                    dbc.Button("Add", id="wl-add-btn", color="success", className="mt-3"),
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


@callback(
    Output("ml-status", "children"),
    Input("wl-add-btn", "n_clicks"),
    prevent_initial_call=False,
)
def update_ml_status(n_clicks):
    model_path = os.path.join(settings.ml_model_dir, "stock_predictor.joblib")
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        return html.Div([
            html.P("Status: Trained", style={"color": "green"}),
            html.P(f"Model path: {model_path}"),
            html.P(f"Size: {size_mb:.1f} MB"),
        ])
    return html.P("Status: Not trained. Run pipeline to train model.", style={"color": "orange"})
