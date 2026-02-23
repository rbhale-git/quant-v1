import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import date, timedelta
from data.store import Store
from data.fetcher import Fetcher
from analysis.indicators import compute_all_indicators
from config import settings
from assets.plotly_theme import TERMINAL_LAYOUT_SUBPLOTS

dash.register_page(__name__, path="/stock", name="Stock Detail")

layout = dbc.Container([
    html.H2("Stock Detail"),
    dbc.Row([
        dbc.Col([
            dbc.Input(id="ticker-input", placeholder="Enter ticker (e.g. AAPL)", type="text"),
        ], width=5),
        dbc.Col([
            dbc.Button("Analyze", id="analyze-btn", color="primary"),
        ], width=3),
    ], className="mb-3", align="center"),
    dcc.Loading([
        html.Div(id="stock-chart"),
        html.Div(id="stock-indicators"),
        html.Div(id="stock-sentiment"),
    ]),
], fluid=True)


@callback(
    Output("stock-chart", "children"),
    Input("analyze-btn", "n_clicks"),
    State("ticker-input", "value"),
    prevent_initial_call=True,
)
def update_stock_chart(n_clicks, ticker):
    if not ticker:
        return html.P("Enter a ticker symbol.")
    ticker = ticker.upper().strip()
    try:
        store = Store(settings.db_path)
        end = date.today()
        start = end - timedelta(days=365)
        df = store.load_daily_prices(ticker, start, end)
        store.close()

        if df.empty:
            fetcher = Fetcher()
            df = fetcher.fetch_daily(ticker, period="1y")
            if df.empty:
                return html.P(f"No data available for {ticker}.")
            store = Store(settings.db_path)
            store.save_daily_prices(df)
            store.close()

        df = compute_all_indicators(df)

        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True, row_heights=[0.6, 0.2, 0.2],
            vertical_spacing=0.03,
        )

        fig.add_trace(go.Candlestick(
            x=df["date"], open=df["open"], high=df["high"],
            low=df["low"], close=df["close"], name="Price",
        ), row=1, col=1)

        if "sma_20" in df.columns:
            fig.add_trace(go.Scatter(x=df["date"], y=df["sma_20"], name="SMA 20", line=dict(width=1)), row=1, col=1)
        if "sma_50" in df.columns:
            fig.add_trace(go.Scatter(x=df["date"], y=df["sma_50"], name="SMA 50", line=dict(width=1)), row=1, col=1)
        if "bb_upper" in df.columns:
            fig.add_trace(go.Scatter(x=df["date"], y=df["bb_upper"], name="BB Upper", line=dict(width=1, dash="dot")), row=1, col=1)
            fig.add_trace(go.Scatter(x=df["date"], y=df["bb_lower"], name="BB Lower", line=dict(width=1, dash="dot")), row=1, col=1)

        if "rsi" in df.columns:
            fig.add_trace(go.Scatter(x=df["date"], y=df["rsi"], name="RSI"), row=2, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

        if "macd" in df.columns:
            fig.add_trace(go.Scatter(x=df["date"], y=df["macd"], name="MACD"), row=3, col=1)
            fig.add_trace(go.Scatter(x=df["date"], y=df["macd_signal"], name="Signal"), row=3, col=1)
            fig.add_trace(go.Bar(x=df["date"], y=df["macd_histogram"], name="Histogram"), row=3, col=1)

        fig.update_layout(
            **TERMINAL_LAYOUT_SUBPLOTS, height=800,
            xaxis_rangeslider_visible=False,
            title=f"{ticker} Analysis",
        )
        return dcc.Graph(figure=fig)
    except Exception as e:
        return html.P(f"Error: {e}")
