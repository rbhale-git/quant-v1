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


@callback(
    Output("stock-guide-collapse", "is_open"),
    Input("stock-guide-toggle", "n_clicks"),
    State("stock-guide-collapse", "is_open"),
    prevent_initial_call=True,
)
def toggle_stock_guide(n_clicks, is_open):
    return not is_open
