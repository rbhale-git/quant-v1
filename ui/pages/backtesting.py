import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import date, timedelta
from data.store import Store
from analysis.indicators import compute_all_indicators
from backtest.engine import BacktestEngine
from backtest.strategies import SmaCrossover, RsiBollinger, CompositeStrategy
from config import settings

dash.register_page(__name__, path="/backtest", name="Backtesting")

STRATEGIES = {
    "SMA Crossover": SmaCrossover,
    "RSI + Bollinger": RsiBollinger,
    "Composite": lambda: CompositeStrategy([SmaCrossover(), RsiBollinger()]),
}

layout = dbc.Container([
    html.H2("Backtesting"),
    dbc.Row([
        dbc.Col([
            dbc.Label("Ticker"),
            dbc.Input(id="bt-ticker", placeholder="AAPL", type="text"),
        ], width=2),
        dbc.Col([
            dbc.Label("Strategy"),
            dcc.Dropdown(
                id="bt-strategy",
                options=[{"label": k, "value": k} for k in STRATEGIES],
                value="SMA Crossover",
            ),
        ], width=3),
        dbc.Col([
            dbc.Label("Start Date"),
            dcc.DatePickerSingle(id="bt-start", date=date.today() - timedelta(days=365)),
        ], width=2),
        dbc.Col([
            dbc.Label("End Date"),
            dcc.DatePickerSingle(id="bt-end", date=date.today()),
        ], width=2),
        dbc.Col([
            dbc.Label("\u00a0"),
            html.Br(),
            dbc.Button("Run Backtest", id="bt-run", color="primary"),
        ], width=2),
    ], className="mb-3"),
    dcc.Loading([
        html.Div(id="bt-results"),
        html.Div(id="bt-chart"),
    ]),
], fluid=True)


@callback(
    [Output("bt-results", "children"), Output("bt-chart", "children")],
    Input("bt-run", "n_clicks"),
    [State("bt-ticker", "value"), State("bt-strategy", "value"),
     State("bt-start", "date"), State("bt-end", "date")],
    prevent_initial_call=True,
)
def run_backtest(n_clicks, ticker, strategy_name, start_date, end_date):
    if not ticker:
        return html.P("Enter a ticker."), ""
    ticker = ticker.upper().strip()

    try:
        store = Store(settings.db_path)
        df = store.load_daily_prices(ticker, start_date, end_date)
        store.close()

        if df.empty:
            return html.P(f"No data for {ticker}."), ""

        df = compute_all_indicators(df)
        strategy_cls = STRATEGIES.get(strategy_name, SmaCrossover)
        strategy = strategy_cls() if callable(strategy_cls) else strategy_cls

        engine = BacktestEngine()
        result = engine.run(df, strategy)

        metrics_card = dbc.Card([
            dbc.CardHeader("Performance Metrics"),
            dbc.CardBody([
                html.P(f"Total Return: {result['total_return']:.2%}"),
                html.P(f"Annualized Return: {result['annualized_return']:.2%}"),
                html.P(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}"),
                html.P(f"Max Drawdown: {result['max_drawdown']:.2%}"),
                html.P(f"Win Rate: {result['win_rate']:.2%}"),
                html.P(f"Trade Count: {result['trade_count']}"),
            ]),
        ])

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.5, 0.5])
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["close"], name="Price",
        ), row=1, col=1)

        for trade in result["trades"]:
            color = "green" if trade["action"] == "BUY" else "red"
            marker = "triangle-up" if trade["action"] == "BUY" else "triangle-down"
            fig.add_trace(go.Scatter(
                x=[trade["date"]], y=[trade["price"]],
                mode="markers", name=trade["action"],
                marker=dict(color=color, size=12, symbol=marker),
                showlegend=False,
            ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=df["date"], y=result["equity_curve"], name="Portfolio",
        ), row=2, col=1)
        fig.update_layout(template="plotly_dark", height=600, title=f"Backtest: {strategy_name} on {ticker}")

        return metrics_card, dcc.Graph(figure=fig)
    except Exception as e:
        return html.P(f"Error: {e}"), ""
