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
from backtest.engine import BacktestEngine
from backtest.strategies import SmaCrossover, STRATEGIES
from config import settings
from assets.plotly_theme import TERMINAL_LAYOUT_SUBPLOTS

dash.register_page(__name__, path="/backtest", name="Backtesting")

layout = dbc.Container([
    html.H2("Backtesting"),
    dbc.Card(
        dbc.CardBody(
            dbc.Row([
                dbc.Col([
                    dbc.Label("Ticker"),
                    dbc.Input(id="bt-ticker", placeholder="AAPL", type="text"),
                ], width=3),
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
                    html.Div(dcc.DatePickerSingle(id="bt-end", date=date.today())),
                ], width=2),
                dbc.Col([
                    dbc.Label("\u00a0"),
                    html.Br(),
                    dbc.Button("Run Backtest", id="bt-run", color="primary"),
                ], width=2),
            ], align="end"),
        ),
        className="mb-3",
    ),
    dcc.Loading(
        html.Div([
            html.Div(id="bt-results"),
            html.Div(id="bt-chart"),
        ]),
        type="default",
    ),
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
            fetcher = Fetcher()
            df = fetcher.fetch_daily(ticker, start=str(start_date), end=str(end_date))
            if df.empty:
                return html.P(f"No data available for {ticker}."), ""
            store = Store(settings.db_path)
            store.save_daily_prices(df)
            store.close()

        df = compute_all_indicators(df)
        strategy_cls = STRATEGIES.get(strategy_name, SmaCrossover)
        strategy = strategy_cls() if callable(strategy_cls) else strategy_cls

        engine = BacktestEngine()
        result = engine.run(df, strategy)

        asset_change = (df["close"].iloc[-1] / df["close"].iloc[0]) - 1
        alpha = result['total_return'] - asset_change

        METRIC_DEFINITIONS = {
            "Total Return": "The overall percentage gain or loss of the strategy over the entire backtest period.",
            "Asset Change": "The buy-and-hold return of the underlying asset over the same period, used as a benchmark.",
            "Alpha": "The excess return of the strategy compared to simply holding the asset. Positive alpha means the strategy outperformed.",
            "Annualized Return": "The total return scaled to a yearly rate, allowing comparison across different time periods.",
            "Sharpe Ratio": "Risk-adjusted return measuring excess return per unit of volatility. Above 1.0 is good, above 2.0 is excellent.",
            "Max Drawdown": "The largest peak-to-trough decline in portfolio value. Indicates the worst-case loss you would have experienced.",
            "Win Rate": "The percentage of closed trades that were profitable. A higher win rate doesn't always mean better performance.",
            "Trade Count": "The total number of completed round-trip trades (buy then sell) executed by the strategy.",
        }

        def _metric(label, value):
            return dbc.Col(html.Div([
                dbc.Card(dbc.CardBody([
                    html.P(label, className="text-muted mb-1", style={"fontSize": "0.78rem"}),
                    html.H5(value, className="mb-0"),
                ]), className="h-100", id=f"metric-{label.replace(' ', '-').lower()}",
                   style={"cursor": "pointer"}),
                dbc.Tooltip(
                    METRIC_DEFINITIONS[label],
                    target=f"metric-{label.replace(' ', '-').lower()}",
                    placement="bottom",
                ),
            ]), width=3)

        metrics_card = dbc.Card([
            dbc.CardHeader("Performance Metrics"),
            dbc.CardBody([
                dbc.Row([
                    _metric("Total Return", f"{result['total_return']:.2%}"),
                    _metric("Asset Change", f"{asset_change:.2%}"),
                    _metric("Alpha", f"{alpha:+.2%}"),
                    _metric("Annualized Return", f"{result['annualized_return']:.2%}"),
                ], className="mb-3"),
                dbc.Row([
                    _metric("Sharpe Ratio", f"{result['sharpe_ratio']:.2f}"),
                    _metric("Max Drawdown", f"{result['max_drawdown']:.2%}"),
                    _metric("Win Rate", f"{result['win_rate']:.2%}"),
                    _metric("Trade Count", str(result['trade_count'])),
                ]),
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
        fig.update_layout(**TERMINAL_LAYOUT_SUBPLOTS, height=600, title=f"Backtest: {strategy_name} on {ticker}")

        return metrics_card, dcc.Graph(figure=fig)
    except Exception as e:
        return html.P(f"Error: {e}"), ""
