import dash
from dash import html, dcc, callback, Input, Output, State, dash_table, no_update, ctx
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
from datetime import date, timedelta
from data.store import Store
from data.fetcher import Fetcher
from analysis.indicators import compute_all_indicators
from backtest.engine import BacktestEngine
from backtest.strategies import STRATEGIES
from config import settings
from assets.plotly_theme import TERMINAL_LAYOUT, TERMINAL_COLORS

dash.register_page(__name__, path="/portfolios", name="Portfolios")

STRATEGY_OPTIONS = [{"label": k, "value": k} for k in STRATEGIES]


def _load_portfolio_options():
    store = Store(settings.db_path)
    portfolios = store.list_portfolios()
    store.close()
    return [{"label": p["name"], "value": p["id"]} for p in portfolios]


def layout():
    return dbc.Container([
    html.H2("Portfolio Backtesting"),

    # Portfolio selector row
    dbc.Card(dbc.CardBody(
        dbc.Row([
            dbc.Col([
                dbc.Label("Portfolio"),
                dcc.Dropdown(id="pf-selector", options=_load_portfolio_options(), placeholder="Select portfolio..."),
            ], width=6),
            dbc.Col([
                dbc.Label("\u00a0"),
                html.Br(),
                dbc.Button("New", id="pf-new-btn", color="success", className="me-2"),
                dbc.Button("Delete", id="pf-delete-btn", color="danger"),
            ], width=6),
        ], align="end"),
    ), className="mb-3"),

    # New portfolio modal
    dbc.Modal([
        dbc.ModalHeader("Create Portfolio"),
        dbc.ModalBody([
            dbc.Label("Name"),
            dbc.Input(id="pf-new-name", placeholder="My Portfolio"),
            dbc.Label("Default Strategy", className="mt-2"),
            dcc.Dropdown(id="pf-new-strategy", options=STRATEGY_OPTIONS, value="SMA Crossover"),
        ]),
        dbc.ModalFooter(
            dbc.Button("Create", id="pf-new-confirm", color="primary"),
        ),
    ], id="pf-new-modal", is_open=False),

    # Portfolio config card
    dbc.Card(dbc.CardBody([
        dbc.Row([
            dbc.Col([
                html.H5(id="pf-name-display"),
                html.P(id="pf-strategy-display", className="text-muted"),
            ], width=6),
        ]),
        html.Hr(),
        dbc.Row([
            dbc.Col([
                dbc.InputGroup([
                    dbc.Input(id="pf-add-ticker", placeholder="Ticker (e.g. AAPL)"),
                    dbc.Button("Add Stock", id="pf-add-ticker-btn", color="primary"),
                ]),
            ], width=4),
            dbc.Col([
                dbc.Button("Remove Selected", id="pf-remove-ticker-btn", color="warning", size="sm", className="me-2"),
                dbc.Button("Save Overrides", id="pf-save-overrides-btn", color="success", size="sm"),
                html.Span(id="pf-save-status", className="ms-2"),
            ], width=5),
        ], className="mb-3", align="center"),
        dash_table.DataTable(
            id="pf-stocks-table",
            columns=[
                {"name": "Ticker", "id": "ticker", "editable": False},
                {"name": "Strategy Override", "id": "strategy_override",
                 "editable": True, "presentation": "dropdown"},
            ],
            dropdown={
                "strategy_override": {
                    "options": [{"label": "(Use Default)", "value": ""}] + STRATEGY_OPTIONS,
                    "clearable": False,
                },
            },
            data=[],
            row_selectable="multi",
            selected_rows=[],
            style_table={"overflowX": "auto", "overflowY": "visible"},
            style_cell={"textAlign": "left"},
            style_cell_conditional=[
                {"if": {"column_id": "ticker"}, "fontWeight": "600"},
            ],
            css=[
                {"selector": ".Select-menu-outer", "rule": "display: block !important; z-index: 9999 !important;"},
                {"selector": ".row-selector-container", "rule": "display: none !important;"},
            ],
        ),
    ]), id="pf-config-card", className="mb-3", style={"display": "none"}),

    # Run controls
    dbc.Card(dbc.CardBody(
        dbc.Row([
            dbc.Col([
                dbc.Label("Start Date"),
                dcc.DatePickerSingle(id="pf-start", date=date.today() - timedelta(days=365)),
            ], width=4),
            dbc.Col([
                dbc.Label("End Date"),
                dcc.DatePickerSingle(id="pf-end", date=date.today()),
            ], width=4),
            dbc.Col([
                dbc.Label("\u00a0"),
                html.Br(),
                dbc.Button("Run Portfolio Backtest", id="pf-run-btn", color="primary"),
            ], width=4),
        ], align="end"),
    ), className="mb-3"),

    # Results
    dcc.Loading(html.Div([
        html.Div(id="pf-results"),
        html.Div(id="pf-chart"),
    ]), type="default"),
], fluid=True)


# Callback 1: New portfolio
@callback(
    [Output("pf-new-modal", "is_open"),
     Output("pf-selector", "options"),
     Output("pf-selector", "value"),
     Output("pf-new-name", "value")],
    [Input("pf-new-btn", "n_clicks"),
     Input("pf-new-confirm", "n_clicks")],
    [State("pf-new-modal", "is_open"),
     State("pf-new-name", "value"),
     State("pf-new-strategy", "value")],
    prevent_initial_call=True,
)
def handle_new_portfolio(open_clicks, confirm_clicks, is_open, name, strategy):
    trigger = ctx.triggered_id
    if trigger == "pf-new-btn":
        return True, no_update, no_update, ""
    if trigger == "pf-new-confirm" and name:
        store = Store(settings.db_path)
        new_id = store.create_portfolio(name.strip(), strategy)
        store.close()
        return False, _load_portfolio_options(), new_id, ""
    return is_open, no_update, no_update, no_update


# Callback 2: Delete portfolio
@callback(
    [Output("pf-selector", "options", allow_duplicate=True),
     Output("pf-selector", "value", allow_duplicate=True)],
    Input("pf-delete-btn", "n_clicks"),
    State("pf-selector", "value"),
    prevent_initial_call=True,
)
def handle_delete_portfolio(n_clicks, portfolio_id):
    if not portfolio_id:
        return no_update, no_update
    store = Store(settings.db_path)
    store.delete_portfolio(portfolio_id)
    store.close()
    return _load_portfolio_options(), None


# Callback 3: Load portfolio on selection
@callback(
    [Output("pf-name-display", "children"),
     Output("pf-strategy-display", "children"),
     Output("pf-stocks-table", "data"),
     Output("pf-config-card", "style")],
    Input("pf-selector", "value"),
    prevent_initial_call=True,
)
def load_portfolio(portfolio_id):
    if not portfolio_id:
        return "", "", [], {"display": "none"}
    store = Store(settings.db_path)
    portfolio = store.get_portfolio(portfolio_id)
    stocks = store.get_portfolio_stocks(portfolio_id)
    store.close()
    if not portfolio:
        return "", "", [], {"display": "none"}
    rows = [{"ticker": s["ticker"], "strategy_override": s["strategy_override"] or ""} for s in stocks]
    return (
        portfolio["name"],
        f"Default Strategy: {portfolio['default_strategy']}",
        rows,
        {"display": "block"},
    )


# Callback 4: Add ticker
@callback(
    [Output("pf-stocks-table", "data", allow_duplicate=True),
     Output("pf-add-ticker", "value")],
    Input("pf-add-ticker-btn", "n_clicks"),
    [State("pf-add-ticker", "value"),
     State("pf-selector", "value")],
    prevent_initial_call=True,
)
def add_ticker(n_clicks, ticker, portfolio_id):
    if not ticker or not portfolio_id:
        return no_update, no_update
    ticker = ticker.upper().strip()
    store = Store(settings.db_path)
    store.add_portfolio_stock(portfolio_id, ticker)
    stocks = store.get_portfolio_stocks(portfolio_id)
    store.close()
    rows = [{"ticker": s["ticker"], "strategy_override": s["strategy_override"] or ""} for s in stocks]
    return rows, ""


# Callback: Click ticker cell to toggle row selection
@callback(
    Output("pf-stocks-table", "selected_rows"),
    Input("pf-stocks-table", "active_cell"),
    State("pf-stocks-table", "selected_rows"),
    prevent_initial_call=True,
)
def toggle_row_selection(active_cell, selected_rows):
    if not active_cell or active_cell.get("column_id") != "ticker":
        return no_update
    row = active_cell["row"]
    selected = selected_rows or []
    if row in selected:
        selected = [r for r in selected if r != row]
    else:
        selected = selected + [row]
    return selected


# Callback 5a: Save strategy overrides — use a hidden Div + confirm button
#   DataTable dropdown edits change `data` directly, so we use a dedicated Save button.
@callback(
    Output("pf-save-status", "children"),
    Input("pf-save-overrides-btn", "n_clicks"),
    [State("pf-stocks-table", "data"),
     State("pf-selector", "value")],
    prevent_initial_call=True,
)
def save_overrides(n_clicks, rows, portfolio_id):
    if not portfolio_id or not rows:
        return ""
    store = Store(settings.db_path)
    for row in rows:
        override = row.get("strategy_override") or None
        store.add_portfolio_stock(portfolio_id, row["ticker"], override)
    store.close()
    return html.Small("Saved.", style={"color": "#00d632"})


# Callback 5b: Remove selected stocks
@callback(
    Output("pf-stocks-table", "data", allow_duplicate=True),
    Input("pf-remove-ticker-btn", "n_clicks"),
    [State("pf-stocks-table", "data"),
     State("pf-stocks-table", "selected_rows"),
     State("pf-selector", "value")],
    prevent_initial_call=True,
)
def remove_stocks(n_clicks, rows, selected_rows, portfolio_id):
    if not portfolio_id or not rows or not selected_rows:
        return no_update
    store = Store(settings.db_path)
    for idx in sorted(selected_rows, reverse=True):
        store.remove_portfolio_stock(portfolio_id, rows[idx]["ticker"])
    stocks = store.get_portfolio_stocks(portfolio_id)
    store.close()
    return [{"ticker": s["ticker"], "strategy_override": s["strategy_override"] or ""} for s in stocks]


# Callback 6: Run portfolio backtest
@callback(
    [Output("pf-results", "children"),
     Output("pf-chart", "children")],
    Input("pf-run-btn", "n_clicks"),
    [State("pf-selector", "value"),
     State("pf-start", "date"),
     State("pf-end", "date")],
    prevent_initial_call=True,
)
def run_portfolio_backtest(n_clicks, portfolio_id, start_date, end_date):
    if not portfolio_id:
        return html.P("Select a portfolio first."), ""

    store = Store(settings.db_path)
    portfolio = store.get_portfolio(portfolio_id)
    stocks = store.get_portfolio_stocks(portfolio_id)
    if not portfolio or not stocks:
        store.close()
        return html.P("Portfolio is empty. Add stocks first."), ""

    default_strategy = portfolio["default_strategy"]
    results = []
    equity_curves = {}

    try:
        for stock in stocks:
            ticker = stock["ticker"]
            strategy_name = stock["strategy_override"] or default_strategy

            df = store.load_daily_prices(ticker, start_date, end_date)
            if df.empty:
                fetcher = Fetcher()
                df = fetcher.fetch_daily(ticker, start=str(start_date), end=str(end_date))
                if df.empty:
                    results.append({"Ticker": ticker, "Strategy": strategy_name, "Error": "No data"})
                    continue
                store.save_daily_prices(df)

            df = compute_all_indicators(df)
            strategy_cls = STRATEGIES[strategy_name]
            strategy = strategy_cls()

            engine = BacktestEngine()
            result = engine.run(df, strategy)

            asset_change = (df["close"].iloc[-1] / df["close"].iloc[0]) - 1
            alpha = result["total_return"] - asset_change

            results.append({
                "Ticker": ticker,
                "Strategy": strategy_name,
                "Total Return": f"{result['total_return']:.2%}",
                "Asset Change": f"{asset_change:.2%}",
                "Alpha": f"{alpha:+.2%}",
                "Annualized": f"{result['annualized_return']:.2%}",
                "Sharpe": f"{result['sharpe_ratio']:.2f}",
                "Max DD": f"{result['max_drawdown']:.2%}",
                "Win Rate": f"{result['win_rate']:.2%}",
                "Trades": result["trade_count"],
            })

            # Normalize equity curve to % return
            eq = result["equity_curve"]
            equity_curves[f"{ticker} ({strategy_name})"] = ((eq / eq.iloc[0]) - 1) * 100

        store.close()

        # Metrics table
        metrics_df = pd.DataFrame(results)
        metrics_table = dash_table.DataTable(
            data=metrics_df.to_dict("records"),
            columns=[{"name": c, "id": c} for c in metrics_df.columns],
        )

        # Overlaid equity curves chart
        fig = go.Figure()
        for i, (label, curve) in enumerate(equity_curves.items()):
            fig.add_trace(go.Scatter(
                x=list(range(len(curve))),
                y=curve,
                mode="lines",
                name=label,
                line=dict(color=TERMINAL_COLORS[i % len(TERMINAL_COLORS)], width=1.5),
            ))
        fig.update_layout(
            **TERMINAL_LAYOUT,
            height=500,
            title="Equity Curves — Normalized % Return",
            xaxis_title="Trading Days",
            yaxis_title="Return (%)",
        )
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

        return (
            dbc.Card([dbc.CardHeader("Comparison Metrics"), dbc.CardBody(metrics_table)], className="mb-3"),
            dcc.Graph(figure=fig),
        )
    except Exception as e:
        store.close()
        return html.P(f"Error: {e}"), ""
