import dash
from dash import html, dcc, DiskcacheManager, callback, Input, Output, State, ctx
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

welcome_modal = dbc.Modal(
    [
        dbc.ModalHeader(dbc.ModalTitle("Welcome to Stock Analyzer Terminal")),
        dbc.ModalBody(
            dbc.Tabs(
                [
                    dbc.Tab(
                        html.Div([
                            html.P(
                                "A full-stack stock analysis platform that combines "
                                "real-time market data, technical indicators, machine "
                                "learning signals, and strategy backtesting — all in a "
                                "single terminal-style dashboard."
                            ),
                            html.Ul([
                                html.Li([
                                    html.Strong("Dashboard"), ": ",
                                    "Fetch stock data for major indices (S&P 500, "
                                    "NASDAQ-100, Dow 30), run the screener to find "
                                    "trading opportunities, and monitor your watchlist "
                                    "in real time.",
                                ]),
                                html.Li([
                                    html.Strong("Stock Detail"), ": ",
                                    "Analyze any stock with interactive candlestick "
                                    "charts and technical indicators including SMA, "
                                    "RSI, MACD, and Bollinger Bands.",
                                ]),
                                html.Li([
                                    html.Strong("Backtesting"), ": ",
                                    "Test trading strategies against historical data "
                                    "and see performance metrics like Sharpe ratio, "
                                    "max drawdown, and win rate.",
                                ]),
                                html.Li([
                                    html.Strong("Portfolios"), ": ",
                                    "Group stocks into portfolios, assign per-stock "
                                    "strategy overrides, and compare performance "
                                    "across all holdings.",
                                ]),
                                html.Li([
                                    html.Strong("Settings"), ": ",
                                    "Configure screener thresholds, manage your "
                                    "watchlist, and check ML model training status.",
                                ]),
                            ]),
                        ]),
                        label="Overview",
                        tab_id="tab-overview",
                    ),
                    dbc.Tab(
                        html.Div([
                            html.Ol([
                                html.Li(
                                    "Go to Dashboard and select an index "
                                    "(S&P 500, NASDAQ-100, or Dow 30)."
                                ),
                                html.Li(
                                    "Click Fetch Data to download the latest stock "
                                    "prices. This runs in the background and may "
                                    "take a minute."
                                ),
                                html.Li(
                                    "Click Run Scan to screen for stocks that match "
                                    "your signal criteria (volume, volatility, momentum)."
                                ),
                                html.Li(
                                    "Check the Top Signals table for buy/sell "
                                    "recommendations ranked by confidence score."
                                ),
                                html.Li(
                                    "Go to Stock Detail, enter a ticker symbol, and "
                                    "click Analyze to view its full technical chart."
                                ),
                                html.Li(
                                    "Try Backtesting — pick a strategy and date range "
                                    "to see how it would have performed on historical data."
                                ),
                                html.Li(
                                    "Use Portfolios to group stocks together and compare "
                                    "strategy results side by side."
                                ),
                            ]),
                        ]),
                        label="Quick Start",
                        tab_id="tab-quickstart",
                    ),
                ],
                active_tab="tab-overview",
            ),
        ),
        dbc.ModalFooter(
            dbc.Button("Get Started", id="welcome-close-btn", className="ms-auto"),
        ),
    ],
    id="welcome-modal",
    size="lg",
    is_open=False,
    centered=True,
)

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
    Output("welcome-modal", "is_open"),
    Output("welcome-seen", "data"),
    Input("welcome-seen", "data"),
    Input("welcome-close-btn", "n_clicks"),
    prevent_initial_call=False,
)
def toggle_welcome_modal(seen_data, n_clicks):
    triggered = ctx.triggered_id
    if triggered == "welcome-close-btn":
        return False, True
    # First visit: seen_data is None or falsy
    if not seen_data:
        return True, dash.no_update
    return False, dash.no_update


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8050))
    debug = os.environ.get("RAILWAY_ENVIRONMENT") is None
    app.run(host="0.0.0.0", port=port, debug=debug)
