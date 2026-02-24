import dash
from dash import html, dcc, DiskcacheManager
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

app.layout = dbc.Container([
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

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8050))
    debug = os.environ.get("RAILWAY_ENVIRONMENT") is None
    app.run(host="0.0.0.0", port=port, debug=debug)
