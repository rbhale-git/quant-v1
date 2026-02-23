import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

app = dash.Dash(
    __name__,
    use_pages=True,
    pages_folder="ui/pages",
    external_stylesheets=[dbc.themes.DARKLY],
    suppress_callback_exceptions=True,
)

sidebar = dbc.Nav(
    [
        dbc.NavLink("Dashboard", href="/", active="exact"),
        dbc.NavLink("Stock Detail", href="/stock", active="exact"),
        dbc.NavLink("Backtesting", href="/backtest", active="exact"),
        dbc.NavLink("Settings", href="/settings", active="exact"),
    ],
    vertical=True,
    pills=True,
    className="bg-dark",
)

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(
            html.Div([
                html.H4("Stock Analyzer", className="text-light p-3"),
                sidebar,
            ]),
            width=2,
            className="bg-dark vh-100 position-fixed",
        ),
        dbc.Col(
            dash.page_container,
            width=10,
            className="ms-auto p-4",
        ),
    ]),
], fluid=True, className="bg-dark text-light")

if __name__ == "__main__":
    app.run(debug=True)
