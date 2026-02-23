"""Shared Plotly layout defaults for the terminal theme."""

_AXIS_STYLE = dict(gridcolor="#2a2a2a", zerolinecolor="#333333", tickfont=dict(color="#999999"))

# Base layout — safe for single-axis figures
TERMINAL_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#131313",
    plot_bgcolor="#0a0a0a",
    font=dict(family="JetBrains Mono, Consolas, monospace", size=11, color="#999999"),
    title_font=dict(size=13, color="#f0f0f0"),
    xaxis=dict(**_AXIS_STYLE),
    yaxis=dict(**_AXIS_STYLE),
    legend=dict(font=dict(size=10, color="#999999")),
    margin=dict(l=48, r=24, t=48, b=36),
)

# For subplots — excludes xaxis/yaxis so it won't clobber subplot axes
TERMINAL_LAYOUT_SUBPLOTS = dict(
    template="plotly_dark",
    paper_bgcolor="#131313",
    plot_bgcolor="#0a0a0a",
    font=dict(family="JetBrains Mono, Consolas, monospace", size=11, color="#999999"),
    title_font=dict(size=13, color="#f0f0f0"),
    legend=dict(font=dict(size=10, color="#999999")),
    margin=dict(l=48, r=24, t=48, b=36),
)

TERMINAL_COLORS = ["#00d632", "#ff3b30", "#ff9f0a", "#5ac8fa", "#bf5af2", "#64d2ff"]
