# UI Redesign: "Dark Precision"

**Date:** 2026-02-24
**Direction:** Hybrid Bloomberg/Robinhood — Bloomberg's data density and credibility with Robinhood's polish and modern UI patterns.
**Stack:** Dash + dash-bootstrap-components + custom CSS (no new dependencies).

---

## Design System

### Color Palette

| Token | Value | Usage |
|-------|-------|-------|
| `--bg-base` | `#000000` | Page background, sidebar |
| `--bg-raised` | `#080808` | Main content area |
| `--bg-surface` | `#111111` | Card backgrounds |
| `--bg-elevated` | `#181818` | Card headers, dropdowns, tooltips |
| `--bg-overlay` | `rgba(0,0,0,0.6)` | Modal backdrops |
| `--border-subtle` | `#1e1e1e` | Default borders |
| `--border-medium` | `#2a2a2a` | Interactive borders, dividers |
| `--border-focus` | `#00d632` | Focus rings |
| `--text-primary` | `#ebebeb` | Headings, primary content |
| `--text-secondary` | `#8a8a8a` | Labels, descriptions |
| `--text-muted` | `#4a4a4a` | Placeholders, disabled |
| `--accent-green` | `#00d632` | Buy, positive, primary actions |
| `--accent-green-dim` | `rgba(0,214,50,0.08)` | Green tinted backgrounds |
| `--accent-red` | `#ff3b30` | Sell, negative, destructive |
| `--accent-red-dim` | `rgba(255,59,48,0.08)` | Red tinted backgrounds |
| `--accent-blue` | `#00aaff` | Informational, chart secondary |
| `--accent-blue-dim` | `rgba(0,170,255,0.08)` | Blue tinted backgrounds |
| `--accent-amber` | `#ff9f0a` | Warnings, hold signals |

### Typography

| Role | Font | Weight | Size | Transform |
|------|------|--------|------|-----------|
| Display | Geist Mono | 700 | 1.5rem | uppercase |
| Section headers | Geist Mono | 600 | 0.85rem | uppercase, ls 0.12em |
| Body | Geist Sans | 400 | 0.9rem | none |
| Data/numbers | Geist Mono | 500 | 0.9rem | tabular-nums |
| Labels | Geist Mono | 500 | 0.75rem | uppercase, ls 0.1em |
| Small | Geist Sans | 400 | 0.8rem | none |

### Elevation

| Level | Background | Shadow | Usage |
|-------|-----------|--------|-------|
| 0 | `--bg-base` | none | Page, sidebar |
| 1 | `--bg-raised` | none | Main content area |
| 2 | `--bg-surface` | `0 1px 3px rgba(0,0,0,0.4)` | Cards |
| 3 | `--bg-elevated` | `0 4px 16px rgba(0,0,0,0.5)` | Dropdowns, tooltips |
| Glass | `rgba(20,20,20,0.7)` + blur(20px) | `0 8px 32px rgba(0,0,0,0.6)` | Modals |

### Spacing

`4px / 8px / 12px / 16px / 24px / 32px / 48px / 64px`

### Border Radius

Cards, buttons, inputs: `2px`. Modals: `4px`. Everything else: `0`.

---

## Layout

### Sidebar (fixed, always expanded, 240px)

- Background: `--bg-base`, right border 1px `--border-subtle`
- Brand: "STOCK ANALYZER" — Geist Mono 600, 0.85rem, uppercase, green
- Nav: 5 links with icon + label, 44px row height
  - Active: 3px green left border, green text, green-dim background
  - Hover: `--bg-surface` background
- Footer: version text in `--text-muted`

### Main Content

- `margin-left: 240px`
- Background: `--bg-raised`
- Padding: `32px 40px`

### Page Header

- Title: Geist Mono 700, 1.5rem, uppercase
- Description: Geist Sans, `--text-secondary`, 0.85rem
- Divider: 1px `--border-subtle`
- Context actions right-aligned

### Card Pattern

- Border: 1px `--border-subtle`, 2px radius
- Header: `--bg-elevated`, 12px 16px padding, mono uppercase label
- Body: `--bg-surface`, 24px padding

---

## Pages

### Dashboard (/)

**Row 1 — Market Overview Bar:** 3 index tiles (S&P 500, NASDAQ-100, Dow 30). Each shows name, price, change %. Selected index has green bottom border.

**Row 2 — Two columns:**
- Left (60%): Screener controls (inline horizontal row) + Signals table with colored pill badges and confidence bars
- Right (40%): Watchlist with compact rows, colored left borders for pos/neg

### Stock Detail (/stock)

- Ticker input + Analyze button inline
- Hero: large ticker, price, change display
- Full-width candlestick + SMA/BB overlays (65% viewport height)
- RSI + MACD subplots below

### Backtesting (/backtest)

- Config: single inline row (Ticker | Strategy | Dates | Run button)
- Metrics: 4-card grid (Total Return, Sharpe, Max DD, Win Rate) + second row
- Chart: price with trade markers + equity curve

### Portfolios (/portfolios)

- Portfolio selector as horizontal pills
- Stock table with signal pills
- Equity comparison chart

### Settings (/settings)

- Three-card grid (Screener Thresholds, Watchlist, ML Status)
- Refined with new card pattern

---

## Animations

- **Page load:** Cards stagger in — `translateY(12px)→0` + `opacity 0→1`, 0.3s, 0.05s delay per card
- **Hover:** Border color transitions, row highlights, button brightness
- **Data updates:** Brief green/red pulse on changed numbers
- **Loading:** Skeleton shimmer animation on card bodies
- **Transitions:** All use `0.15s ease`

---

## Plotly Theme Updates

- Paper/plot backgrounds align with new elevation tokens
- Font: Geist Mono
- Gradient fills under SMA lines
- Refined gridlines (`--border-subtle`)
- Crosshair cursor mode
- Colors: green, red, amber, blue, purple, cyan
