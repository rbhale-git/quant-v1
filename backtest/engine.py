import pandas as pd
import numpy as np
from backtest.strategies import Strategy
from config import settings


class BacktestEngine:
    def __init__(
        self,
        starting_capital: float = None,
        commission: float = None,
        slippage: float = None,
    ):
        self.starting_capital = starting_capital or settings.backtest_starting_capital
        self.commission = commission or settings.backtest_commission
        self.slippage = slippage or settings.backtest_slippage

    def run(self, df: pd.DataFrame, strategy: Strategy) -> dict:
        signals = strategy.generate_signals(df)
        cash = self.starting_capital
        shares = 0
        equity_curve = []
        trades = []
        entry_price = 0.0

        for i in range(len(df)):
            price = df["close"].iloc[i]
            signal = signals.iloc[i]
            date = df["date"].iloc[i] if "date" in df.columns else i

            if signal == "BUY" and shares == 0:
                exec_price = price * (1 + self.slippage)
                shares = int(cash / exec_price)
                if shares > 0:
                    cost = shares * exec_price + self.commission
                    cash -= cost
                    entry_price = exec_price
                    trades.append({
                        "date": date, "action": "BUY",
                        "price": exec_price, "shares": shares,
                    })

            elif signal == "SELL" and shares > 0:
                exec_price = price * (1 - self.slippage)
                revenue = shares * exec_price - self.commission
                cash += revenue
                trades.append({
                    "date": date, "action": "SELL",
                    "price": exec_price, "shares": shares,
                    "pnl": (exec_price - entry_price) * shares,
                })
                shares = 0

            portfolio_value = cash + shares * price
            equity_curve.append(portfolio_value)

        equity = pd.Series(equity_curve)
        metrics = self._compute_metrics(equity, trades)
        metrics["equity_curve"] = equity
        metrics["trades"] = trades
        return metrics

    def _compute_metrics(self, equity: pd.Series, trades: list[dict]) -> dict:
        total_return = (equity.iloc[-1] / equity.iloc[0]) - 1
        n_days = len(equity)
        annualized_return = (1 + total_return) ** (252 / max(n_days, 1)) - 1

        daily_returns = equity.pct_change().dropna()
        sharpe = 0.0
        if daily_returns.std() > 0:
            sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)

        running_max = equity.cummax()
        drawdown = (equity - running_max) / running_max
        max_drawdown = drawdown.min()

        sell_trades = [t for t in trades if t["action"] == "SELL"]
        wins = [t for t in sell_trades if t.get("pnl", 0) > 0]
        win_rate = len(wins) / max(len(sell_trades), 1)

        avg_win = np.mean([t["pnl"] for t in wins]) if wins else 0
        losses = [t for t in sell_trades if t.get("pnl", 0) <= 0]
        avg_loss = abs(np.mean([t["pnl"] for t in losses])) if losses else 0
        win_loss_ratio = avg_win / max(avg_loss, 0.01)

        return {
            "total_return": round(total_return, 4),
            "annualized_return": round(annualized_return, 4),
            "sharpe_ratio": round(sharpe, 4),
            "max_drawdown": round(max_drawdown, 4),
            "win_rate": round(win_rate, 4),
            "win_loss_ratio": round(win_loss_ratio, 4),
            "trade_count": len(sell_trades),
        }
