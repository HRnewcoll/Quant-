"""
backtester.py – Event-Driven Backtesting Engine
================================================

Features
--------
  * Supports any bot that exposes a ``generate_signals(df)`` method
    returning a DataFrame with a ``signal`` column (+1 long, -1 short, 0 flat)
    and optional ``stop_loss`` / ``take_profit`` columns.
  * Realistic transaction costs (commission + slippage).
  * Per-trade stop-loss and take-profit (intrabar simulation using High/Low).
  * Full equity-curve tracking.
  * Rich performance metrics:
      Total Return, CAGR, Sharpe, Sortino, Calmar,
      Max Drawdown, Win Rate, Profit Factor, # Trades,
      Avg Trade Duration, Best/Worst Day.
  * Side-by-side comparison of multiple bots.
  * Equity-curve and drawdown charts saved to PNG.

Usage
-----
    from backtester import Backtester, compare_bots
    from quant_bot import QuantBot
    from ai_quant_bot import AIQuantBot
    from utils import fetch_data

    df = fetch_data("SPY", "2018-01-01", "2024-01-01")

    quant_result = Backtester().run(QuantBot(), df.copy())
    ai_result    = Backtester().run(AIQuantBot(), df.copy())

    compare_bots([quant_result, ai_result])
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from utils import sharpe_ratio, sortino_ratio, max_drawdown, calmar_ratio

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
INITIAL_CAPITAL = 100_000.0   # USD
COMMISSION = 0.001            # 0.10 % per trade (round-trip leg)
SLIPPAGE = 0.0005             # 0.05 % adverse slippage per execution


# ---------------------------------------------------------------------------
# Trade record
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    entry_date: Any
    exit_date: Optional[Any]
    direction: int          # +1 long, -1 short
    entry_price: float
    exit_price: float = 0.0
    shares: float = 0.0
    pnl: float = 0.0
    exit_reason: str = ""   # "signal", "stop_loss", "take_profit", "end_of_data"


# ---------------------------------------------------------------------------
# BacktestResult
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    bot_name: str
    symbol: str
    start: Any
    end: Any
    equity_curve: pd.Series = field(default_factory=pd.Series)
    returns: pd.Series = field(default_factory=pd.Series)
    trades: List[Trade] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"\n{'='*60}",
            f"  {self.bot_name}  |  {self.symbol}  |  {self.start} → {self.end}",
            f"{'='*60}",
        ]
        for k, v in self.metrics.items():
            if isinstance(v, float):
                lines.append(f"  {k:<28}: {v:>10.4f}")
            else:
                lines.append(f"  {k:<28}: {v!s:>10}")
        lines.append(f"{'='*60}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Backtester
# ---------------------------------------------------------------------------

class Backtester:
    """Event-driven backtester.

    Example
    -------
        result = Backtester(initial_capital=50_000).run(bot, df)
        print(result.summary())
    """

    def __init__(
        self,
        initial_capital: float = INITIAL_CAPITAL,
        commission: float = COMMISSION,
        slippage: float = SLIPPAGE,
    ) -> None:
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage

    # ------------------------------------------------------------------
    # Execution helpers
    # ------------------------------------------------------------------

    def _fill_price(self, price: float, direction: int) -> float:
        """Apply slippage in the direction that hurts the trader."""
        return price * (1 + direction * self.slippage)

    def _trade_cost(self, value: float) -> float:
        """One-side commission on the notional value."""
        return abs(value) * self.commission

    # ------------------------------------------------------------------
    # Core simulation loop
    # ------------------------------------------------------------------

    def run(self, bot: Any, df: pd.DataFrame, symbol: str = "ASSET") -> BacktestResult:
        """Run the backtest for a single bot.

        Parameters
        ----------
        bot     : any object with a ``generate_signals(df)`` method
        df      : raw OHLCV DataFrame (indicators will be added by the bot)
        symbol  : label used in reports / charts

        Returns
        -------
        BacktestResult
        """
        logger.info("Running backtest for %s on %s", bot.name, symbol)

        # Let the bot enrich the DataFrame and produce signals
        df = bot.generate_signals(df.copy())
        df.dropna(subset=["Close"], inplace=True)

        cash = self.initial_capital
        shares = 0.0
        position = 0          # +1, -1, or 0
        entry_price = 0.0
        entry_date = None
        stop_loss = np.nan
        take_profit = np.nan

        equity_series = pd.Series(index=df.index, dtype=float)
        trades: List[Trade] = []

        for i, (date, row) in enumerate(df.iterrows()):
            close = float(row["Close"])
            high = float(row["High"])
            low = float(row["Low"])
            sig = int(row.get("signal", 0))
            sl = float(row["stop_loss"]) if "stop_loss" in row and not pd.isna(row["stop_loss"]) else np.nan
            tp = float(row["take_profit"]) if "take_profit" in row and not pd.isna(row["take_profit"]) else np.nan

            # ---- Intrabar stop-loss / take-profit check ----
            if position != 0:
                exit_reason = None
                exit_price_raw = close  # default

                if position == 1:
                    if not np.isnan(stop_loss) and low <= stop_loss:
                        exit_price_raw = stop_loss
                        exit_reason = "stop_loss"
                    elif not np.isnan(take_profit) and high >= take_profit:
                        exit_price_raw = take_profit
                        exit_reason = "take_profit"
                elif position == -1:
                    if not np.isnan(stop_loss) and high >= stop_loss:
                        exit_price_raw = stop_loss
                        exit_reason = "stop_loss"
                    elif not np.isnan(take_profit) and low <= take_profit:
                        exit_price_raw = take_profit
                        exit_reason = "take_profit"

                if exit_reason:
                    fill = self._fill_price(exit_price_raw, -position)
                    proceeds = shares * fill * position
                    cost = self._trade_cost(shares * fill)
                    cash += proceeds - cost
                    pnl = (fill - entry_price) * position * shares - cost
                    trades.append(Trade(
                        entry_date=entry_date,
                        exit_date=date,
                        direction=position,
                        entry_price=entry_price,
                        exit_price=fill,
                        shares=shares,
                        pnl=pnl,
                        exit_reason=exit_reason,
                    ))
                    position = 0
                    shares = 0.0
                    stop_loss = np.nan
                    take_profit = np.nan

            # ---- Signal-driven entry / flip ----
            if sig != 0 and sig != position:
                # Close existing position if any
                if position != 0:
                    fill = self._fill_price(close, -position)
                    proceeds = shares * fill * position
                    cost = self._trade_cost(shares * fill)
                    cash += proceeds - cost
                    pnl = (fill - entry_price) * position * shares - cost
                    trades.append(Trade(
                        entry_date=entry_date,
                        exit_date=date,
                        direction=position,
                        entry_price=entry_price,
                        exit_price=fill,
                        shares=shares,
                        pnl=pnl,
                        exit_reason="signal",
                    ))
                    position = 0
                    shares = 0.0

                # Open new position
                fill = self._fill_price(close, sig)
                equity_now = cash + shares * close * position if position != 0 else cash
                risk_amount = equity_now * 0.01  # risk 1 % of equity
                atr = float(row.get("atr_14", close * 0.01))
                stop_dist = 2.0 * atr
                shares_new = max(risk_amount / stop_dist, 1.0) if stop_dist > 0 else 1.0
                notional = shares_new * fill
                cost = self._trade_cost(notional)

                # Check affordability (long only: need enough cash)
                if sig == 1 and notional + cost > cash:
                    shares_new = max((cash * 0.95) / fill, 0.0)
                    notional = shares_new * fill
                    cost = self._trade_cost(notional)

                if shares_new > 0:
                    if sig == 1:
                        cash -= notional + cost
                    else:  # short: receive proceeds but pay cost
                        cash += notional - cost

                    shares = shares_new
                    position = sig
                    entry_price = fill
                    entry_date = date
                    stop_loss = sl if not np.isnan(sl) else (fill - sig * 2.0 * atr)
                    take_profit = tp if not np.isnan(tp) else (fill + sig * 3.0 * atr)

            elif sig == 0 and position != 0:
                # Optional: flat signal closes position
                pass  # hold until opposite signal or stop/target

            # Mark equity to market
            # Short equity = cash + shares × (2×entry − mark)
            # i.e. cash already contains short proceeds; unrealised PnL offsets mark
            mark = close
            if position == 1:
                equity_series.loc[date] = cash + shares * mark
            elif position == -1:
                # Short P&L: profit when price falls below entry
                equity_series.loc[date] = cash + shares * (2 * entry_price - mark)
            else:
                equity_series.loc[date] = cash

        # ---- Close any open position at end ----
        if position != 0 and len(df) > 0:
            last_row = df.iloc[-1]
            last_date = df.index[-1]
            fill = self._fill_price(float(last_row["Close"]), -position)
            proceeds = shares * fill * position
            cost = self._trade_cost(shares * fill)
            cash += proceeds - cost
            pnl = (fill - entry_price) * position * shares - cost
            trades.append(Trade(
                entry_date=entry_date,
                exit_date=last_date,
                direction=position,
                entry_price=entry_price,
                exit_price=fill,
                shares=shares,
                pnl=pnl,
                exit_reason="end_of_data",
            ))
            equity_series.loc[last_date] = cash

        equity_curve = equity_series.ffill().fillna(self.initial_capital)
        daily_returns = equity_curve.pct_change().dropna()

        metrics = self._compute_metrics(
            equity_curve, daily_returns, trades, df
        )

        result = BacktestResult(
            bot_name=bot.name,
            symbol=symbol,
            start=df.index[0].date(),
            end=df.index[-1].date(),
            equity_curve=equity_curve,
            returns=daily_returns,
            trades=trades,
            metrics=metrics,
        )
        logger.info("Backtest complete – %s", result.summary())
        return result

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _compute_metrics(
        self,
        equity: pd.Series,
        returns: pd.Series,
        trades: List[Trade],
        df: pd.DataFrame,
    ) -> Dict[str, Any]:
        n_years = max(len(equity) / 252, 1e-6)
        total_return = (equity.iloc[-1] / equity.iloc[0]) - 1
        cagr = (1 + total_return) ** (1 / n_years) - 1

        mdd = max_drawdown(equity)
        sharpe = sharpe_ratio(returns)
        sortino = sortino_ratio(returns)
        calmar = calmar_ratio(returns, equity)

        winning = [t for t in trades if t.pnl > 0]
        losing = [t for t in trades if t.pnl <= 0]
        win_rate = len(winning) / len(trades) if trades else 0.0
        gross_profit = sum(t.pnl for t in winning)
        gross_loss = abs(sum(t.pnl for t in losing))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        avg_pnl = np.mean([t.pnl for t in trades]) if trades else 0.0

        durations = []
        for t in trades:
            if t.exit_date and t.entry_date:
                try:
                    d = (pd.Timestamp(t.exit_date) - pd.Timestamp(t.entry_date)).days
                    durations.append(d)
                except Exception:
                    pass
        avg_duration = float(np.mean(durations)) if durations else 0.0

        buy_hold_return = (df["Close"].iloc[-1] / df["Close"].iloc[0]) - 1

        return {
            "Initial Capital ($)": self.initial_capital,
            "Final Equity ($)": round(equity.iloc[-1], 2),
            "Total Return (%)": round(total_return * 100, 2),
            "CAGR (%)": round(cagr * 100, 2),
            "Buy & Hold Return (%)": round(buy_hold_return * 100, 2),
            "Sharpe Ratio": round(sharpe, 3),
            "Sortino Ratio": round(sortino, 3),
            "Calmar Ratio": round(calmar, 3),
            "Max Drawdown (%)": round(mdd * 100, 2),
            "Win Rate (%)": round(win_rate * 100, 2),
            "Profit Factor": round(profit_factor, 3),
            "# Trades": len(trades),
            "Avg Trade PnL ($)": round(avg_pnl, 2),
            "Avg Trade Duration (days)": round(avg_duration, 1),
        }


# ---------------------------------------------------------------------------
# Comparison & plotting
# ---------------------------------------------------------------------------

def compare_bots(
    results: List[BacktestResult],
    save_path: str = "backtest_comparison.png",
) -> None:
    """Print side-by-side metrics table and save equity/drawdown charts."""

    # ---- Console table ----
    print("\n" + "=" * 80)
    print("  BOT COMPARISON")
    print("=" * 80)

    all_metrics = list(results[0].metrics.keys())
    col_w = 30
    header = f"{'Metric':<{col_w}}" + "".join(f"{r.bot_name:>22}" for r in results)
    print(header)
    print("-" * (col_w + 22 * len(results)))
    for m in all_metrics:
        row = f"{m:<{col_w}}"
        for r in results:
            val = r.metrics.get(m, "N/A")
            if isinstance(val, float):
                row += f"{val:>22.4f}"
            else:
                row += f"{str(val):>22}"
        print(row)
    print("=" * 80)

    # ---- Charts ----
    fig, axes = plt.subplots(3, 1, figsize=(14, 14))
    fig.suptitle("Backtester: Bot Comparison", fontsize=16, fontweight="bold")

    # Equity curves
    ax0 = axes[0]
    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]
    for i, r in enumerate(results):
        eq_norm = r.equity_curve / r.equity_curve.iloc[0] * 100
        ax0.plot(eq_norm.index, eq_norm.values, label=r.bot_name, color=colors[i % len(colors)], linewidth=1.5)
    ax0.set_title("Equity Curves (Indexed to 100)")
    ax0.set_ylabel("Equity (indexed)")
    ax0.legend(loc="upper left")
    ax0.grid(True, alpha=0.3)
    ax0.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    # Drawdown
    ax1 = axes[1]
    for i, r in enumerate(results):
        rolling_max = r.equity_curve.cummax()
        dd = (r.equity_curve - rolling_max) / rolling_max * 100
        ax1.fill_between(dd.index, dd.values, 0, alpha=0.35, color=colors[i % len(colors)], label=r.bot_name)
        ax1.plot(dd.index, dd.values, color=colors[i % len(colors)], linewidth=0.8)
    ax1.set_title("Drawdown (%)")
    ax1.set_ylabel("Drawdown (%)")
    ax1.legend(loc="lower left")
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    # Monthly returns bar chart (first bot only, for readability)
    ax2 = axes[2]
    for i, r in enumerate(results):
        monthly = r.equity_curve.resample("ME").last().pct_change().dropna() * 100
        ax2.bar(
            monthly.index, monthly.values,
            width=20, alpha=0.6,
            color=colors[i % len(colors)],
            label=r.bot_name,
        )
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_title("Monthly Returns (%)")
    ax2.set_ylabel("Return (%)")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nChart saved → {save_path}")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from quant_bot import QuantBot
    from ai_quant_bot import AIQuantBot
    from utils import fetch_data

    parser = argparse.ArgumentParser(description="Run and compare trading bots via backtesting.")
    parser.add_argument("--symbol", default="SPY", help="Ticker symbol (default: SPY)")
    parser.add_argument("--start", default="2019-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default="2024-01-01", help="End date YYYY-MM-DD")
    parser.add_argument("--capital", type=float, default=100_000.0, help="Starting capital in USD")
    parser.add_argument("--output", default="backtest_comparison.png", help="Chart output file")
    args = parser.parse_args()

    print(f"\nFetching data for {args.symbol} ({args.start} → {args.end})…")
    raw_df = fetch_data(args.symbol, args.start, args.end)

    backtester = Backtester(initial_capital=args.capital)

    print("\n[1/2] Running Rule-Based Quant Bot…")
    quant_result = backtester.run(QuantBot(), raw_df.copy(), symbol=args.symbol)
    print(quant_result.summary())

    print("\n[2/2] Running AI Quant Bot (walk-forward training may take a minute)…")
    ai_result = backtester.run(AIQuantBot(), raw_df.copy(), symbol=args.symbol)
    print(ai_result.summary())

    compare_bots([quant_result, ai_result], save_path=args.output)
