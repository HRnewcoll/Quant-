# Quant-

A complete quantitative trading suite consisting of:

1. **`quant_bot.py`** — Traditional rule-based quant trading bot  
2. **`ai_quant_bot.py`** — AI/ML quant trading bot (walk-forward ensemble)  
3. **`backtester.py`** — Event-driven backtesting engine with comparison charts  
4. **`utils.py`** — Shared data-fetching and 30+ technical indicator helpers

---

## Strategies at a Glance

### Rule-Based Quant Bot (`quant_bot.py`)

Combines **five technical signals** into a scored confluence system.
A trade is only placed when ≥ 3 signals agree, cutting false entries.

| Signal | Condition (long) |
|---|---|
| EMA Trend | 20-EMA > 50-EMA |
| RSI | RSI-14 < 40 (oversold) |
| MACD | Histogram positive & rising |
| Bollinger Bands | Price in lower 20 % of BB range |
| Volume | Volume > 1.2× 20-day average |

- **Stop-loss**: entry − 2 × ATR  
- **Take-profit**: entry + 3 × ATR (1.5 : 1 reward/risk)  
- **Position sizing**: risk 1 % of equity per trade via ATR-based share count

### AI Quant Bot (`ai_quant_bot.py`)

Uses a **walk-forward ensemble classifier** (Random Forest + Gradient Boosting)
that re-trains every quarter to stay current without lookahead bias.

| Category | Features |
|---|---|
| Returns | 1d, 3d, 5d, 10d, 20d + 5 lags |
| Trend | Close vs EMA-20/50/200 |
| Momentum | RSI-9/14, MACD histogram, ROC-14, Stochastic %K/%D |
| Volatility | BB %B, BB width, ATR/close ratio |
| Volume | Volume ratio |
| Calendar | Day of week, month |

- **Label**: 1 if next-bar forward return > 0.3 %, else 0  
- **Entry**: ensemble probability ≥ 0.58 (long) or ≤ 0.42 (short)  
- **Walk-forward window**: train on 252 bars, predict 63 bars, re-train every 63 bars

---

## Setup

```bash
pip install -r requirements.txt
```

Python ≥ 3.9 required.

---

## Quick Start

### 1. Run individual bots

```bash
# Rule-based bot
python quant_bot.py

# AI bot (takes ~30 s for walk-forward training)
python ai_quant_bot.py
```

### 2. Backtest and compare both bots

```bash
python backtester.py \
    --symbol SPY \
    --start  2019-01-01 \
    --end    2024-01-01 \
    --capital 100000 \
    --output backtest_comparison.png
```

### 3. Use the API in your own code

```python
from utils import fetch_data
from quant_bot import QuantBot
from ai_quant_bot import AIQuantBot
from backtester import Backtester, compare_bots

# Fetch historical data
df = fetch_data("AAPL", "2019-01-01", "2024-01-01")

# Run backtests
bt = Backtester(initial_capital=100_000)
quant_result = bt.run(QuantBot(), df.copy(), symbol="AAPL")
ai_result    = bt.run(AIQuantBot(), df.copy(), symbol="AAPL")

# Print metrics & save chart
print(quant_result.summary())
print(ai_result.summary())
compare_bots([quant_result, ai_result], save_path="comparison.png")
```

---

## Backtester Features

- **Realistic costs**: 0.10 % commission + 0.05 % slippage per execution  
- **Intrabar stop/target**: checks High and Low each bar to simulate fills  
- **Position sizing**: ATR-based, 1 % equity risk per trade  
- **Performance metrics**: Total Return, CAGR, Sharpe, Sortino, Calmar, Max Drawdown, Win Rate, Profit Factor, # Trades, Avg Trade Duration, Buy & Hold benchmark  
- **Comparison chart** (`backtest_comparison.png`): equity curves, drawdown chart, monthly return bars

---

## Files

| File | Description |
|---|---|
| `quant_bot.py` | Rule-based trading bot |
| `ai_quant_bot.py` | ML ensemble trading bot |
| `backtester.py` | Backtesting engine + comparison |
| `utils.py` | Data fetching + 30+ indicators |
| `requirements.txt` | Python dependencies |
| `backtest_comparison.png` | Example output chart |

---

## Disclaimer

This software is for **educational and research purposes only**.  
It does not constitute financial advice. Past performance on backtests
does not guarantee future profitability. Always paper-trade before
risking real capital, and consult a licensed financial advisor.
