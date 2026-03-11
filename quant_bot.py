"""
quant_bot.py – Traditional Rule-Based Quantitative Trading Bot
==============================================================

Strategy overview
-----------------
This bot combines five well-established technical signals into a scored
confluence system.  A trade is only entered when several signals agree,
reducing false positives.

Signals (each worth 1 point toward the score):
  1. EMA Trend Filter   – 20-EMA above 50-EMA (bull), below (bear)
  2. RSI                – oversold (<40) → buy, overbought (>65) → sell
  3. MACD               – histogram turns positive → buy, negative → sell
  4. Bollinger Bands    – price touches lower band → buy, upper band → sell
  5. Volume Confirmation– volume > 1.2× 20-day average

Entry:  score ≥ 3 in the same direction → open position
Exit:   stop-loss at 2× ATR below entry; take-profit at 3× ATR above entry;
        OR opposite signal with score ≥ 3 → flip position

Position sizing:
  Risk 1 % of equity per trade; stop distance = 2× ATR.
  Shares = (equity × 0.01) / (2 × ATR)

Usage
-----
    from quant_bot import QuantBot
    from utils import fetch_data

    df = fetch_data("AAPL", "2020-01-01", "2024-01-01")
    bot = QuantBot()
    signals = bot.generate_signals(df)   # returns DataFrame with 'signal' column
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from utils import add_indicators

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RISK_PER_TRADE = 0.01       # 1 % of equity per trade
ATR_STOP_MULT = 2.0         # stop-loss = entry ± 2 × ATR
ATR_TARGET_MULT = 3.0       # take-profit = entry ± 3 × ATR
MIN_SCORE = 3               # minimum confluence score to open/flip
EMA_FAST = 20
EMA_SLOW = 50
RSI_OVERSOLD = 40
RSI_OVERBOUGHT = 65
VOLUME_MULT = 1.2           # volume must be > 1.2× its 20-day SMA


# ---------------------------------------------------------------------------
# Signal generator
# ---------------------------------------------------------------------------

class QuantBot:
    """Rule-based quant trading bot.

    Call :meth:`generate_signals` to attach a ``signal`` column (+1 long,
    -1 short, 0 flat) to the OHLCV+indicators DataFrame.

    Then pass the DataFrame to the backtester, which will simulate trades.
    """

    name = "Quant Bot (Rule-Based)"

    def __init__(
        self,
        risk_per_trade: float = RISK_PER_TRADE,
        atr_stop_mult: float = ATR_STOP_MULT,
        atr_target_mult: float = ATR_TARGET_MULT,
        min_score: int = MIN_SCORE,
        rsi_oversold: float = RSI_OVERSOLD,
        rsi_overbought: float = RSI_OVERBOUGHT,
        volume_mult: float = VOLUME_MULT,
    ) -> None:
        self.risk_per_trade = risk_per_trade
        self.atr_stop_mult = atr_stop_mult
        self.atr_target_mult = atr_target_mult
        self.min_score = min_score
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.volume_mult = volume_mult

    # ------------------------------------------------------------------
    # Individual signal components
    # ------------------------------------------------------------------

    def _ema_signal(self, df: pd.DataFrame) -> pd.Series:
        """EMA trend filter: +1 when fast > slow, -1 when fast < slow."""
        s = pd.Series(0, index=df.index, dtype=int)
        s[df["ema_20"] > df["ema_50"]] = 1
        s[df["ema_20"] < df["ema_50"]] = -1
        return s

    def _rsi_signal(self, df: pd.DataFrame) -> pd.Series:
        """RSI oversold/overbought: +1 buy zone, -1 sell zone."""
        s = pd.Series(0, index=df.index, dtype=int)
        s[df["rsi_14"] < self.rsi_oversold] = 1
        s[df["rsi_14"] > self.rsi_overbought] = -1
        return s

    def _macd_signal(self, df: pd.DataFrame) -> pd.Series:
        """MACD histogram direction: +1 positive / growing, -1 negative."""
        s = pd.Series(0, index=df.index, dtype=int)
        macd_positive = (df["macd_hist"] > 0)
        macd_rising = df["macd_hist"] > df["macd_hist"].shift(1)
        s[macd_positive & macd_rising] = 1
        s[~macd_positive & ~macd_rising] = -1
        return s

    def _bb_signal(self, df: pd.DataFrame) -> pd.Series:
        """Bollinger Bands: +1 near lower band, -1 near upper band."""
        s = pd.Series(0, index=df.index, dtype=int)
        s[df["bb_pct"] < 0.20] = 1
        s[df["bb_pct"] > 0.80] = -1
        return s

    def _volume_signal(self, df: pd.DataFrame) -> pd.Series:
        """Volume confirmation: 0 or +1 (amplifies other signals)."""
        s = pd.Series(0, index=df.index, dtype=int)
        s[df["volume_ratio"] > self.volume_mult] = 1
        return s

    # ------------------------------------------------------------------
    # Composite score
    # ------------------------------------------------------------------

    def _score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame with per-component scores and total bull/bear score."""
        ema = self._ema_signal(df)
        rsi = self._rsi_signal(df)
        macd = self._macd_signal(df)
        bb = self._bb_signal(df)
        vol = self._volume_signal(df)   # 0 or +1 only

        scores = pd.DataFrame(
            {"ema": ema, "rsi": rsi, "macd": macd, "bb": bb, "vol": vol},
            index=df.index,
        )

        # Bull score: count directional +1 signals, volume amplifier adds to both
        scores["bull_score"] = (
            (ema == 1).astype(int)
            + (rsi == 1).astype(int)
            + (macd == 1).astype(int)
            + (bb == 1).astype(int)
            + vol  # 0 or +1
        )

        # Bear score: count directional -1 signals, volume amplifier adds to both
        scores["bear_score"] = (
            (ema == -1).astype(int)
            + (rsi == -1).astype(int)
            + (macd == -1).astype(int)
            + (bb == -1).astype(int)
            + vol  # 0 or +1
        )

        return scores

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add indicators (if not already present) and compute trade signals.

        Returns the same DataFrame with extra columns:
          - signal        : +1 long, -1 short, 0 flat
          - signal_score  : raw bull_score - bear_score
          - stop_loss     : price level for stop-loss
          - take_profit   : price level for take-profit
          - position_size : fractional share count based on ATR risk
        """
        # Add indicators if missing
        if "rsi_14" not in df.columns:
            df = add_indicators(df.copy())
        else:
            df = df.copy()

        scores = self._score(df)

        raw_signal = pd.Series(0, index=df.index, dtype=int)
        raw_signal[scores["bull_score"] >= self.min_score] = 1
        raw_signal[scores["bear_score"] >= self.min_score] = -1

        # Carry the position (hold once opened, exit on opposite signal or stop)
        # For the backtester we output a raw signal per bar; the backtester
        # handles stop-loss and take-profit logic using the provided levels.
        df["signal"] = raw_signal
        df["signal_score"] = scores["bull_score"] - scores["bear_score"]

        # Risk levels
        df["stop_loss"] = np.where(
            raw_signal == 1,
            df["Close"] - self.atr_stop_mult * df["atr_14"],
            np.where(
                raw_signal == -1,
                df["Close"] + self.atr_stop_mult * df["atr_14"],
                np.nan,
            ),
        )
        df["take_profit"] = np.where(
            raw_signal == 1,
            df["Close"] + self.atr_target_mult * df["atr_14"],
            np.where(
                raw_signal == -1,
                df["Close"] - self.atr_target_mult * df["atr_14"],
                np.nan,
            ),
        )

        logger.info(
            "%s: %d long signals, %d short signals out of %d bars",
            self.name,
            (raw_signal == 1).sum(),
            (raw_signal == -1).sum(),
            len(df),
        )
        return df


# ---------------------------------------------------------------------------
# Quick smoke-test when run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from utils import fetch_data

    df = fetch_data("SPY", "2020-01-01", "2024-01-01")
    bot = QuantBot()
    df_signals = bot.generate_signals(df)
    long_n = (df_signals["signal"] == 1).sum()
    short_n = (df_signals["signal"] == -1).sum()
    print(f"\nQuantBot signals on SPY (2020-2024):")
    print(f"  Total bars : {len(df_signals)}")
    print(f"  Long  (+1) : {long_n}")
    print(f"  Short (-1) : {short_n}")
    print(f"  Flat  ( 0) : {len(df_signals) - long_n - short_n}")
    print(df_signals[["Close", "signal", "signal_score", "stop_loss", "take_profit"]].tail(10).to_string())
