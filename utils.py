"""
utils.py – Shared data fetching and technical-indicator helpers.

Used by both quant_bot.py and ai_quant_bot.py.
"""

import warnings
import logging
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
import ta

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def fetch_data(
    symbol: str,
    start: str,
    end: str,
    interval: str = "1d",
) -> pd.DataFrame:
    """Download OHLCV data from Yahoo Finance and return a clean DataFrame.

    Returns columns: Open, High, Low, Close, Volume.
    Raises ValueError if no data is returned.
    """
    logger.info("Fetching %s  %s → %s  (interval=%s)", symbol, start, end, interval)
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start, end=end, interval=interval, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data returned for {symbol} ({start} – {end})")
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.dropna(inplace=True)
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    logger.info("Fetched %d rows for %s", len(df), symbol)
    return df


# ---------------------------------------------------------------------------
# Technical indicators
# ---------------------------------------------------------------------------

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute a comprehensive set of technical indicators and append them.

    The original OHLCV columns are preserved unchanged.
    New columns are added in-place and the dataframe is returned.
    """
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # --- Trend ---
    df["ema_9"] = ta.trend.EMAIndicator(close, window=9).ema_indicator()
    df["ema_20"] = ta.trend.EMAIndicator(close, window=20).ema_indicator()
    df["ema_50"] = ta.trend.EMAIndicator(close, window=50).ema_indicator()
    df["ema_200"] = ta.trend.EMAIndicator(close, window=200).ema_indicator()
    df["sma_20"] = ta.trend.SMAIndicator(close, window=20).sma_indicator()
    df["sma_50"] = ta.trend.SMAIndicator(close, window=50).sma_indicator()

    # --- MACD ---
    macd_obj = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    df["macd"] = macd_obj.macd()
    df["macd_signal"] = macd_obj.macd_signal()
    df["macd_hist"] = macd_obj.macd_diff()

    # --- RSI ---
    df["rsi_14"] = ta.momentum.RSIIndicator(close, window=14).rsi()
    df["rsi_9"] = ta.momentum.RSIIndicator(close, window=9).rsi()

    # --- Stochastic ---
    stoch = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    # --- Bollinger Bands ---
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_middle"] = bb.bollinger_mavg()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_pct"] = bb.bollinger_pband()   # (close - lower) / (upper - lower)
    df["bb_width"] = bb.bollinger_wband() # (upper - lower) / middle

    # --- ATR (volatility) ---
    df["atr_14"] = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()

    # --- Volume ---
    df["vwap_14"] = ta.volume.VolumeWeightedAveragePrice(
        high, low, close, volume, window=14
    ).volume_weighted_average_price()
    df["volume_ratio"] = volume / volume.rolling(20).mean()
    df["obv"] = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()

    # --- Returns ---
    df["return_1d"] = close.pct_change(1)
    df["return_3d"] = close.pct_change(3)
    df["return_5d"] = close.pct_change(5)
    df["return_10d"] = close.pct_change(10)
    df["return_20d"] = close.pct_change(20)

    # --- Lagged returns (used as ML features) ---
    for lag in [1, 2, 3, 5, 10]:
        df[f"lag_return_{lag}d"] = df["return_1d"].shift(lag)

    # --- Price relative to MAs ---
    df["close_vs_ema20"] = (close - df["ema_20"]) / df["ema_20"]
    df["close_vs_ema50"] = (close - df["ema_50"]) / df["ema_50"]
    df["close_vs_ema200"] = (close - df["ema_200"]) / df["ema_200"]

    # --- Momentum score (14-day rate of change) ---
    df["roc_14"] = ta.momentum.ROCIndicator(close, window=14).roc()

    # Calendar features
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month

    df.dropna(inplace=True)
    return df


# ---------------------------------------------------------------------------
# Performance helpers (used by backtester)
# ---------------------------------------------------------------------------

def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.04, periods_per_year: int = 252) -> float:
    """Annualised Sharpe ratio."""
    excess = returns - risk_free_rate / periods_per_year
    if excess.std() == 0:
        return 0.0
    return float((excess.mean() / excess.std()) * np.sqrt(periods_per_year))


def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.04, periods_per_year: int = 252) -> float:
    """Annualised Sortino ratio."""
    excess = returns - risk_free_rate / periods_per_year
    downside = excess[excess < 0].std()
    if downside == 0:
        return 0.0
    return float((excess.mean() / downside) * np.sqrt(periods_per_year))


def max_drawdown(equity_curve: pd.Series) -> float:
    """Maximum peak-to-trough drawdown as a decimal (e.g. -0.25 = -25 %)."""
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    return float(drawdown.min())


def calmar_ratio(returns: pd.Series, equity_curve: pd.Series, periods_per_year: int = 252) -> float:
    """Calmar ratio = annualised return / |max drawdown|."""
    ann_return = (1 + returns.mean()) ** periods_per_year - 1
    mdd = abs(max_drawdown(equity_curve))
    if mdd == 0:
        return 0.0
    return float(ann_return / mdd)
