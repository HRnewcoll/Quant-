"""
ai_quant_bot.py – AI / Machine-Learning Quantitative Trading Bot
================================================================

Strategy overview
-----------------
This bot uses a walk-forward ensemble of gradient-boosted trees
(HistGradientBoostingClassifier) plus a Random Forest to generate
probabilistic trade signals.

Walk-forward design (avoids look-ahead bias)
--------------------------------------------
  * Train window : 252 trading days (~1 year)
  * Predict step : 63 trading days (~1 quarter)
  * Re-train every step so the model continuously adapts.

Features (30+)
--------------
  Returns : 1d, 3d, 5d, 10d, 20d and their lags
  Trend   : close vs EMA-20/50/200, EMA-20/50 spread
  Momentum: RSI-9, RSI-14, MACD histogram, ROC-14, Stochastic %K/%D
  Vol     : BB %width, BB %B, ATR/close ratio
  Volume  : volume_ratio, OBV normalised
  Calendar: day_of_week, month

Label
-----
  1 (buy) if the next-bar forward return exceeds +THRESHOLD
  0 (hold/sell) otherwise

A trade is placed when the ensemble probability ≥ 0.60 (long) or
≤ 0.40 (short).

Usage
-----
    from ai_quant_bot import AIQuantBot
    from utils import fetch_data

    df = fetch_data("AAPL", "2018-01-01", "2024-01-01")
    bot = AIQuantBot()
    df_with_signals = bot.generate_signals(df)
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from utils import add_indicators

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TRAIN_WINDOW = 252          # bars used for training
PREDICT_STEP = 63           # bars between re-trains (rolling window)
BUY_PROB_THRESHOLD = 0.58   # ensemble probability ≥ this → long
SELL_PROB_THRESHOLD = 0.42  # ensemble probability ≤ this → short
# The asymmetric thresholds create a ±0.08 dead-zone around 0.50, biasing
# the bot toward cash when the model lacks conviction.  This reduces
# overtrading on borderline signals and is intentional.
LABEL_THRESHOLD = 0.003     # next-bar return > 0.3 % → label=1


# ---------------------------------------------------------------------------
# Feature list
# ---------------------------------------------------------------------------
FEATURES: List[str] = [
    # Returns
    "return_1d", "return_3d", "return_5d", "return_10d", "return_20d",
    # Lagged returns
    "lag_return_1d", "lag_return_2d", "lag_return_3d", "lag_return_5d", "lag_return_10d",
    # Trend
    "close_vs_ema20", "close_vs_ema50", "close_vs_ema200",
    # Momentum
    "rsi_9", "rsi_14", "macd_hist", "roc_14", "stoch_k", "stoch_d",
    # Volatility
    "bb_pct", "bb_width", "atr_ratio",
    # Volume
    "volume_ratio",
    # Calendar
    "day_of_week", "month",
]


# ---------------------------------------------------------------------------
# Helper: add ATR-ratio feature (not in utils)
# ---------------------------------------------------------------------------

def _add_atr_ratio(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["atr_ratio"] = df["atr_14"] / df["Close"]
    return df


# ---------------------------------------------------------------------------
# Walk-forward ML bot
# ---------------------------------------------------------------------------

class AIQuantBot:
    """AI quantitative trading bot using a walk-forward ensemble classifier.

    Call :meth:`generate_signals` to train the walk-forward models and
    attach ``signal`` (+1/−1/0) and ``pred_proba`` columns to the DataFrame.
    """

    name = "AI Quant Bot (ML Ensemble)"

    def __init__(
        self,
        train_window: int = TRAIN_WINDOW,
        predict_step: int = PREDICT_STEP,
        buy_threshold: float = BUY_PROB_THRESHOLD,
        sell_threshold: float = SELL_PROB_THRESHOLD,
        label_threshold: float = LABEL_THRESHOLD,
        n_estimators: int = 200,
        random_state: int = 42,
    ) -> None:
        self.train_window = train_window
        self.predict_step = predict_step
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.label_threshold = label_threshold
        self.n_estimators = n_estimators
        self.random_state = random_state
        self._model: Optional[Pipeline] = None

    # ------------------------------------------------------------------
    # Feature / label preparation
    # ------------------------------------------------------------------

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all indicators and atr_ratio; return the enriched DataFrame."""
        if "rsi_14" not in df.columns:
            df = add_indicators(df.copy())
        else:
            df = df.copy()
        df = _add_atr_ratio(df)
        return df

    def _make_labels(self, df: pd.DataFrame) -> pd.Series:
        """Binary label: 1 if next bar's return > label_threshold, else 0."""
        fwd = df["Close"].pct_change(1).shift(-1)
        return (fwd > self.label_threshold).astype(int)

    # ------------------------------------------------------------------
    # Model factory
    # ------------------------------------------------------------------

    def _build_model(self) -> Pipeline:
        """Return a calibrated voting ensemble wrapped in a sklearn Pipeline."""
        rf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=6,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=self.random_state,
        )
        gb = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_leaf=5,
            random_state=self.random_state,
        )
        ensemble = VotingClassifier(
            estimators=[("rf", rf), ("gb", gb)],
            voting="soft",
        )
        calibrated = CalibratedClassifierCV(ensemble, method="isotonic", cv=3)
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", calibrated),
        ])
        return pipeline

    # ------------------------------------------------------------------
    # Walk-forward engine
    # ------------------------------------------------------------------

    def _walk_forward(
        self,
        feature_matrix: pd.DataFrame,
        labels: pd.Series,
    ) -> Tuple[pd.Series, pd.Series]:
        """Run walk-forward training and return (signal, pred_proba) Series."""
        n = len(feature_matrix)
        signals = pd.Series(0, index=feature_matrix.index, dtype=int)
        probas = pd.Series(np.nan, index=feature_matrix.index)

        n_folds = 0
        for start in range(self.train_window, n - self.predict_step + 1, self.predict_step):
            train_idx = range(start - self.train_window, start)
            pred_idx = range(start, min(start + self.predict_step, n))

            X_train = feature_matrix.iloc[train_idx]
            y_train = labels.iloc[train_idx]
            X_pred = feature_matrix.iloc[pred_idx]

            # Skip fold if only one class present
            if y_train.nunique() < 2:
                continue

            model = self._build_model()
            try:
                model.fit(X_train, y_train)
            except Exception as exc:
                logger.warning("Walk-forward fold skipped (%s)", exc)
                continue

            proba = model.predict_proba(X_pred)[:, 1]
            idx = feature_matrix.index[list(pred_idx)]
            probas.loc[idx] = proba
            signals.loc[idx] = np.where(
                proba >= self.buy_threshold, 1,
                np.where(proba <= self.sell_threshold, -1, 0),
            )
            n_folds += 1

        logger.info("%s: completed %d walk-forward folds", self.name, n_folds)
        return signals, probas

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features, run walk-forward training, attach signals.

        Returns the enriched DataFrame with additional columns:
          - signal     : +1 long, -1 short, 0 flat
          - pred_proba : ensemble probability of an up-move
        """
        df = self._prepare(df)

        # Subset to available features (drop any that are missing)
        available_feats = [f for f in FEATURES if f in df.columns]
        missing = set(FEATURES) - set(available_feats)
        if missing:
            logger.warning("Missing features (ignored): %s", missing)

        feature_matrix = df[available_feats].replace([np.inf, -np.inf], np.nan).ffill(limit=5).dropna()
        labels = self._make_labels(df).loc[feature_matrix.index]

        # Align indices
        feature_matrix, labels = feature_matrix.align(labels, join="inner", axis=0)
        feature_matrix.dropna(inplace=True)
        labels = labels.loc[feature_matrix.index]

        signals, probas = self._walk_forward(feature_matrix, labels)

        df["signal"] = signals.reindex(df.index).fillna(0).astype(int)
        df["pred_proba"] = probas.reindex(df.index)

        long_n = (df["signal"] == 1).sum()
        short_n = (df["signal"] == -1).sum()
        logger.info(
            "%s: %d long signals, %d short signals out of %d bars",
            self.name,
            long_n,
            short_n,
            len(df),
        )
        return df


# ---------------------------------------------------------------------------
# Quick smoke-test when run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from utils import fetch_data

    df = fetch_data("SPY", "2018-01-01", "2024-01-01")
    bot = AIQuantBot()
    print("Training AI bot on SPY 2018-2024 (this may take ~30 s)…")
    df_signals = bot.generate_signals(df)
    long_n = (df_signals["signal"] == 1).sum()
    short_n = (df_signals["signal"] == -1).sum()
    print(f"\nAIQuantBot signals on SPY (2018-2024):")
    print(f"  Total bars : {len(df_signals)}")
    print(f"  Long  (+1) : {long_n}")
    print(f"  Short (-1) : {short_n}")
    print(f"  Flat  ( 0) : {len(df_signals) - long_n - short_n}")
    print(df_signals[["Close", "signal", "pred_proba"]].tail(10).to_string())
