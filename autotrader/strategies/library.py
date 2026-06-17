"""Built-in strategy library.

Each strategy is long/flat/short aware and parameterized so the optimizer
can tune it. Signals are *target positions* — the engine adds entry lag.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from autotrader.strategies import indicators as ind
from autotrader.strategies.base import Strategy
from autotrader.strategies.registry import register


@register
class MACrossover(Strategy):
    """Trend-following: long when fast EMA > slow EMA, short otherwise."""

    name = "ma_crossover"
    param_space = {
        "fast": ("int", 5, 50),
        "slow": ("int", 20, 200),
        "allow_short": ("categorical", [True, False]),
    }

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        fast = ind.ema(df["close"], int(self.params["fast"]))
        slow = ind.ema(df["close"], int(self.params["slow"]))
        long = (fast > slow).astype(float)
        if self.params["allow_short"]:
            pos = long * 2 - 1  # -1 / +1
        else:
            pos = long          # 0 / +1
        return pos.where(slow.notna(), 0.0)


@register
class RSIReversion(Strategy):
    """Mean-reversion: long when oversold, short when overbought."""

    name = "rsi_reversion"
    param_space = {
        "period": ("int", 5, 30),
        "oversold": ("int", 15, 40),
        "overbought": ("int", 60, 85),
        "allow_short": ("categorical", [True, False]),
    }

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        r = ind.rsi(df["close"], int(self.params["period"]))
        pos = pd.Series(0.0, index=df.index)
        pos[r < self.params["oversold"]] = 1.0
        if self.params["allow_short"]:
            pos[r > self.params["overbought"]] = -1.0
        else:
            pos[r > self.params["overbought"]] = 0.0
        # hold position until the opposite signal fires
        pos = pos.replace(0.0, np.nan).ffill().fillna(0.0)
        return pos


@register
class Breakout(Strategy):
    """Donchian-style breakout: long on new highs, short on new lows."""

    name = "breakout"
    param_space = {
        "lookback": ("int", 10, 100),
        "allow_short": ("categorical", [True, False]),
    }

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        lb = int(self.params["lookback"])
        upper = df["high"].rolling(lb, min_periods=lb).max().shift(1)
        lower = df["low"].rolling(lb, min_periods=lb).min().shift(1)
        pos = pd.Series(np.nan, index=df.index)
        pos[df["close"] > upper] = 1.0
        pos[df["close"] < lower] = -1.0 if self.params["allow_short"] else 0.0
        return pos.ffill().fillna(0.0)


@register
class BollingerReversion(Strategy):
    """Mean-reversion off Bollinger bands; exits at the midline."""

    name = "bollinger_reversion"
    param_space = {
        "period": ("int", 10, 50),
        "num_std": ("float", 1.5, 3.5),
        "allow_short": ("categorical", [True, False]),
    }

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        lower, mid, upper = ind.bollinger(
            df["close"], int(self.params["period"]), float(self.params["num_std"])
        )
        close = df["close"]
        pos = pd.Series(np.nan, index=df.index)
        pos[close < lower] = 1.0
        pos[close > upper] = -1.0 if self.params["allow_short"] else 0.0
        # flatten when price reverts through the midline
        pos[(close >= mid) & (close.shift(1) < mid)] = 0.0
        pos[(close <= mid) & (close.shift(1) > mid)] = 0.0
        return pos.ffill().fillna(0.0)


@register
class MACDTrend(Strategy):
    """Trend-following on the MACD histogram sign."""

    name = "macd_trend"
    param_space = {
        "fast": ("int", 8, 20),
        "slow": ("int", 21, 60),
        "signal": ("int", 5, 15),
        "allow_short": ("categorical", [True, False]),
    }

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        _, _, hist = ind.macd(
            df["close"], int(self.params["fast"]),
            int(self.params["slow"]), int(self.params["signal"]),
        )
        long = (hist > 0).astype(float)
        pos = long * 2 - 1 if self.params["allow_short"] else long
        return pos.where(hist.notna(), 0.0)


@register
class MomentumROC(Strategy):
    """Cross-sectional-style momentum: long when N-bar return is positive."""

    name = "momentum_roc"
    param_space = {
        "window": ("int", 12, 120),
        "threshold": ("float", 0.0, 0.05),
        "allow_short": ("categorical", [True, False]),
    }

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        r = ind.roc(df["close"], int(self.params["window"]))
        thr = float(self.params["threshold"])
        pos = pd.Series(0.0, index=df.index)
        pos[r > thr] = 1.0
        if self.params["allow_short"]:
            pos[r < -thr] = -1.0
        return pos.where(r.notna(), 0.0)


@register
class Supertrend(Strategy):
    """ATR-based Supertrend trend follower."""

    name = "supertrend"
    param_space = {
        "period": ("int", 7, 30),
        "multiplier": ("float", 1.5, 5.0),
        "allow_short": ("categorical", [True, False]),
    }

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        trend = ind.supertrend(df, int(self.params["period"]), float(self.params["multiplier"]))
        if self.params["allow_short"]:
            return trend
        return trend.clip(lower=0.0)


@register
class StochasticReversion(Strategy):
    """Mean-reversion on the stochastic oscillator."""

    name = "stochastic_reversion"
    param_space = {
        "window": ("int", 7, 30),
        "smooth": ("int", 1, 5),
        "oversold": ("int", 10, 30),
        "overbought": ("int", 70, 90),
        "allow_short": ("categorical", [True, False]),
    }

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        k = ind.stochastic(df, int(self.params["window"]), int(self.params["smooth"]))
        pos = pd.Series(np.nan, index=df.index)
        pos[k < self.params["oversold"]] = 1.0
        pos[k > self.params["overbought"]] = -1.0 if self.params["allow_short"] else 0.0
        return pos.ffill().fillna(0.0)
