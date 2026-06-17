"""Vectorized technical indicators (pure pandas/numpy, no TA-Lib dependency)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).mean()


def ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False, min_periods=window).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()


def adx(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Average Directional Index — trend strength (0..100)."""
    high, low = df["high"], df["low"]
    up = high.diff()
    down = -low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = atr(df, window)
    atr_safe = tr.replace(0.0, np.nan)
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(
        alpha=1 / window, adjust=False
    ).mean() / atr_safe
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(
        alpha=1 / window, adjust=False
    ).mean() / atr_safe
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    return dx.ewm(alpha=1 / window, adjust=False).mean().fillna(0.0)


def bollinger(series: pd.Series, window: int, num_std: float):
    mid = sma(series, window)
    sd = series.rolling(window, min_periods=window).std()
    return mid - num_std * sd, mid, mid + num_std * sd


def macd(series: pd.Series, fast: int, slow: int, signal: int):
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return macd_line, signal_line, macd_line - signal_line


def roc(series: pd.Series, window: int) -> pd.Series:
    """Rate of change (momentum) over `window` bars, as a fraction."""
    return series.pct_change(window)


def stochastic(df: pd.DataFrame, window: int, smooth: int):
    low = df["low"].rolling(window, min_periods=window).min()
    high = df["high"].rolling(window, min_periods=window).max()
    k = 100 * (df["close"] - low) / (high - low).replace(0.0, np.nan)
    k = k.fillna(50.0)
    return k.rolling(smooth, min_periods=1).mean()


def supertrend(df: pd.DataFrame, period: int, multiplier: float) -> pd.Series:
    """Return +1 (uptrend) / -1 (downtrend) per bar."""
    hl2 = (df["high"] + df["low"]) / 2
    atr_ = atr(df, period)
    upper = hl2 + multiplier * atr_
    lower = hl2 - multiplier * atr_
    close = df["close"].to_numpy()
    up = upper.to_numpy().copy()
    lo = lower.to_numpy().copy()
    trend = np.ones(len(df))
    for i in range(1, len(df)):
        if close[i] > up[i - 1]:
            trend[i] = 1
        elif close[i] < lo[i - 1]:
            trend[i] = -1
        else:
            trend[i] = trend[i - 1]
            # tighten bands in the direction of the trend
            if trend[i] > 0:
                lo[i] = max(lo[i], lo[i - 1])
            else:
                up[i] = min(up[i], up[i - 1])
    return pd.Series(trend, index=df.index)
