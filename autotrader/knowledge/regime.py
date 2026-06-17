"""Market regime classification.

The bot's "knowledge" is indexed by regime: it learns which strategy+params
work when the market is trending vs ranging, calm vs volatile. Keeping the
label small (4 buckets) keeps the knowledge base dense enough to be useful.
"""
from __future__ import annotations

import pandas as pd

from autotrader.strategies import indicators as ind

REGIMES = ["trending_up", "trending_down", "ranging_calm", "ranging_volatile"]


def classify(df: pd.DataFrame, window: int = 200, adx_window: int = 14) -> str:
    """Classify the regime of the most recent `window` bars."""
    recent = df.tail(window)
    if len(recent) < adx_window + 2:
        return "ranging_calm"

    trend_strength = float(ind.adx(recent, adx_window).iloc[-1])
    ema_fast = ind.ema(recent["close"], 20).iloc[-1]
    ema_slow = ind.ema(recent["close"], 50).iloc[-1]

    returns = recent["close"].pct_change().dropna()
    vol = float(returns.std())
    # volatility "high" relative to this asset's own recent history
    long_vol = float(df["close"].pct_change().rolling(window).std().median())
    volatile = vol > (long_vol or vol) * 1.3

    if trend_strength >= 25:
        return "trending_up" if ema_fast >= ema_slow else "trending_down"
    return "ranging_volatile" if volatile else "ranging_calm"


def regime_series(
    df: pd.DataFrame,
    adx_window: int = 14,
    vol_window: int = 50,
    base_window: int = 300,
) -> pd.Series:
    """Per-bar regime label, computed causally (every input is trailing).

    Vectorized counterpart to `classify` for use in walk-forward analysis —
    each bar is labelled from only the data available up to that bar.
    """
    adx_s = ind.adx(df, adx_window)
    ema_fast = ind.ema(df["close"], 20)
    ema_slow = ind.ema(df["close"], 50)
    ret = df["close"].pct_change()
    vol = ret.rolling(vol_window, min_periods=vol_window).std()
    base = vol.rolling(base_window, min_periods=vol_window).median()
    volatile = vol > base * 1.3

    out = pd.Series("ranging_calm", index=df.index)
    trending = adx_s >= 25
    up = trending & (ema_fast >= ema_slow)
    down = trending & (ema_fast < ema_slow)
    out[trending & up] = "trending_up"
    out[trending & down] = "trending_down"
    ranging = ~trending
    out[ranging & volatile] = "ranging_volatile"
    out[ranging & ~volatile] = "ranging_calm"
    return out
