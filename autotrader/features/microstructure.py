"""Microstructure / order-flow features computed from extended klines.

All features are causal: they use only data available at each bar's close, so a
strategy that positions on the *next* bar (the engine adds that lag) has no
look-ahead. Each feature returns a pandas Series aligned to df.index.

Requires the extended columns from `fetch_ohlcv_extended`:
  taker_buy_base, quote_volume, num_trades. `has_flow(df)` tells you if they're
present; feature functions return NaN-filled neutral series if not.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FLOW_COLUMNS = ("taker_buy_base", "quote_volume", "num_trades")


def has_flow(df: pd.DataFrame) -> bool:
    return all(c in df.columns for c in ("taker_buy_base", "volume"))


def order_flow_imbalance(df: pd.DataFrame, window: int = 1) -> pd.Series:
    """Aggressor imbalance in [-1, 1]: (taker_buy - taker_sell) / total volume,
    summed over `window` bars. +1 = all aggressive buying, -1 = all selling.

    This is the single most informative backtest-able order-flow signal we can
    get from free historical data — it measures who is *crossing the spread*.
    """
    if not has_flow(df):
        return pd.Series(0.0, index=df.index)
    buy = df["taker_buy_base"]
    sell = df["volume"] - buy
    net = (buy - sell).rolling(window, min_periods=1).sum()
    tot = df["volume"].rolling(window, min_periods=1).sum().replace(0.0, np.nan)
    return (net / tot).clip(-1, 1).fillna(0.0)


def vwap(df: pd.DataFrame, window: int) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    pv = (tp * df["volume"]).rolling(window, min_periods=window).sum()
    v = df["volume"].rolling(window, min_periods=window).sum().replace(0.0, np.nan)
    return pv / v


def vwap_deviation(df: pd.DataFrame, window: int = 24) -> pd.Series:
    """Fractional distance of price from rolling VWAP — a fair-value gap proxy."""
    vw = vwap(df, window)
    return ((df["close"] - vw) / vw).fillna(0.0)


def amihud_illiquidity(df: pd.DataFrame, window: int = 24) -> pd.Series:
    """|return| per unit quote volume — high = thin/illiquid, prone to impact."""
    ret = df["close"].pct_change().abs()
    qv = df.get("quote_volume", df["close"] * df["volume"]).replace(0.0, np.nan)
    return (ret / qv).rolling(window, min_periods=1).mean().fillna(0.0)


def realized_vol(df: pd.DataFrame, window: int = 24) -> pd.Series:
    return df["close"].pct_change().rolling(window, min_periods=2).std().fillna(0.0)


def trade_intensity(df: pd.DataFrame, window: int = 24) -> pd.Series:
    """z-score of trade count — bursts of activity often precede moves."""
    if "num_trades" not in df.columns:
        return pd.Series(0.0, index=df.index)
    nt = df["num_trades"].astype(float)
    mean = nt.rolling(window, min_periods=window).mean()
    std = nt.rolling(window, min_periods=window).std().replace(0.0, np.nan)
    return ((nt - mean) / std).fillna(0.0)


def signed_flow_momentum(df: pd.DataFrame, window: int = 12) -> pd.Series:
    """Smoothed cumulative order-flow imbalance — flow trend, not price trend."""
    ofi = order_flow_imbalance(df, window=1)
    return ofi.ewm(span=window, adjust=False, min_periods=window).mean().fillna(0.0)


def feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """All standard features as a DataFrame — used by the IC analyzer."""
    return pd.DataFrame(
        {
            "ofi_1": order_flow_imbalance(df, 1),
            "ofi_6": order_flow_imbalance(df, 6),
            "flow_mom_12": signed_flow_momentum(df, 12),
            "vwap_dev_24": vwap_deviation(df, 24),
            "amihud_24": amihud_illiquidity(df, 24),
            "rvol_24": realized_vol(df, 24),
            "trade_z_24": trade_intensity(df, 24),
        },
        index=df.index,
    )
