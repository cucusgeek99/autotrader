"""The core book — the real-capital strategy (spot, long-only, no leverage, daily).

After the whole research arc (no candle-based alpha survived out-of-sample, see
SUMMARY.md), the strategy that actually helps a trader keep and grow money is not
a secret edge but disciplined risk-taking on the market itself:

  weight_i = (1/N) × min(1, TARGET_VOL / realized_vol_i) × [close_i > SMA_i]

  * vol-targeting  — size down when a coin gets wild, so one crash can't wreck you
  * SMA200 filter  — sit in cash (USDT) while a coin is in a long-term downtrend
  * equal sleeves  — 1/N per coin, leftover stays in stablecoins

Textbook parameters, NOT tuned: a 36-combo grid (SMA 100-250 × vol 30-60% ×
window 20-60) gives Sharpe 0.81-1.15 and maxDD -22%..-55% on BTC/ETH 2017-2026,
vs buy&hold Sharpe 0.77 / maxDD -88% — a plateau, not a lucky point.
Cost of the insurance: it lags buy&hold in a straight-up bull market.

Decisions use the LAST CLOSED daily candle only (no look-ahead).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CORE_UNIVERSE = ["BTC/USDT", "ETH/USDT"]
TARGET_VOL = 0.40      # annualized vol each sleeve is scaled to
VOL_WINDOW = 30        # days of returns for realized vol
SMA_WINDOW = 200       # long-term trend filter
FEE = 0.001            # spot taker fee per side (0.1%)
BAND = 0.02            # skip trades smaller than 2% of equity (fee saver)


def core_weights(closes: pd.DataFrame, target_vol: float = TARGET_VOL,
                 vol_window: int = VOL_WINDOW, sma_window: int = SMA_WINDOW,
                 periods_per_year: int = 365) -> pd.DataFrame:
    """Target weight per coin for each bar, using data up to and including that bar.

    Rows sum to ≤ 1 (the rest is cash). NaN warm-up rows come out as 0 weight.
    """
    rets = closes.pct_change()
    vol = rets.rolling(vol_window).std() * np.sqrt(periods_per_year)
    scale = (target_vol / vol).clip(upper=1.0)
    trend = (closes > closes.rolling(sma_window).mean()).astype(float)
    w = scale * trend / closes.shape[1]
    return w.fillna(0.0)


def core_returns(closes: pd.DataFrame, fee: float = FEE, **kw) -> pd.Series:
    """Net daily returns of the core book (weights decided on yesterday's close)."""
    w = core_weights(closes, **kw).shift(1).fillna(0.0)
    gross = (w * closes.pct_change().fillna(0.0)).sum(axis=1)
    cost = w.diff().abs().fillna(0.0).sum(axis=1) * fee
    return gross - cost


def summarize(rets: pd.Series, periods_per_year: int = 365) -> dict:
    """CAGR / Sharpe / maxDD / worst calendar year for a daily return series."""
    eq = (1 + rets).cumprod()
    years = len(rets) / periods_per_year
    yearly = eq.resample("YE").last()
    yr = yearly.pct_change()
    yr.iloc[0] = yearly.iloc[0] - 1
    sd = rets.std()
    return {
        "cagr": float(eq.iloc[-1] ** (1 / years) - 1) if years > 0 else 0.0,
        "sharpe": float(rets.mean() / sd * np.sqrt(periods_per_year)) if sd > 0 else 0.0,
        "max_dd": float((eq / eq.cummax() - 1).min()),
        "worst_year": float(yr.min()),
        "multiple": float(eq.iloc[-1]),
    }


def closed_daily(df: pd.DataFrame, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Drop today's still-forming daily candle so decisions only see closed bars."""
    today = (now or pd.Timestamp.now(tz="UTC")).normalize()
    if df.index.tz is None and today.tz is not None:
        today = today.tz_localize(None)
    elif df.index.tz is not None and today.tz is None:
        today = today.tz_localize("UTC")
    return df[df.index < today]
