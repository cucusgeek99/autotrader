"""Funding-rate fade — fade crowded perp positioning.

Economic prior (direction fixed a priori, no fitting): when funding is extremely
positive, longs are crowded and paying shorts → fade by going short; when
extremely negative, shorts are crowded → go long. Two return sources:
  1. price mean-reversion of the crowded side, and
  2. the funding carry itself — a short on a high-positive-funding perp RECEIVES
     funding every settlement.

We report both price-only (conservative) and carry-inclusive (realistic). The
whole thing is run through the same purged / nested / panel rigor as everything
else — a structurally different signal still has to clear the same bar.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from autotrader.backtest.metrics import compute
from autotrader.analysis.purged_wf import FoldResult, _aggregate, _pos_mask

FUNDING_PERIOD_BARS = 8  # Binance funds every 8h; on 1h bars that's 8 bars


def funding_zscore(df: pd.DataFrame, window: int) -> pd.Series:
    f = df.get("funding")
    if f is None:
        return pd.Series(0.0, index=df.index)
    mean = f.rolling(window, min_periods=window // 2).mean()
    std = f.rolling(window, min_periods=window // 2).std().replace(0.0, np.nan)
    return ((f - mean) / std).fillna(0.0)


def fade_signal(
    df: pd.DataFrame, lookback: int, threshold: float, hold: int,
    allow_short: bool, direction: int = -1,
) -> pd.Series:
    """Target position from extreme funding z-score.

    direction=-1 (fade, default): SHORT extreme-positive funding, LONG negative.
    direction=+1 (momentum):      LONG extreme-positive funding, SHORT negative.
    """
    z = funding_zscore(df, lookback)
    base = pd.Series(np.nan, index=df.index)
    base[z > threshold] = 1.0
    base[z < -threshold] = -1.0
    pos = direction * base
    if not allow_short:
        pos = pos.clip(lower=0.0)
    return pos.ffill(limit=hold).fillna(0.0)


def fade_backtest(
    df: pd.DataFrame,
    pos: pd.Series,
    taker_fee: float,
    slippage: float,
    bars_per_year: float,
    include_carry: bool,
):
    """Backtest a fade position series. With carry, a held position earns/pays
    funding each bar (per-bar ≈ funding / 8h-period)."""
    positions = pos.shift(1).fillna(0.0)             # enter next bar
    bar_ret = df["close"].pct_change().fillna(0.0)
    gross = positions * bar_ret
    traded = positions.diff().abs().fillna(positions.abs())
    cost = traded * (taker_fee + slippage)
    net = gross - cost
    if include_carry and "funding" in df.columns:
        # short (pos<0) receives funding when funding>0:  carry = -position * funding
        carry = -positions * (df["funding"].fillna(0.0) / FUNDING_PERIOD_BARS)
        net = net + carry
    return compute(net, positions, bars_per_year)


def _eval(df, fwd, lo, hi, lookback, threshold, hold, allow_short, fees, bpy, carry, direction):
    sig = fade_signal(df, lookback, threshold, hold, allow_short, direction)
    m = fade_backtest(df.iloc[lo:hi], sig.iloc[lo:hi],
                      fees["taker"], fees["slippage"], bpy, carry)
    return m


def purged_fade(
    df: pd.DataFrame,
    *,
    horizon: int = 24,
    lookback: int = 168,
    thresholds: tuple = (1.0, 1.5, 2.0, 2.5),
    hold: int | None = None,
    allow_short: bool = True,
    train: int = 1500,
    test: int = 400,
    step: int = 300,
    embargo: int = 24,
    inner_train: int = 800,
    inner_test: int = 300,
    inner_step: int = 300,
    include_carry: bool = True,
    direction: int = -1,
    fees: dict | None = None,
    bars_per_year: float = 8760,
) -> dict:
    """Nested purged walk-forward of the funding fade. The threshold is tuned per
    fold on an inner walk-forward (training data only); the *direction* is fixed
    a priori (chosen on the dev symbol), so there are no fitted weights to leak."""
    fees = fees or {"taker": 0.0006, "slippage": 0.0005}
    hold = hold if hold is not None else horizon
    close = df["close"]
    fwd = close.pct_change(horizon).shift(-horizon)
    n = len(df)
    gap = horizon + embargo

    folds: list[FoldResult] = []
    chosen: list[float] = []
    price_only_returns: list[float] = []
    start = 0
    i = 0
    while start + train + gap + test <= n:
        # inner threshold selection inside the outer-train window only
        best_thr, igap = thresholds[len(thresholds) // 2], horizon + 12
        inner_scores = {t: [] for t in thresholds}
        s = start
        inner_n = 0
        while s + inner_train + igap + inner_test <= start + train:
            ite_lo = s + inner_train + igap
            for thr in thresholds:
                m = _eval(df, fwd, ite_lo, ite_lo + inner_test, lookback, thr,
                          hold, allow_short, fees, bars_per_year, include_carry, direction)
                inner_scores[thr].append(m.score)
            s += inner_step
            inner_n += 1
        if inner_n:
            means = {t: float(np.mean(v)) for t, v in inner_scores.items()}
            best_thr = max(means, key=means.get)

        test_lo = start + train + gap
        m = _eval(df, fwd, test_lo, test_lo + test, lookback, best_thr,
                  hold, allow_short, fees, bars_per_year, include_carry, direction)
        m_price = _eval(df, fwd, test_lo, test_lo + test, lookback, best_thr,
                        hold, allow_short, fees, bars_per_year, False, direction)
        folds.append(FoldResult(
            index=i, weights={"chosen_threshold": best_thr},
            test_score=m.score, test_sharpe=m.sharpe,
            test_return=m.total_return, num_trades=m.num_trades,
        ))
        price_only_returns.append(m_price.total_return)
        chosen.append(best_thr)
        start += step
        i += 1

    out = _aggregate(folds)
    out["chosen_thresholds"] = chosen
    out["mean_price_only_return"] = (
        round(float(np.mean(price_only_returns)), 4) if price_only_returns else 0.0
    )
    return out
