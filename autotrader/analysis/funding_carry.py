"""Funding CARRY harvesting — the structural crypto edge we never tested.

We previously tested funding as a *price predictor* (fade/momentum) — it failed.
This is different: a market-neutral book that tilts to *collect* funding. Short
the high-funding coins (you receive funding from crowded longs), long the
low/negative-funding coins (you receive funding there too), dollar-neutral. The
return has two parts:
  * **carry** — the funding you collect each settlement. Smooth, structural, and
    positive by construction if you stay on the receiving side.
  * **price** — the divergence risk of the long/short legs. Roughly cancels in a
    market-neutral book; this is where the tail risk lives.

This is cash-and-carry's crypto cousin — one of the few genuinely real, if
crowded and decaying, edges. We report carry vs price separately and put the
whole thing through the same OOS + deflated-Sharpe bar. Honest test.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from autotrader.analysis.cross_sectional import apply_band, standardize_xs, to_weights
from autotrader.analysis.cross_validation import _block_returns, deflated_sharpe_ratio

FUNDING_PERIOD_BARS = 8  # 8h funding on 1h bars


def carry_signal(funding: pd.DataFrame, lookback: int = 24) -> pd.DataFrame:
    """Tilt to receive funding: short high-funding coins, long low-funding."""
    sm = funding.ewm(span=lookback, adjust=False, min_periods=lookback // 2).mean()
    return -standardize_xs(sm)


def carry_backtest(
    closes: pd.DataFrame,
    funding: pd.DataFrame,
    signal: pd.DataFrame,
    rebalance: int,
    taker_fee: float,
    slippage: float,
    bars_per_year: float,
    band: float = 0.0,
) -> dict:
    """Backtest the carry book. Returns net + carry/price decomposition + stats."""
    weights = to_weights(signal)
    if rebalance > 1:
        keep = pd.Series(np.arange(len(weights)) % rebalance == 0, index=weights.index)
        weights = weights.where(keep, other=np.nan).ffill().fillna(0.0)
    weights = apply_band(weights, band)
    held = weights.shift(1).fillna(0.0)

    price = (held * closes.pct_change().fillna(0.0)).sum(axis=1)
    # short positive-funding receives: carry_i = -held_i * funding_i (per settlement,
    # spread over the 8-bar period)
    carry = (-held * funding.fillna(0.0) / FUNDING_PERIOD_BARS).sum(axis=1)
    turnover = held.diff().abs().sum(axis=1).fillna(0.0)
    cost = turnover * (taker_fee + slippage)
    net = (price + carry - cost).dropna()

    eq = (1 + net).cumprod()
    mean, std = net.mean(), net.std()
    return {
        "net": net,
        "equity": eq,
        "total_return": round(float(eq.iloc[-1] - 1), 4) if len(eq) else 0.0,
        "sharpe": round(float(mean / std * np.sqrt(bars_per_year)), 3) if std > 0 else 0.0,
        "max_drawdown": round(float((eq / eq.cummax() - 1).min()), 4) if len(eq) else 0.0,
        "carry_contrib": round(float(carry.sum()), 4),     # cumulative carry collected
        "price_contrib": round(float(price.sum()), 4),     # cumulative price P&L
        "cost_drag": round(float(cost.sum()), 4),
        "ann_turnover": round(float(turnover.mean() * bars_per_year), 1),
    }


@dataclass
class CarryOOS:
    oos_returns: pd.Series
    oos_sharpe_ann: float
    oos_total_return: float
    n_folds: int


def walk_forward_carry(
    closes: pd.DataFrame,
    funding: pd.DataFrame,
    *,
    lookback: int = 24,
    rebalance: int = 8,
    band: float = 0.01,
    train: int = 2500,
    test: int = 700,
    step: int = 700,
    embargo: int = 24,
    fees: dict | None = None,
    bars_per_year: float = 8760,
) -> CarryOOS:
    """Walk-forward OOS for the carry book. The signal direction is fixed a
    priori (receive funding), so there are no fitted weights to leak — each test
    window simply harvests with the same rule."""
    fees = fees or {"taker": 0.0006, "slippage": 0.0005}
    sig = carry_signal(funding, lookback)
    n = len(closes)
    gap = rebalance + embargo
    parts = []
    start = 0
    while start + train + gap + test <= n:
        lo = start + train + gap
        hi = lo + test
        res = carry_backtest(closes.iloc[lo:hi], funding.iloc[lo:hi], sig.iloc[lo:hi],
                             rebalance, fees["taker"], fees["slippage"], bars_per_year, band)
        parts.append(res["net"])
        start += step
    if not parts:
        return CarryOOS(pd.Series(dtype=float), 0.0, 0.0, 0)
    oos = pd.concat(parts).dropna()
    blocks = _block_returns(oos, rebalance)
    bsr = float(blocks.mean() / blocks.std()) if blocks.std() > 0 else 0.0
    sharpe = bsr * math.sqrt(bars_per_year / rebalance)
    return CarryOOS(oos, round(sharpe, 3), round(float((1 + oos).prod() - 1), 4), len(parts))
