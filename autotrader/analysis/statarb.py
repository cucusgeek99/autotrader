"""Statistical arbitrage — cointegration pairs (relative-value mean reversion).

A genuinely different mechanism from factor tilts or carry: find pairs of coins
whose price spread mean-reverts, then trade the spread (long the cheap leg, short
the rich leg) when it diverges, exit when it converges. Market-neutral per pair.

Leak-free protocol: on each training window we estimate the hedge ratio (OLS on
log prices), the spread's mean/std, and its mean-reversion speed (AR(1)
half-life); we select the pairs that mean-revert best *in-sample*; then we trade
those pairs on the *test* window with the train-fixed parameters. Aggregate the
top-K pairs into an equal-weight book and judge it OOS + deflated Sharpe.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from autotrader.analysis.cross_validation import _block_returns, deflated_sharpe_ratio


def _ols_beta(y: np.ndarray, x: np.ndarray) -> float:
    vx = x.var()
    return float(np.cov(y, x)[0, 1] / vx) if vx > 0 else 0.0


def _half_life(spread: np.ndarray) -> float:
    s = spread[:-1]
    ds = np.diff(spread)
    lam = _ols_beta(ds, s)  # Δs = lam * s_lag
    return float(-math.log(2) / lam) if lam < 0 else np.inf


def _z_positions(z: np.ndarray, entry: float, exit: float) -> np.ndarray:
    pos = np.zeros(len(z))
    cur = 0
    for t in range(len(z)):
        if cur == 0:
            if z[t] > entry:
                cur = -1          # spread rich → short spread
            elif z[t] < -entry:
                cur = 1           # spread cheap → long spread
        elif abs(z[t]) < exit:
            cur = 0
        pos[t] = cur
    return pos


@dataclass
class PairParams:
    i: str
    j: str
    beta: float
    mean: float
    std: float
    half_life: float
    train_sharpe: float


def _spread(closes: pd.DataFrame, i: str, j: str, beta: float) -> pd.Series:
    return np.log(closes[i]) - beta * np.log(closes[j])


def pair_net(closes: pd.DataFrame, p: PairParams, entry: float, exit: float,
             fee: float, slip: float) -> pd.Series:
    """Net return series of one pair over `closes`, using train-fixed params."""
    spread = _spread(closes, p.i, p.j, p.beta)
    z = ((spread - p.mean) / p.std).to_numpy()
    pos = pd.Series(_z_positions(z, entry, exit), index=closes.index).shift(1).fillna(0.0)
    spread_ret = closes[p.i].pct_change().fillna(0.0) - p.beta * closes[p.j].pct_change().fillna(0.0)
    gross = pos * spread_ret
    turn = pos.diff().abs().fillna(pos.abs()) * (1 + abs(p.beta))
    return gross - turn * (fee + slip)


def select_pairs(closes_train: pd.DataFrame, k: int, entry: float, exit: float,
                 fees: dict, bpy: float, hl_range=(5, 400)) -> list:
    """Rank all pairs by in-sample mean-reversion quality; return top-k params."""
    syms = list(closes_train.columns)
    cands = []
    for a in range(len(syms)):
        for b in range(a + 1, len(syms)):
            i, j = syms[a], syms[b]
            yi, yj = np.log(closes_train[i].to_numpy()), np.log(closes_train[j].to_numpy())
            beta = _ols_beta(yi, yj)
            if beta <= 0:
                continue
            spread = yi - beta * yj
            hl = _half_life(spread)
            if not (hl_range[0] <= hl <= hl_range[1]):
                continue
            mean, std = float(spread.mean()), float(spread.std())
            if std <= 0:
                continue
            p = PairParams(i, j, beta, mean, std, hl, 0.0)
            net = pair_net(closes_train, p, entry, exit, fees["taker"], fees["slippage"]).dropna()
            if net.std() > 0 and net.abs().sum() > 0:
                p.train_sharpe = float(net.mean() / net.std() * math.sqrt(bpy))
                cands.append(p)
    cands.sort(key=lambda p: p.train_sharpe, reverse=True)
    return cands[:k]


def book_net(closes: pd.DataFrame, pairs: list, entry: float, exit: float,
             fees: dict) -> pd.Series:
    """Equal-weight portfolio of pair returns."""
    if not pairs:
        return pd.Series(0.0, index=closes.index)
    series = [pair_net(closes, p, entry, exit, fees["taker"], fees["slippage"]) for p in pairs]
    return pd.concat(series, axis=1).mean(axis=1)


@dataclass
class StatArbOOS:
    oos_returns: pd.Series
    oos_sharpe_ann: float
    oos_total_return: float
    n_folds: int
    avg_pairs: float


def walk_forward_statarb(
    closes: pd.DataFrame, *, k: int = 8, entry: float = 2.0, exit: float = 0.5,
    train: int = 2500, test: int = 700, step: int = 700, embargo: int = 24,
    fees: dict | None = None, bars_per_year: float = 8760,
) -> StatArbOOS:
    fees = fees or {"taker": 0.0006, "slippage": 0.0005}
    n = len(closes)
    gap = embargo
    parts, npairs = [], []
    start = 0
    while start + train + gap + test <= n:
        tr = closes.iloc[start:start + train]
        pairs = select_pairs(tr, k, entry, exit, fees, bars_per_year)
        npairs.append(len(pairs))
        lo = start + train + gap
        te = closes.iloc[lo:lo + test]
        parts.append(book_net(te, pairs, entry, exit, fees))
        start += step
    if not parts:
        return StatArbOOS(pd.Series(dtype=float), 0.0, 0.0, 0, 0.0)
    oos = pd.concat(parts).dropna()
    blocks = _block_returns(oos, 24)
    bsr = float(blocks.mean() / blocks.std()) if blocks.std() > 0 else 0.0
    sharpe = bsr * math.sqrt(bars_per_year / 24)
    return StatArbOOS(oos, round(sharpe, 3), round(float((1 + oos).prod() - 1), 4),
                      len(parts), round(float(np.mean(npairs)), 1))
