"""Out-of-sample validation for cross-sectional factors.

Two tools, both standard in serious quant research:

1. Walk-forward OOS — for each fold, select the factor's lookback on the TRAIN
   window only, purge+embargo the boundary, then measure rank IC and the
   long-short net return on the TEST window. OOS results are stitched from test
   windows the selection never saw.

2. Deflated Sharpe Ratio (Bailey & López de Prado, 2014) — haircuts the observed
   Sharpe for (a) the number of strategy configurations tried (selection bias)
   and (b) the skew/kurtosis of the returns. Returns the probability the true
   Sharpe is > 0 after that haircut. This is the multiple-testing correction.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from autotrader.analysis.cross_sectional import (
    ic_series, liquidity_mask, screen_signal, xs_backtest, xs_net,
)

_EULER = 0.5772156649015329


# ---- normal CDF / inverse-CDF (no scipy dependency) ---------------------
def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Acklam's rational approximation of the inverse normal CDF."""
    if p <= 0.0:
        return -np.inf
    if p >= 1.0:
        return np.inf
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def deflated_sharpe_ratio(
    returns: pd.Series, n_trials: int, trial_sharpes: list[float] | None = None
) -> dict:
    """Probability the true (per-observation) Sharpe > 0 after deflation.

    `returns` should be non-overlapping period returns. `n_trials` is how many
    configurations were tried; `trial_sharpes` (per-observation Sharpes of all
    trials) sets the null's dispersion — if omitted, a default spread is assumed.
    """
    r = returns.dropna()
    T = len(r)
    if T < 10 or r.std() == 0:
        return {"dsr": 0.0, "sr": 0.0, "sr0": 0.0, "T": T}
    sr = float(r.mean() / r.std())
    skew = float(r.skew())
    kurt = float(r.kurt() + 3.0)  # pandas .kurt() is excess; DSR wants full

    if trial_sharpes and len(trial_sharpes) > 1:
        var_trials = float(np.var(trial_sharpes, ddof=1))
    else:
        var_trials = sr * sr  # conservative fallback
    std_trials = math.sqrt(max(var_trials, 1e-12))

    n = max(n_trials, 1)
    # expected maximum Sharpe under the null given n trials
    z1 = _norm_ppf(1 - 1.0 / n)
    z2 = _norm_ppf(1 - 1.0 / (n * math.e))
    sr0 = std_trials * ((1 - _EULER) * z1 + _EULER * z2)

    denom = math.sqrt(max(1 - skew * sr + (kurt - 1) / 4.0 * sr * sr, 1e-9))
    dsr = _norm_cdf((sr - sr0) * math.sqrt(T - 1) / denom)
    return {"dsr": round(dsr, 4), "sr": round(sr, 4), "sr0": round(sr0, 4), "T": T}


# ---- walk-forward OOS for a cross-sectional factor ----------------------
@dataclass
class XSWalkForward:
    oos_returns: pd.Series      # stitched non-overlapping-friendly net returns (per bar)
    oos_ic_mean: float
    oos_ic_t: float
    oos_sharpe_ann: float
    oos_total_return: float
    chosen_lookbacks: list
    n_folds: int


def walk_forward_xs(
    builder,
    closes: pd.DataFrame,
    *,
    horizon: int,
    lookback_grid: list[int],
    train: int = 2500,
    test: int = 700,
    step: int = 700,
    embargo: int = 24,
    fees: dict | None = None,
    bars_per_year: float = 8760,
) -> XSWalkForward:
    """`builder(lookback) -> signal panel`. Lookback chosen on train (by train
    long-short Sharpe), evaluated on the purged+embargoed test window."""
    fees = fees or {"taker": 0.0006, "slippage": 0.0005}
    n = len(closes)
    gap = horizon + embargo
    oos_ret_parts, oos_ic_parts, chosen = [], [], []
    start = 0
    while start + train + gap + test <= n:
        tr = closes.iloc[start:start + train]
        # select lookback on train only
        best_lb, best_sr = lookback_grid[len(lookback_grid) // 2], -1e9
        for lb in lookback_grid:
            sig_tr = builder(lb).iloc[start:start + train]
            ps = xs_backtest(sig_tr, tr, horizon, fees["taker"], fees["slippage"],
                             bars_per_year)
            if ps.sharpe > best_sr:
                best_sr, best_lb = ps.sharpe, lb
        chosen.append(best_lb)

        test_lo = start + train + gap
        test_hi = test_lo + test
        te = closes.iloc[test_lo:test_hi]
        sig_full = builder(best_lb)
        sig_te = sig_full.iloc[test_lo:test_hi]
        # collect the test net-return series (recompute inline to keep the series)
        w = _held_weights(sig_te, horizon)
        rets = te.pct_change().fillna(0.0)
        port = (w * rets).sum(axis=1)
        turn = w.diff().abs().sum(axis=1).fillna(0.0)
        net = port - turn * (fees["taker"] + fees["slippage"])
        oos_ret_parts.append(net)
        oos_ic_parts.append(ic_series(sig_full, closes, horizon).iloc[test_lo:test_hi])
        start += step

    if not oos_ret_parts:
        return XSWalkForward(pd.Series(dtype=float), 0.0, 0.0, 0.0, 0.0, [], 0)

    oos = pd.concat(oos_ret_parts).dropna()
    oos_ic = pd.concat(oos_ic_parts).dropna()
    # block (non-overlapping) returns for honest Sharpe / t-stat
    blocks = _block_returns(oos, horizon)
    bsr = float(blocks.mean() / blocks.std()) if blocks.std() > 0 else 0.0
    blocks_per_year = bars_per_year / horizon
    sharpe_ann = bsr * math.sqrt(blocks_per_year)
    total = float((1 + oos).prod() - 1)

    ic_nonover = oos_ic.iloc[::horizon]
    ic_t = (float(ic_nonover.mean() / (ic_nonover.std(ddof=1) / math.sqrt(len(ic_nonover))))
            if len(ic_nonover) > 1 and ic_nonover.std() > 0 else 0.0)

    return XSWalkForward(
        oos_returns=oos,
        oos_ic_mean=round(float(oos_ic.mean()), 4),
        oos_ic_t=round(ic_t, 2),
        oos_sharpe_ann=round(sharpe_ann, 3),
        oos_total_return=round(total, 4),
        chosen_lookbacks=chosen,
        n_folds=len(oos_ret_parts),
    )


def walk_forward_composite(
    factor_panels: list,
    closes: pd.DataFrame,
    *,
    horizon: int,
    train: int = 2500,
    test: int = 700,
    step: int = 700,
    embargo: int = 24,
    band: float = 0.0,
    dollar_vol: pd.DataFrame | None = None,
    top_n: int | None = None,
    screen_lookback: int = 336,
    fees: dict | None = None,
    bars_per_year: float = 8760,
) -> XSWalkForward:
    """Cost-aware multi-factor walk-forward. Each fold sets combination weights
    by INVERSE TURNOVER measured on the train window only (an a-priori rule using
    a signal property, not returns), so the expensive factor gets a smaller,
    cost-justified allocation. Composite then evaluated on the purged test window.
    """
    from autotrader.analysis.cross_sectional import combine_factors, xs_backtest

    fees = fees or {"taker": 0.0006, "slippage": 0.0005}
    n = len(closes)
    gap = horizon + embargo
    oos_ret_parts, oos_ic_parts, chosen = [], [], []
    start = 0
    while start + train + gap + test <= n:
        tr = closes.iloc[start:start + train]
        inv = []
        for p in factor_panels:
            turn = xs_backtest(p.iloc[start:start + train], tr, horizon,
                               fees["taker"], fees["slippage"], bars_per_year).ann_turnover
            inv.append(1.0 / max(turn, 1.0))
        s = sum(inv) or 1.0
        weights = [x / s for x in inv]
        composite = combine_factors(factor_panels, weights)
        if top_n and dollar_vol is not None:
            composite = screen_signal(composite, liquidity_mask(dollar_vol, top_n, screen_lookback))

        test_lo = start + train + gap
        test_hi = test_lo + test
        te = closes.iloc[test_lo:test_hi]
        net, _ = xs_net(composite.iloc[test_lo:test_hi], te, horizon,
                        fees["taker"], fees["slippage"], band)
        oos_ret_parts.append(net)
        oos_ic_parts.append(ic_series(composite, closes, horizon).iloc[test_lo:test_hi])
        chosen.append([round(x, 2) for x in weights])
        start += step

    if not oos_ret_parts:
        return XSWalkForward(pd.Series(dtype=float), 0.0, 0.0, 0.0, 0.0, [], 0)
    oos = pd.concat(oos_ret_parts).dropna()
    oos_ic = pd.concat(oos_ic_parts).dropna()
    blocks = _block_returns(oos, horizon)
    bsr = float(blocks.mean() / blocks.std()) if blocks.std() > 0 else 0.0
    sharpe_ann = bsr * math.sqrt(bars_per_year / horizon)
    ic_nonover = oos_ic.iloc[::horizon]
    ic_t = (float(ic_nonover.mean() / (ic_nonover.std(ddof=1) / math.sqrt(len(ic_nonover))))
            if len(ic_nonover) > 1 and ic_nonover.std() > 0 else 0.0)
    return XSWalkForward(oos, round(float(oos_ic.mean()), 4), round(ic_t, 2),
                         round(sharpe_ann, 3), round(float((1 + oos).prod() - 1), 4),
                         chosen, len(oos_ret_parts))


def walk_forward_nested_universe(
    factor_panels: list,
    closes: pd.DataFrame,
    dollar_vol: pd.DataFrame,
    *,
    horizon: int,
    n_grid: tuple = (6, 8, 10, 12),
    train: int = 2500,
    test: int = 700,
    step: int = 700,
    embargo: int = 24,
    band: float = 0.0,
    screen_lookback: int = 336,
    fees: dict | None = None,
    bars_per_year: float = 8760,
) -> XSWalkForward:
    """Nested universe-size selection. Each fold picks the universe size N from
    `n_grid` using ONLY its training window (by train Sharpe), then applies that N
    to the purged test window. N is never chosen on test data — so the OOS number
    is free of the universe-size peek. `chosen_lookbacks` carries the per-fold N."""
    from autotrader.analysis.cross_sectional import (
        combine_factors, liquidity_mask, screen_signal, xs_backtest,
    )
    fees = fees or {"taker": 0.0006, "slippage": 0.0005}
    n = len(closes)
    gap = horizon + embargo
    parts, ic_parts, chosen = [], [], []
    start = 0
    while start + train + gap + test <= n:
        tr_lo, tr_hi = start, start + train
        # cost-aware factor weights from train
        inv = []
        for p in factor_panels:
            turn = xs_backtest(p.iloc[tr_lo:tr_hi], closes.iloc[tr_lo:tr_hi], horizon,
                               fees["taker"], fees["slippage"], bars_per_year).ann_turnover
            inv.append(1.0 / max(turn, 1.0))
        s = sum(inv) or 1.0
        composite = combine_factors(factor_panels, [x / s for x in inv])

        # inner: pick N by TRAIN Sharpe only
        best_n, best_sr = n_grid[len(n_grid) // 2], -1e9
        for N in n_grid:
            cs = screen_signal(composite, liquidity_mask(dollar_vol, N, screen_lookback))
            net_tr, _ = xs_net(cs.iloc[tr_lo:tr_hi], closes.iloc[tr_lo:tr_hi], horizon,
                               fees["taker"], fees["slippage"], band)
            net_tr = net_tr.dropna()
            sr = float(net_tr.mean() / net_tr.std()) if net_tr.std() > 0 else -1e9
            if sr > best_sr:
                best_sr, best_n = sr, N
        chosen.append(best_n)

        cs = screen_signal(composite, liquidity_mask(dollar_vol, best_n, screen_lookback))
        lo = start + train + gap
        hi = lo + test
        net, _ = xs_net(cs.iloc[lo:hi], closes.iloc[lo:hi], horizon,
                        fees["taker"], fees["slippage"], band)
        parts.append(net)
        ic_parts.append(ic_series(cs, closes, horizon).iloc[lo:hi])
        start += step

    if not parts:
        return XSWalkForward(pd.Series(dtype=float), 0.0, 0.0, 0.0, 0.0, [], 0)
    oos = pd.concat(parts).dropna()
    oos_ic = pd.concat(ic_parts).dropna()
    blocks = _block_returns(oos, horizon)
    bsr = float(blocks.mean() / blocks.std()) if blocks.std() > 0 else 0.0
    sharpe = bsr * math.sqrt(bars_per_year / horizon)
    ic_nonover = oos_ic.iloc[::horizon]
    ic_t = (float(ic_nonover.mean() / (ic_nonover.std(ddof=1) / math.sqrt(len(ic_nonover))))
            if len(ic_nonover) > 1 and ic_nonover.std() > 0 else 0.0)
    return XSWalkForward(oos, round(float(oos_ic.mean()), 4), round(ic_t, 2),
                         round(sharpe, 3), round(float((1 + oos).prod() - 1), 4),
                         chosen, len(parts))


def _held_weights(signal: pd.DataFrame, rebalance: int) -> pd.DataFrame:
    from autotrader.analysis.cross_sectional import to_weights
    w = to_weights(signal)
    if rebalance > 1:
        keep = pd.Series(np.arange(len(w)) % rebalance == 0, index=w.index)
        w = w.where(keep, other=np.nan).ffill().fillna(0.0)
    return w.shift(1).fillna(0.0)


def _block_returns(returns: pd.Series, block: int) -> pd.Series:
    grp = np.arange(len(returns)) // block
    return (1 + returns).groupby(grp).prod() - 1
