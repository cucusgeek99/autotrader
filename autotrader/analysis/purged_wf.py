"""Purged, embargoed walk-forward evaluation of the order-flow composite.

Why this exists: `edge_composite` (the registered strategy) fixes its feature
*directions* from the full-sample IC — a look-ahead peek. This module removes
that peek entirely:

  * Feature weights are fit **only on each training fold** (Spearman IC of each
    feature vs the in-train forward return).
  * Training labels are **purged**: a sample's label spans the next `horizon`
    bars, so we only fit on samples whose entire label window lies inside the
    training fold (López de Prado purging).
  * A **gap of `horizon + embargo` bars** separates train end from test start,
    so no training label can leak into the test period, and serial correlation
    across the boundary is embargoed.

The test fold is then traded with train-derived weights and train-derived
standardization only — a genuinely leak-free out-of-sample read.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from autotrader.backtest.engine import run_backtest
from autotrader.features import microstructure as ms
from autotrader.knowledge.regime import REGIMES, regime_series
from autotrader.strategies.base import Strategy

FEATURES = ("flow_mom", "ofi6", "trade_z", "vwap_dev")


def composite_features(df: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Causal feature set for the composite, parameterized by lookback."""
    zero = pd.Series(0.0, index=df.index)
    flow = ms.signed_flow_momentum(df, lookback) if ms.has_flow(df) else zero
    ofi6 = ms.order_flow_imbalance(df, 6) if ms.has_flow(df) else zero
    return pd.DataFrame(
        {
            "flow_mom": flow,
            "ofi6": ofi6,
            "trade_z": ms.trade_intensity(df, lookback),
            "vwap_dev": ms.vwap_deviation(df, lookback),
        },
        index=df.index,
    )


class _PrecomputedSignal(Strategy):
    """Wraps a precomputed target-position series so the engine can score it."""

    name = "_precomputed"

    def __init__(self, signal: pd.Series):
        self.signal = signal
        self.params = {}

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        return self.signal.reindex(df.index).fillna(0.0)


@dataclass
class FoldResult:
    index: int
    weights: dict
    test_score: float
    test_sharpe: float
    test_return: float
    num_trades: int


def _spearman(a: pd.Series, b: pd.Series) -> float:
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(a.rank().corr(b.rank()))


def _fit_weights(
    feats: pd.DataFrame, close: pd.Series, train_pos: range, horizon: int
) -> dict:
    """Spearman-IC feature weights, fit on purged training rows only."""
    fwd = close.pct_change(horizon).shift(-horizon)  # NaN in last `horizon` rows → purged
    tr = feats.iloc[train_pos.start : train_pos.stop]
    fwd_tr = fwd.iloc[train_pos.start : train_pos.stop]
    valid = fwd_tr.notna()
    weights = {}
    for col in FEATURES:
        weights[col] = _spearman(tr[col][valid], fwd_tr[valid])
    norm = sum(abs(w) for w in weights.values()) or 1.0
    return {k: v / norm for k, v in weights.items()}


def _signal_from_weights(
    feats: pd.DataFrame,
    weights: dict,
    train_pos: range,
    threshold: float,
    hold: int,
    allow_short: bool,
) -> pd.Series:
    # standardize each feature using TRAIN statistics only
    tr = feats.iloc[train_pos.start : train_pos.stop]
    mu, sd = tr.mean(), tr.std().replace(0.0, np.nan)
    z = (feats - mu) / sd
    score = sum(weights[c] * z[c].fillna(0.0) for c in FEATURES)
    # standardize the score itself with train stats, threshold in z-units
    s_mu = score.iloc[train_pos.start : train_pos.stop].mean()
    s_sd = score.iloc[train_pos.start : train_pos.stop].std() or 1.0
    sz = (score - s_mu) / s_sd
    raw = pd.Series(np.nan, index=feats.index)
    raw[sz > threshold] = 1.0
    raw[sz < -threshold] = -1.0 if allow_short else 0.0
    return raw.ffill(limit=hold).fillna(0.0)


def _ic_weights(feats: pd.DataFrame, fwd: pd.Series, mask: pd.Series) -> dict:
    """Spearman-IC weights over the rows selected by `mask` (already purged)."""
    weights = {}
    for col in FEATURES:
        weights[col] = _spearman(feats[col][mask], fwd[mask])
    norm = sum(abs(w) for w in weights.values()) or 1.0
    return {k: v / norm for k, v in weights.items()}


def _regime_signal(
    feats: pd.DataFrame,
    weights_by_regime: dict,
    regimes: pd.Series,
    train_mask: pd.Series,
    threshold: float,
    hold: int,
    allow_short: bool,
) -> pd.Series:
    """Score each bar with its own regime's weights; standardize per regime
    using train-fold statistics; threshold in z-units."""
    mu, sd = feats[train_mask].mean(), feats[train_mask].std().replace(0.0, np.nan)
    z = (feats - mu) / sd
    sz = pd.Series(0.0, index=feats.index)
    for r in REGIMES:
        w = weights_by_regime[r]
        rmask = regimes == r
        if not rmask.any():
            continue
        s = sum(w[c] * z[c].fillna(0.0) for c in FEATURES)
        train_r = train_mask & rmask
        s_mu = s[train_r].mean() if train_r.any() else 0.0
        s_sd = (s[train_r].std() if train_r.any() else 1.0) or 1.0
        sz[rmask] = ((s - s_mu) / s_sd)[rmask]
    raw = pd.Series(np.nan, index=feats.index)
    raw[sz > threshold] = 1.0
    raw[sz < -threshold] = -1.0 if allow_short else 0.0
    return raw.ffill(limit=hold).fillna(0.0)


def purged_walk_forward_regime(
    df: pd.DataFrame,
    *,
    horizon: int = 12,
    lookback: int = 24,
    threshold: float = 0.5,
    hold: int | None = None,
    allow_short: bool = True,
    train: int = 1500,
    test: int = 500,
    step: int = 500,
    embargo: int = 12,
    min_regime_samples: int = 60,
    fees: dict | None = None,
    bars_per_year: float = 8760,
) -> dict:
    """Regime-CONDITIONAL purged walk-forward.

    Same leak-free discipline as `purged_walk_forward`, but feature weights are
    fit *separately per regime* on the training fold. Tests whether the
    order-flow → return relationship is stable *within* a regime even though it
    flips globally. Regimes with too few training samples fall back to the
    pooled (global) weights so we still trade them.
    """
    fees = fees or {"taker": 0.0006, "slippage": 0.0005}
    hold = hold if hold is not None else horizon
    feats = composite_features(df, lookback)
    close = df["close"]
    regimes = regime_series(df)
    fwd = close.pct_change(horizon).shift(-horizon)
    n = len(df)
    gap = horizon + embargo
    pos_index = pd.RangeIndex(n)

    folds: list[FoldResult] = []
    regime_weight_log: dict[str, list] = {r: [] for r in REGIMES}
    start = 0
    i = 0
    while start + train + gap + test <= n:
        train_mask = (pos_index >= start) & (pos_index < start + train) & fwd.notna().values
        train_mask = pd.Series(train_mask, index=df.index)
        global_w = _ic_weights(feats, fwd, train_mask)

        weights_by_regime = {}
        for r in REGIMES:
            rmask = train_mask & (regimes == r)
            if int(rmask.sum()) >= min_regime_samples:
                weights_by_regime[r] = _ic_weights(feats, fwd, rmask)
                regime_weight_log[r].append(weights_by_regime[r])
            else:
                weights_by_regime[r] = global_w  # fallback for sparse regimes

        signal = _regime_signal(
            feats, weights_by_regime, regimes, train_mask, threshold, hold, allow_short
        )
        test_start = start + train + gap
        test_df = df.iloc[test_start : test_start + test]
        test_sig = signal.iloc[test_start : test_start + test]
        res = run_backtest(
            _PrecomputedSignal(test_sig), test_df,
            fees["taker"], fees["slippage"], bars_per_year,
        )
        m = res.metrics
        folds.append(FoldResult(
            index=i, weights={}, test_score=m.score, test_sharpe=m.sharpe,
            test_return=m.total_return, num_trades=m.num_trades,
        ))
        start += step
        i += 1

    out = _aggregate(folds)
    out["regime_stability"] = _regime_stability(regime_weight_log)
    return out


def _pos_mask(df: pd.DataFrame, lo: int, hi: int) -> pd.Series:
    pos = pd.RangeIndex(len(df))
    return pd.Series((pos >= lo) & (pos < hi), index=df.index)


def _fit_eval_regime(
    feats, fwd, regimes, df, train_lo, train_hi, test_lo, test_hi,
    threshold, hold, allow_short, min_regime_samples, fees, bpy,
):
    """Fit per-regime weights on [train_lo,train_hi) (purged via fwd NaN) and
    evaluate on [test_lo,test_hi). Returns (metrics, weights_by_regime)."""
    train_mask = _pos_mask(df, train_lo, train_hi) & fwd.notna()
    global_w = _ic_weights(feats, fwd, train_mask)
    wbr = {}
    for r in REGIMES:
        rmask = train_mask & (regimes == r)
        wbr[r] = (_ic_weights(feats, fwd, rmask)
                  if int(rmask.sum()) >= min_regime_samples else global_w)
    signal = _regime_signal(feats, wbr, regimes, train_mask, threshold, hold, allow_short)
    res = run_backtest(
        _PrecomputedSignal(signal.iloc[test_lo:test_hi]),
        df.iloc[test_lo:test_hi], fees["taker"], fees["slippage"], bpy,
    )
    return res.metrics, wbr


def _select_threshold_inner(
    feats, fwd, regimes, df, outer_train_lo, train, thresholds,
    horizon, hold, allow_short, min_regime_samples, fees, bpy,
    inner_train, inner_test, inner_step,
):
    """Choose a threshold using an inner walk-forward INSIDE the outer-train
    window only. The outer test set is never touched here."""
    gap = horizon + 12  # inner purge+embargo
    scores: dict[float, list] = {t: [] for t in thresholds}
    s = outer_train_lo
    inner_folds = 0
    while s + inner_train + gap + inner_test <= outer_train_lo + train:
        itr_lo, itr_hi = s, s + inner_train
        ite_lo = s + inner_train + gap
        ite_hi = ite_lo + inner_test
        for thr in thresholds:
            m, _ = _fit_eval_regime(
                feats, fwd, regimes, df, itr_lo, itr_hi, ite_lo, ite_hi,
                thr, hold, allow_short, min_regime_samples, fees, bpy,
            )
            scores[thr].append(m.score)  # score is net + risk-adjusted
        s += inner_step
        inner_folds += 1

    if inner_folds == 0:  # not enough room — default to the middle candidate
        return float(sorted(thresholds)[len(thresholds) // 2]), {}
    mean_scores = {t: float(np.mean(v)) for t, v in scores.items()}
    best = max(mean_scores, key=mean_scores.get)
    return float(best), {round(k, 2): round(v, 3) for k, v in mean_scores.items()}


def purged_nested_regime(
    df: pd.DataFrame,
    *,
    horizon: int = 12,
    lookback: int = 24,
    thresholds: tuple = (0.5, 1.0, 1.5, 2.0, 2.5),
    hold: int | None = None,
    allow_short: bool = True,
    train: int = 1500,
    test: int = 500,
    step: int = 500,
    embargo: int = 12,
    inner_train: int = 800,
    inner_test: int = 300,
    inner_step: int = 300,
    min_regime_samples: int = 60,
    fees: dict | None = None,
    bars_per_year: float = 8760,
) -> dict:
    """Nested purged walk-forward: the entry threshold is tuned per outer fold on
    an inner walk-forward carved from that fold's TRAINING data only. The outer
    OOS result therefore has no threshold-selection peek — the cleanest read.
    """
    fees = fees or {"taker": 0.0006, "slippage": 0.0005}
    hold = hold if hold is not None else horizon
    feats = composite_features(df, lookback)
    close = df["close"]
    regimes = regime_series(df)
    fwd = close.pct_change(horizon).shift(-horizon)
    n = len(df)
    gap = horizon + embargo

    folds: list[FoldResult] = []
    chosen: list[float] = []
    start = 0
    i = 0
    while start + train + gap + test <= n:
        best_thr, inner_table = _select_threshold_inner(
            feats, fwd, regimes, df, start, train, thresholds,
            horizon, hold, allow_short, min_regime_samples, fees, bars_per_year,
            inner_train, inner_test, inner_step,
        )
        test_lo = start + train + gap
        m, _ = _fit_eval_regime(
            feats, fwd, regimes, df, start, start + train, test_lo, test_lo + test,
            best_thr, hold, allow_short, min_regime_samples, fees, bars_per_year,
        )
        folds.append(FoldResult(
            index=i, weights={"chosen_threshold": best_thr, "inner": inner_table},
            test_score=m.score, test_sharpe=m.sharpe,
            test_return=m.total_return, num_trades=m.num_trades,
        ))
        chosen.append(best_thr)
        start += step
        i += 1

    out = _aggregate(folds)
    out["chosen_thresholds"] = chosen
    return out


def _regime_stability(log: dict[str, list]) -> dict:
    """For each regime, report how consistent the fitted weights are across
    folds — the test of the hypothesis. A regime whose dominant-feature sign is
    stable across folds is exploitable; one that flips is not."""
    summary = {}
    for r, weight_dicts in log.items():
        if len(weight_dicts) < 2:
            continue
        per_feature = {}
        for c in FEATURES:
            vals = [w[c] for w in weight_dicts]
            same_sign = len({1 if v > 0 else (-1 if v < 0 else 0) for v in vals}) == 1
            per_feature[c] = {
                "mean": round(float(np.mean(vals)), 3),
                "sign_stable": same_sign,
            }
        # stable if the strongest feature keeps its sign across folds
        strongest = max(FEATURES, key=lambda c: abs(per_feature[c]["mean"]))
        summary[r] = {
            "n_folds": len(weight_dicts),
            "strongest_feature": strongest,
            "strongest_sign_stable": per_feature[strongest]["sign_stable"],
            "features": per_feature,
        }
    return summary


def purged_walk_forward(
    df: pd.DataFrame,
    *,
    horizon: int = 12,
    lookback: int = 24,
    threshold: float = 0.5,
    hold: int | None = None,
    allow_short: bool = True,
    train: int = 1500,
    test: int = 500,
    step: int = 500,
    embargo: int = 12,
    fees: dict | None = None,
    bars_per_year: float = 8760,
) -> dict:
    """Run the purged walk-forward. Returns aggregate OOS stats + per-fold rows."""
    fees = fees or {"taker": 0.0006, "slippage": 0.0005}
    hold = hold if hold is not None else horizon
    feats = composite_features(df, lookback)
    close = df["close"]
    n = len(df)
    gap = horizon + embargo

    folds: list[FoldResult] = []
    start = 0
    i = 0
    while start + train + gap + test <= n:
        train_pos = range(start, start + train)
        test_start = start + train + gap
        test_pos = range(test_start, test_start + test)

        weights = _fit_weights(feats, close, train_pos, horizon)
        signal = _signal_from_weights(feats, weights, train_pos, threshold, hold, allow_short)

        test_df = df.iloc[test_pos.start : test_pos.stop]
        test_sig = signal.iloc[test_pos.start : test_pos.stop]
        res = run_backtest(
            _PrecomputedSignal(test_sig), test_df,
            fees["taker"], fees["slippage"], bars_per_year,
        )
        m = res.metrics
        folds.append(FoldResult(
            index=i, weights={k: round(v, 3) for k, v in weights.items()},
            test_score=m.score, test_sharpe=m.sharpe,
            test_return=m.total_return, num_trades=m.num_trades,
        ))
        start += step
        i += 1

    return _aggregate(folds)


def _aggregate(folds: list[FoldResult]) -> dict:
    if not folds:
        return {"n_folds": 0, "verdict": "no folds (need more history)"}
    returns = np.array([f.test_return for f in folds])
    sharpes = np.array([f.test_sharpe for f in folds])
    scores = np.array([f.test_score for f in folds])
    n = len(folds)
    # t-stat of mean fold return vs zero (is OOS performance distinguishable from luck?)
    se = returns.std(ddof=1) / np.sqrt(n) if n > 1 else float("inf")
    t_stat = float(returns.mean() / se) if se else 0.0
    pos = int((returns > 0).sum())

    if t_stat > 2 and pos / n >= 0.6:
        verdict = "genuine OOS edge (leak-free)"
    elif returns.mean() > 0 and pos / n >= 0.5:
        verdict = "weak/promising — not significant"
    else:
        verdict = "no leak-free edge — the prior result was the look-ahead peek"

    return {
        "n_folds": n,
        "mean_oos_return": round(float(returns.mean()), 4),
        "mean_oos_sharpe": round(float(sharpes.mean()), 3),
        "mean_oos_score": round(float(scores.mean()), 4),
        "folds_positive": f"{pos}/{n}",
        "return_t_stat": round(t_stat, 2),
        "verdict": verdict,
        "folds": [
            {"fold": f.index, "return": round(f.test_return, 4),
             "sharpe": f.test_sharpe, "trades": f.num_trades, "weights": f.weights}
            for f in folds
        ],
    }
