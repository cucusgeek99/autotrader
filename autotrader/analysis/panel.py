"""Panel / cross-symbol validation of the nested order-flow strategy.

The strategy's hyperparameters (feature set, lookback, horizon, threshold grid)
were all chosen on BTC. The honest test is therefore on *other* symbols, never
seen during development. This runs the fully peek-free nested pipeline on a
basket and reports significance three ways, from optimistic to conservative:

  1. per-symbol      — mean OOS return + t-stat for each instrument
  2. pooled          — every held-out symbol-fold treated as one observation
                       (OPTIMISTIC: crypto folds are cross-correlated, so this
                       overstates significance)
  3. portfolio       — equal-weight the held-out symbols within each time period,
                       then t-stat across periods. Cross-sectional correlation is
                       absorbed into the portfolio, so this is the HONEST headline.
"""
from __future__ import annotations

import numpy as np

from autotrader.analysis.purged_wf import purged_nested_regime


def _stats(returns: list[float]) -> dict:
    n = len(returns)
    if n == 0:
        return {"n": 0, "mean": 0.0, "t_stat": 0.0, "pos": "0/0"}
    arr = np.array(returns, dtype=float)
    se = arr.std(ddof=1) / np.sqrt(n) if n > 1 else float("inf")
    t = float(arr.mean() / se) if se not in (0.0, float("inf")) else 0.0
    return {
        "n": n,
        "mean": round(float(arr.mean()), 4),
        "t_stat": round(t, 2),
        "pos": f"{int((arr > 0).sum())}/{n}",
    }


def panel_validation(
    dfs: dict,
    dev_symbols: set,
    runner=purged_nested_regime,
    **nested_kwargs,
) -> dict:
    """Run a purged WF `runner` per symbol and aggregate held-out significance.

    `dfs` maps symbol -> OHLCV DataFrame. `dev_symbols` are excluded from the
    headline (they were used to design the strategy). `runner` is any function
    returning the standard {folds, n_folds, chosen_thresholds} shape.
    """
    per_symbol = {}
    fold_returns: dict[str, list] = {}
    for sym, df in dfs.items():
        res = runner(df, **nested_kwargs)
        per_symbol[sym] = {
            "stats": _stats([f["return"] for f in res["folds"]]),
            "mean_sharpe": round(
                float(np.mean([f["sharpe"] for f in res["folds"]])) if res["folds"] else 0.0, 3
            ),
            "chosen_thresholds": res.get("chosen_thresholds", []),
            "n_folds": res["n_folds"],
        }
        fold_returns[sym] = [f["return"] for f in res["folds"]]

    held = [s for s in dfs if s not in dev_symbols]
    pooled, portfolio = [], []
    if held:
        k = min(len(fold_returns[s]) for s in held)  # align fold count
        pooled = [r for s in held for r in fold_returns[s]]
        for i in range(k):
            portfolio.append(float(np.mean([fold_returns[s][i] for s in held])))

    pooled_stats = _stats(pooled)
    port_stats = _stats(portfolio)
    sym_means = [per_symbol[s]["stats"]["mean"] for s in held]
    symbols_positive = sum(1 for m in sym_means if m > 0)

    pt = port_stats["t_stat"]
    if pt > 2 and port_stats["pos"] and symbols_positive >= max(1, len(held) * 0.6):
        verdict = "SIGNIFICANT out-of-sample edge across instruments"
    elif port_stats["mean"] > 0 and symbols_positive >= max(1, len(held) * 0.5):
        verdict = "promising out-of-sample — directionally consistent, sub-threshold significance"
    else:
        verdict = "does not generalize out-of-sample — likely overfit to the dev symbol"

    return {
        "held_out": held,
        "dev_symbols": sorted(dev_symbols),
        "per_symbol": per_symbol,
        "pooled": pooled_stats,
        "portfolio": port_stats,
        "symbols_positive": f"{symbols_positive}/{len(held)}",
        "verdict": verdict,
    }
