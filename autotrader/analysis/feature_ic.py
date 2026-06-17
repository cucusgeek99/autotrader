"""Feature information coefficient (IC) analysis.

The honest test of "edge": does a feature actually predict forward returns?
For each feature we compute the Spearman rank correlation against the next
`horizon`-bar return (the IC), its t-statistic (is it distinguishable from
noise?), and a directional hit-rate. A feature with |IC| of even 0.03-0.05 that
is statistically stable is genuinely useful at scale; near-zero IC means no edge
no matter how clever the strategy wrapped around it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ICResult:
    feature: str
    ic: float          # Spearman correlation with forward return
    t_stat: float      # IC * sqrt(n-2) / sqrt(1-IC^2)
    hit_rate: float    # fraction of bars where sign(feature) == sign(fwd return)
    n: int

    def as_row(self) -> dict:
        return {
            "feature": self.feature,
            "IC": round(self.ic, 4),
            "t_stat": round(self.t_stat, 2),
            "hit_rate": round(self.hit_rate, 3),
            "n": self.n,
        }


def _spearman(a: pd.Series, b: pd.Series) -> float:
    return float(a.rank().corr(b.rank()))


def compute_ic(
    features: pd.DataFrame, close: pd.Series, horizon: int = 1
) -> list[ICResult]:
    """IC of every column in `features` vs the forward `horizon`-bar return."""
    fwd = close.pct_change(horizon).shift(-horizon)  # return over the NEXT horizon bars
    results: list[ICResult] = []
    for name in features.columns:
        x = features[name]
        valid = x.notna() & fwd.notna()
        xv, yv = x[valid], fwd[valid]
        n = len(xv)
        if n < 30 or xv.std() == 0:
            results.append(ICResult(name, 0.0, 0.0, 0.5, n))
            continue
        ic = _spearman(xv, yv)
        t = ic * np.sqrt(max(n - 2, 1)) / np.sqrt(max(1 - ic**2, 1e-9))
        # directional hit rate over non-flat feature bars
        nz = xv != 0
        hit = float((np.sign(xv[nz]) == np.sign(yv[nz])).mean()) if nz.any() else 0.5
        results.append(ICResult(name, ic, float(t), hit, n))
    return sorted(results, key=lambda r: abs(r.ic), reverse=True)
