"""Directional overlay — stack market beta on top of the market-neutral alpha.

The findings_xs book is dollar-neutral: it harvests the low-vol/reversal spread
but deliberately throws away the market's direction. That's safe but slow. This
overlay adds a *tunable* net long/short tilt driven by time-series momentum
(TSMOM) — when the market trends up we lean net long and ride it; when it trends
down we lean net short. TSMOM is the directional signal with the strongest
cross-asset, multi-decade evidence (Moskowitz-Ooi-Pedersen 2012).

Combined book:  final = neutral_weights  +  k · directional_weights
  k = 0   → pure market-neutral (the original sleeve)
  k > 0   → bigger swings: you now capture market moves, up AND down.

The exposure is vol-normalized and tanh-bounded to [-1, +1] so a violent move
can't run the tilt to infinity.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def index_returns(closes: pd.DataFrame) -> pd.Series:
    """Equal-weight market return per bar across the universe."""
    return closes.pct_change().mean(axis=1)


def trend_exposure(closes: pd.DataFrame, fast: int = 24, slow: int = 168,
                   scale: float = 1.0) -> pd.Series:
    """Vol-normalized time-series momentum → net market exposure in [-1, +1].

    Uses the equal-weight index's trailing return over `slow` bars, normalized by
    its realized vol, squashed by tanh. `fast` smooths the index first to cut
    whipsaw. `scale` tunes how hard a given trend pushes the tilt (lower = more
    aggressive)."""
    idx = (1.0 + index_returns(closes).fillna(0.0)).cumprod()
    smooth = idx.ewm(span=fast, adjust=False).mean()
    mom = smooth / smooth.shift(slow) - 1.0                      # trailing trend
    vol = index_returns(closes).rolling(slow).std() * np.sqrt(slow)
    z = mom / (vol.replace(0.0, np.nan) * scale)
    return np.tanh(z.fillna(0.0)).clip(-1.0, 1.0)


def directional_weights(closes: pd.DataFrame, exposure: pd.Series) -> pd.DataFrame:
    """Distribute the net exposure equally across the universe (a market tilt).
    Row sum (net) = exposure; gross = |exposure|."""
    n = closes.shape[1]
    per = exposure / max(n, 1)
    return pd.DataFrame(
        np.repeat(per.to_numpy()[:, None], n, axis=1),
        index=closes.index, columns=closes.columns,
    )


def book_net(weights: pd.DataFrame, closes: pd.DataFrame, rebalance: int,
             taker_fee: float, slippage: float, band: float = 0.0):
    """Net return + turnover for an ARBITRARY weight book (net need not be 0).

    Mirrors cross_sectional.xs_net but takes final weights directly, so it works
    for the combined neutral+directional book."""
    from autotrader.analysis.cross_sectional import apply_band
    w = weights.copy()
    if rebalance > 1:
        keep = pd.Series(np.arange(len(w)) % rebalance == 0, index=w.index)
        w = w.where(keep, other=np.nan).ffill().fillna(0.0)
    w = apply_band(w, band)
    held = w.shift(1).fillna(0.0)
    rets = closes.pct_change().fillna(0.0)
    port = (held * rets).sum(axis=1)
    turnover = held.diff().abs().sum(axis=1).fillna(0.0)
    net = port - turnover * (taker_fee + slippage)
    return net, turnover
