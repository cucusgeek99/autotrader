"""Walk-forward confirmation of long-only smart-beta + stablecoin regime gate.

Each fold picks the (factor, lookback) with the best TRAIN Sharpe, applies it to
the purged TEST window with the regime gate. Accumulates true OOS returns. The
honest test of whether the +124%/Sharpe-0.72 in-sample result is real or selection.

Signals are precomputed once (full frame) then sliced per fold — fast.
Run: .venv/bin/python scripts/wf_longonly_regime.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from autotrader.analysis import cross_sectional as xs
from autotrader.analysis.cross_sectional import apply_band, to_weights
from autotrader.config import Config
from autotrader.data.onchain import stablecoin_supply, supply_regime

cfg = Config.load("config_quick.yaml")
fees, bpy = cfg["fees"], cfg.bars_per_year
tc = fees["taker"] + fees["slippage"]
closes = pd.read_pickle(".cache/hourly_2022.pkl")
H = 12
rets = closes.pct_change().fillna(0.0)

# regime: daily stablecoin 30d momentum → hourly, lagged 1 day (no look-ahead)
reg = supply_regime(stablecoin_supply(), 30).shift(1).reindex(
    closes.index.normalize(), method="ffill")
reg.index = closes.index
gate = reg.fillna(1.0)

CANDIDATES = {
    "lowbeta": (lambda lb: xs.sig_lowbeta(closes, lb), [168, 336, 720]),
    "lowvol":  (lambda lb: xs.sig_lowvol(closes, lb), [96, 168, 336]),
    "lottery": (lambda lb: xs.sig_lottery(closes, lb), [48, 72, 168]),
}


def gated_longonly_net(sig):
    """Full-frame net-return series for a regime-gated long-only book."""
    w = to_weights(sig)
    keep = pd.Series(np.arange(len(w)) % H == 0, index=w.index)
    w = w.where(keep, np.nan).ffill().fillna(0.0)
    w = apply_band(w, 0.02)
    lo = w.clip(lower=0)
    lo = lo.div(lo.sum(axis=1), axis=0).fillna(0.0)
    pos = lo.mul(gate, axis=0)                      # regime gate
    held = pos.shift(1).fillna(0.0)
    return (held * rets).sum(axis=1) - held.diff().abs().sum(axis=1).fillna(0.0) * tc


print("precomputing candidate nets...", flush=True)
nets = {}
for name, (b, grid) in CANDIDATES.items():
    for lb in grid:
        nets[(name, lb)] = gated_longonly_net(b(lb))
mkt = rets.mean(axis=1)
print(f"{len(nets)} candidates ready over {len(closes)} bars", flush=True)


def sh(x):
    x = x.dropna()
    return float(x.mean() / x.std() * np.sqrt(bpy)) if len(x) > 2 and x.std() > 0 else 0.0


def mdd(x):
    eq = (1 + x).cumprod()
    return float((eq / eq.cummax() - 1).min()) if len(eq) else 0.0


# walk-forward
train, test, embargo = 4000, 1400, H + 24
n = len(closes)
oos, oos_mkt, picks = [], [], []
start = 0
while start + train + embargo + test <= n:
    tr = slice(start, start + train)
    te = slice(start + train + embargo, start + train + embargo + test)
    best = max(nets, key=lambda k: sh(nets[k].iloc[tr]))     # pick on TRAIN only
    picks.append(best)
    oos.append(nets[best].iloc[te])
    oos_mkt.append(mkt.iloc[te])
    start += test

oos = pd.concat(oos)
oos_mkt = pd.concat(oos_mkt)
print(f"\n{len(picks)} folds | picks: {[f'{f}:{lb}' for f, lb in picks]}", flush=True)
print(f"\n{'book':28} {'OOS ret':>9} {'Sharpe':>7} {'maxDD':>7}", flush=True)
print(f"{'WF long-only+regime (OOS)':28} {(1 + oos).prod() - 1:+9.1%} "
      f"{sh(oos):+7.2f} {mdd(oos):+7.1%}", flush=True)
print(f"{'market (same OOS window)':28} {(1 + oos_mkt).prod() - 1:+9.1%} "
      f"{sh(oos_mkt):+7.2f} {mdd(oos_mkt):+7.1%}", flush=True)
yr = ' '.join(f"{y}:{(1 + oos[oos.index.year == y]).prod() - 1:+.0%}"
              for y in [2022, 2023, 2024, 2025, 2026] if (oos.index.year == y).any())
print(f"\nOOS per-year: {yr}", flush=True)
