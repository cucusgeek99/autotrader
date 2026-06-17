"""The house strategy — a cross-sectional book built from what the research
actually proved (see EDGE.md), not from wishful thinking.

Every design choice is a finding:
  * **Cross-sectional, dollar-neutral** — the only place real signal survived;
    cancels market beta, which dominated single-asset noise.
  * **Low-volatility core** + **breadth factors** (lottery/MAX, betting-against-
    beta, long-term reversal) — all economically-grounded, low-turnover, and
    chosen to be uncorrelated so √breadth lifts combined skill.
  * **Cost-aware (inverse-turnover) factor weighting** — naive equal-weighting
    imports the expensive factor's turnover and loses.
  * **No-trade band + infrequent rebalance** — turnover was the binding
    constraint in every experiment.

Honest status: a market-neutral, low-turnover *sleeve*. Paper only.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from autotrader.analysis.cross_sectional import (
    combine_factors, sig_longrev, sig_lottery, sig_lowbeta, sig_lowvol,
    sig_reversal, to_weights, xs_backtest, xs_net,
)
from autotrader.analysis.directional import (
    book_net, directional_weights, trend_exposure,
)

# (name, builder(closes) -> signal panel). Lookbacks are economic priors, not fit.
DEFAULT_FACTORS = [
    ("lowvol", lambda c: sig_lowvol(c, 168)),
    ("reversal", lambda c: sig_reversal(c, 48)),
]
EXPANDED_FACTORS = DEFAULT_FACTORS + [
    ("lottery", lambda c: sig_lottery(c, 72)),
    ("lowbeta", lambda c: sig_lowbeta(c, 168)),
    ("longrev", lambda c: sig_longrev(c, 720)),
]


@dataclass
class FindingsStrategy:
    factors: list = field(default_factory=lambda: list(DEFAULT_FACTORS))
    rebalance: int = 12      # rebalance ~every 12h
    band: float = 0.02       # no-trade band on per-coin weight
    name: str = "findings_xs"
    # directional overlay (0 = pure market-neutral; >0 stacks market beta on top)
    directional_weight: float = 0.0
    trend_fast: int = 24
    trend_slow: int = 168
    trend_scale: float = 1.0
    # clip the net tilt: "" = both ways, "short_only" = only short on downtrends
    # (used by the risk-off hybrid), "long_only" = only long.
    exposure_clip: str = ""
    # long_only book: hold only the favored (positive-signal) coins, fully invested,
    # no shorts, no directional overlay. The walk-forward-confirmed smart-beta book.
    long_only: bool = False
    max_weight: float = 0.0   # per-coin cap (0 = none); single-name catastrophe insurance

    def _panels(self, closes: pd.DataFrame) -> list:
        return [b(closes) for _, b in self.factors]

    def factor_weights(self, closes: pd.DataFrame, fees: dict, bpy: float):
        """Inverse-turnover weights so cheap factors get the capital and
        expensive ones only a cost-justified tilt."""
        panels = self._panels(closes)
        inv = []
        for p in panels:
            turn = xs_backtest(p, closes, self.rebalance, fees["taker"],
                               fees["slippage"], bpy).ann_turnover
            inv.append(1.0 / max(turn, 1.0))
        s = sum(inv) or 1.0
        return [x / s for x in inv], panels

    def composite(self, closes: pd.DataFrame, fees: dict, bpy: float):
        weights, panels = self.factor_weights(closes, fees, bpy)
        return combine_factors(panels, weights), weights, panels

    def exposure_series(self, closes: pd.DataFrame) -> pd.Series:
        """Net market tilt the overlay wants per bar (0 if directional off)."""
        if self.directional_weight <= 0:
            return pd.Series(0.0, index=closes.index)
        expo = trend_exposure(closes, self.trend_fast, self.trend_slow,
                              self.trend_scale)
        if self.exposure_clip == "short_only":
            expo = expo.clip(upper=0.0)     # only express the tilt when bearish
        elif self.exposure_clip == "long_only":
            expo = expo.clip(lower=0.0)
        return expo

    def long_only_weights(self, closes: pd.DataFrame, fees: dict, bpy: float) -> pd.DataFrame:
        """Hold only the favored (positive-signal) coins, fully invested (row sum 1)."""
        comp, _, _ = self.composite(closes, fees, bpy)
        lo = to_weights(comp).clip(lower=0.0)
        lo = lo.div(lo.sum(axis=1), axis=0).fillna(0.0)
        if self.max_weight and self.max_weight > 0:
            for _ in range(30):                       # water-fill: cap + redistribute
                over = lo > self.max_weight + 1e-12
                if not over.to_numpy().any():
                    break
                excess = (lo.where(over, 0.0) - self.max_weight).clip(lower=0.0).sum(axis=1)
                lo = lo.where(~over, self.max_weight)
                under = lo < self.max_weight - 1e-12
                denom = lo.where(under, 0.0).sum(axis=1).replace(0.0, np.nan)
                lo = lo + lo.where(under, 0.0).mul((excess / denom).fillna(0.0), axis=0)
        return lo

    def target_weights(self, closes: pd.DataFrame, fees: dict, bpy: float) -> pd.DataFrame:
        if self.long_only:
            return self.long_only_weights(closes, fees, bpy)
        comp, _, _ = self.composite(closes, fees, bpy)
        neutral = to_weights(comp)
        if self.directional_weight <= 0:
            return neutral
        expo = self.exposure_series(closes)
        tilt = directional_weights(closes, expo) * self.directional_weight
        return neutral.add(tilt, fill_value=0.0)

    def characterize(self, closes: pd.DataFrame, fees: dict, bpy: float) -> dict:
        comp, fw, _ = self.composite(closes, fees, bpy)
        if self.long_only or self.directional_weight > 0:
            weights = self.target_weights(closes, fees, bpy)
            net, turnover = book_net(weights, closes, self.rebalance,
                                     fees["taker"], fees["slippage"], self.band)
        else:
            net, turnover = xs_net(comp, closes, self.rebalance, fees["taker"],
                                   fees["slippage"], self.band)
        net = net.dropna()
        equity = (1 + net).cumprod()
        mean, std = net.mean(), net.std()
        return {
            "equity": equity,
            "net": net,
            "total_return": round(float(equity.iloc[-1] - 1), 4) if len(equity) else 0.0,
            "sharpe": round(float(mean / std * np.sqrt(bpy)), 3) if std > 0 else 0.0,
            "max_drawdown": round(float((equity / equity.cummax() - 1).min()), 4) if len(equity) else 0.0,
            "ann_turnover": round(float(turnover.mean() * bpy), 1),
            "factor_weights": {name: round(w, 3) for (name, _), w in zip(self.factors, fw)},
            "directional_weight": self.directional_weight,
            "avg_net_exposure": round(float(self.exposure_series(closes).abs().mean()), 3),
        }
