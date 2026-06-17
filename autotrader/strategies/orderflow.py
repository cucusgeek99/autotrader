"""Order-flow strategies — edge-seeking strategies built on microstructure
features rather than pure price action.

These need the extended klines columns (taker_buy_base, etc.). If those are
absent (non-Binance data), `generate_signals` returns flat so the optimizer's
min-trades guard naturally drops them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from autotrader.features import microstructure as ms
from autotrader.strategies.base import Strategy
from autotrader.strategies.registry import register


@register
class OrderFlowMomentum(Strategy):
    """Follow aggressive flow: go with the side that's crossing the spread when
    the smoothed order-flow imbalance is decisively one-directional."""

    name = "orderflow_momentum"
    param_space = {
        "window": ("int", 3, 36),
        "threshold": ("float", 0.02, 0.25),
        "allow_short": ("categorical", [True, False]),
    }

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        if not ms.has_flow(df):
            return pd.Series(0.0, index=df.index)
        flow = ms.signed_flow_momentum(df, int(self.params["window"]))
        thr = float(self.params["threshold"])
        pos = pd.Series(0.0, index=df.index)
        pos[flow > thr] = 1.0
        pos[flow < -thr] = -1.0 if self.params["allow_short"] else 0.0
        return pos


@register
class OrderFlowReversion(Strategy):
    """Fade flow extremes: when aggressive flow spikes to an extreme on a single
    bar, short-horizon price impact tends to mean-revert as it's absorbed."""

    name = "orderflow_reversion"
    param_space = {
        "extreme": ("float", 0.3, 0.8),
        "hold": ("int", 1, 8),
        "allow_short": ("categorical", [True, False]),
    }

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        if not ms.has_flow(df):
            return pd.Series(0.0, index=df.index)
        ofi = ms.order_flow_imbalance(df, 1)
        ext = float(self.params["extreme"])
        raw = pd.Series(np.nan, index=df.index)
        raw[ofi > ext] = -1.0 if self.params["allow_short"] else 0.0  # fade buying
        raw[ofi < -ext] = 1.0                                          # fade selling
        # hold the fade for `hold` bars
        return raw.ffill(limit=int(self.params["hold"])).fillna(0.0)


@register
class VWAPReversion(Strategy):
    """Mean-revert toward rolling VWAP, scaled by realized vol so the band
    adapts to the regime (wider when volatile)."""

    name = "vwap_reversion"
    param_space = {
        "window": ("int", 12, 96),
        "band": ("float", 0.5, 3.0),
        "allow_short": ("categorical", [True, False]),
    }

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        dev = ms.vwap_deviation(df, int(self.params["window"]))
        vol = ms.realized_vol(df, int(self.params["window"])).replace(0.0, np.nan)
        z = (dev / vol).fillna(0.0)
        band = float(self.params["band"])
        pos = pd.Series(0.0, index=df.index)
        pos[z < -band] = 1.0                                       # cheap vs vwap → long
        pos[z > band] = -1.0 if self.params["allow_short"] else 0.0
        return pos.replace(0.0, np.nan).ffill().fillna(0.0)


def _zscore(s: pd.Series, window: int) -> pd.Series:
    mean = s.rolling(window, min_periods=window).mean()
    std = s.rolling(window, min_periods=window).std().replace(0.0, np.nan)
    return ((s - mean) / std).fillna(0.0)


@register
class EdgeComposite(Strategy):
    """Evidence-driven composite of the features that scored a significant IC in
    `autotrader features` analysis (order-flow momentum, 6-bar OFI, trade
    intensity, VWAP reversion). Feature *directions* are fixed from theory; the
    optimizer tunes lookback, entry threshold, and hold — and walk-forward
    judges it honestly OOS. Designed for the multi-hour horizon where the signal
    actually lives, not bar-to-bar.
    """

    name = "edge_composite"
    param_space = {
        "lookback": ("int", 12, 96),
        "threshold": ("float", 0.3, 1.5),
        "hold": ("int", 4, 24),
        "allow_short": ("categorical", [True, False]),
    }

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        lb = int(self.params["lookback"])
        # significant, positively-signed features (continuation); vwap_dev fades
        flow = ms.signed_flow_momentum(df, lb) if ms.has_flow(df) else pd.Series(0.0, index=df.index)
        ofi6 = ms.order_flow_imbalance(df, 6) if ms.has_flow(df) else pd.Series(0.0, index=df.index)
        trade = ms.trade_intensity(df, lb)
        vdev = ms.vwap_deviation(df, lb)
        # combine standardized signals; vwap deviation is mean-reverting (minus)
        score = (
            _zscore(flow, lb) + _zscore(ofi6, lb) + _zscore(trade, lb)
            - _zscore(vdev, lb)
        ) / 4.0
        thr = float(self.params["threshold"])
        raw = pd.Series(np.nan, index=df.index)
        raw[score > thr] = 1.0
        raw[score < -thr] = -1.0 if self.params["allow_short"] else 0.0
        return raw.ffill(limit=int(self.params["hold"])).fillna(0.0)


@register
class FundingFade(Strategy):
    """Fade extreme perp funding (needs a `funding` column from the perp feed).
    Short crowded-long funding, long crowded-short. Flat on spot data."""

    name = "funding_fade"
    param_space = {
        "lookback": ("int", 72, 336),
        "threshold": ("float", 1.0, 2.5),
        "hold": ("int", 8, 48),
        "allow_short": ("categorical", [True, False]),
    }

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        if "funding" not in df.columns:
            return pd.Series(0.0, index=df.index)
        from autotrader.analysis.funding_fade import fade_signal
        return fade_signal(
            df, int(self.params["lookback"]), float(self.params["threshold"]),
            int(self.params["hold"]), bool(self.params["allow_short"]),
        )


@register
class FlowDivergence(Strategy):
    """Accumulation/distribution: price drops but aggressive flow is buying
    (absorption → long), or price rises but flow is selling (distribution → short)."""

    name = "flow_divergence"
    param_space = {
        "window": ("int", 4, 36),
        "flow_thr": ("float", 0.05, 0.3),
        "allow_short": ("categorical", [True, False]),
    }

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        if not ms.has_flow(df):
            return pd.Series(0.0, index=df.index)
        w = int(self.params["window"])
        price_chg = df["close"].pct_change(w)
        flow = ms.signed_flow_momentum(df, w)
        thr = float(self.params["flow_thr"])
        pos = pd.Series(0.0, index=df.index)
        # price down, flow buying → absorption / accumulation
        pos[(price_chg < 0) & (flow > thr)] = 1.0
        if self.params["allow_short"]:
            pos[(price_chg > 0) & (flow < -thr)] = -1.0
        return pos.replace(0.0, np.nan).ffill(limit=w).fillna(0.0)
