"""Backtest the FULL integrated live book, faithfully reproducing the deployed
config: EXPANDED factor book + directional overlay (7d trend, k=0.5, scale 2.0,
leverage 1.5) + stablecoin regime gate (hybrid: risk-off → short-only directional,
de-levered). Pulls multi-year HOURLY history so the window spans a real bear.

Run: .venv/bin/python scripts/backtest_integrated.py
"""
from __future__ import annotations

import time

import ccxt
import numpy as np
import pandas as pd

from autotrader.analysis import cross_sectional as xs
from autotrader.analysis.cross_sectional import to_weights
from autotrader.analysis.directional import book_net, directional_weights, trend_exposure
from autotrader.config import Config
from autotrader.data.onchain import stablecoin_supply, supply_regime
from autotrader.portfolio import EXPANDED_FACTORS, FindingsStrategy

UNI = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", "ADA/USDT",
       "DOGE/USDT", "AVAX/USDT", "LINK/USDT", "LTC/USDT", "DOT/USDT", "TRX/USDT"]


def fetch_hourly(ex, sym, since_ms):
    out, since = [], since_ms
    while True:
        o = ex.fetch_ohlcv(sym, "1h", since=since, limit=1000)
        if not o:
            break
        out += o
        since = o[-1][0] + 3_600_000
        if len(o) < 1000:
            break
    return pd.Series({pd.to_datetime(c[0], unit="ms"): c[4] for c in out})


def stats(net, bpy):
    net = net.dropna()
    eq = (1 + net).cumprod()
    sh = float(net.mean() / net.std() * np.sqrt(bpy)) if net.std() > 0 else 0.0
    dd = float((eq / eq.cummax() - 1).min()) if len(eq) else 0.0
    return float(eq.iloc[-1] - 1) if len(eq) else 0.0, sh, dd


def main():
    cfg = Config.load("config_quick.yaml")
    fees, bpy = cfg["fees"], cfg.bars_per_year
    ex = ccxt.binance({"enableRateLimit": True})
    since = ex.parse8601("2022-01-01T00:00:00Z")
    print("fetching multi-year hourly history (this takes a few min)...")
    series = {}
    for s in UNI:
        try:
            series[s] = fetch_hourly(ex, s, since)
            print(f"  {s}: {len(series[s])} bars")
        except Exception as e:
            print(f"  skip {s}: {e}")
    closes = pd.DataFrame(series).sort_index().dropna()
    print(f"aligned: {len(closes)} hourly bars, {closes.index[0]} → {closes.index[-1]}")

    # regime: daily stablecoin 30d momentum → hourly, lagged 1 day (no look-ahead)
    supply = stablecoin_supply()
    reg_daily = supply_regime(supply, 30).shift(1)
    reg_h = reg_daily.reindex(closes.index.normalize(), method="ffill")
    reg_h.index = closes.index
    risk_on = reg_h.fillna(1.0).to_numpy().astype(bool)
    print(f"risk-on {risk_on.mean():.0%} of hours / risk-off {1 - risk_on.mean():.0%}")

    # building blocks (deployed config)
    strat = FindingsStrategy(factors=list(EXPANDED_FACTORS), rebalance=12,
                             directional_weight=0.5, trend_fast=24, trend_slow=168,
                             trend_scale=2.0)
    comp, _, _ = strat.composite(closes, fees, bpy)
    neutral = to_weights(comp)
    expo = trend_exposure(closes, 24, 168, 2.0)
    k, lev_on, lev_off = 0.5, 1.5, 1.0
    ro = pd.Series(risk_on, index=closes.index)

    def build(mode):
        if mode == "neutral":
            return neutral.mul(1.0)
        if mode == "dir_always":          # directional + leverage, no regime gate
            return (neutral + k * directional_weights(closes, expo)).mul(lev_on)
        # regime-gated books
        e_on = expo
        e_off = expo.clip(upper=0.0) if mode == "hybrid" else expo * 0.0  # flat=neutral
        e_eff = e_on.where(ro, e_off)
        lev = pd.Series(np.where(ro, lev_on, lev_off), index=closes.index)
        return (neutral + k * directional_weights(closes, e_eff)).mul(lev, axis=0)

    print(f"\n{'book':28} {'return':>9} {'Sharpe':>7} {'maxDD':>7} {'turn':>6}")
    results = {}
    for mode in ("neutral", "dir_always", "flat", "hybrid"):
        w = build(mode)
        net, turn = book_net(w, closes, 12, fees["taker"], fees["slippage"], band=0.02)
        tr, sh, dd = stats(net, bpy)
        results[mode] = net
        label = {"neutral": "factor only (no overlay)", "dir_always": "+directional (no regime)",
                 "flat": "+regime FLAT (conservative)", "hybrid": "+regime HYBRID (live config)"}[mode]
        print(f"{label:28} {tr:+9.1%} {sh:+7.2f} {dd:+7.1%} {turn.mean() * bpy:6.0f}x")

    # two-halves OOS on the live (hybrid) book
    print("\n--- OOS two-halves (live hybrid book) ---")
    net = results["hybrid"].dropna()
    mid = len(net) // 2
    for name, seg in [("first half", net.iloc[:mid]), ("second half", net.iloc[mid:])]:
        tr, sh, dd = stats(seg, bpy)
        print(f"  {name:12} return {tr:+7.1%}  Sharpe {sh:+.2f}  maxDD {dd:+.1%}")


if __name__ == "__main__":
    t = time.time()
    main()
    print(f"\ndone in {time.time() - t:.0f}s")
