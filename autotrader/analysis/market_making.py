"""Market-making (spread-capture) economics — the honest feasibility check.

You can't trust a maker *backtest* from klines: whether your limit order fills
depends on queue position and flow you didn't participate in, so fill assumptions
make maker backtests lie. What you CAN do honestly is check the economics from
first principles with live numbers, because three things decide it before any
clever code:

  1. spread you capture per round-trip  (half the bid/ask spread)
  2. the maker FEE you pay (or rebate you earn)
  3. adverse selection — how much price moves against you right after a fill
     (you get picked off by faster/informed traders)

Net edge per fill ≈ half_spread − maker_fee − adverse_selection.
The first two are knowable now. The third is a latency/queue capability.
"""
from __future__ import annotations


def measure_spread_bps(symbol: str, exchange_id: str = "binance") -> float:
    """Live top-of-book spread in basis points."""
    import ccxt
    ex = getattr(ccxt, exchange_id)({"enableRateLimit": True})
    ob = ex.fetch_order_book(symbol, 5)
    if not ob["bids"] or not ob["asks"]:
        return 0.0
    bid, ask = ob["bids"][0][0], ob["asks"][0][0]
    mid = (bid + ask) / 2
    return (ask - bid) / mid * 1e4 if mid else 0.0


def realized_vol_bps(symbol: str, timeframe: str = "1m", bars: int = 1000,
                     exchange_id: str = "binance") -> float:
    """Per-bar realized volatility (std of returns) in bps — a proxy for the
    adverse move you eat when picked off."""
    import ccxt
    import numpy as np
    ex = getattr(ccxt, exchange_id)({"enableRateLimit": True})
    o = ex.fetch_ohlcv(symbol, timeframe, limit=bars)
    closes = np.array([c[4] for c in o], dtype=float)
    rets = np.diff(closes) / closes[:-1]
    return float(np.std(rets) * 1e4)


def mm_economics(spread_bps: float, vol_bps: float, maker_fee_bps: float = 2.0) -> dict:
    """Per-fill maker economics + the adverse-selection sensitivity.

    `avoidance` = fraction of adverse selection you dodge via latency/queue
    (0 = picked off on every move; 1 = never picked off). Net per fill =
    half_spread − fee − vol×(1−avoidance)."""
    half = spread_bps / 2
    gross_margin = half - maker_fee_bps   # before adverse selection
    # avoidance needed to break even against adverse selection
    breakeven_avoid = 1 - gross_margin / vol_bps if vol_bps > 0 else 1.0
    table = []
    for av in (0.0, 0.5, 0.8, 0.9, 0.95, 0.99, 0.999):
        net = gross_margin - vol_bps * (1 - av)
        table.append((av, round(net, 3)))
    return {
        "half_spread_bps": round(half, 4),
        "maker_fee_bps": maker_fee_bps,
        "gross_margin_bps": round(gross_margin, 4),   # half_spread − fee
        "vol_bps": round(vol_bps, 2),
        "breakeven_avoidance": round(breakeven_avoid, 4),
        "table": table,
    }
