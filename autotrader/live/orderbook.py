"""Live L2 order-book features — the part that CANNOT be backtested.

`fetch_order_book` returns only the *current* snapshot, so there is no history
to backtest against. We therefore use the book as a live-only execution filter:
it can confirm or veto an entry the champion strategy wants to make, improving
fills without pretending we had this data historically.

Features:
  imbalance   — (bid_vol - ask_vol) / (bid_vol + ask_vol) over top N levels, [-1,1]
  spread_bps  — relative bid/ask spread in basis points
  microprice  — volume-weighted mid; its sign vs mid hints at near-term drift
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BookFeatures:
    imbalance: float
    spread_bps: float
    micro_edge: float   # (microprice - mid) / mid, signed

    def confirms(self, target: float, min_imbalance: float = 0.05) -> bool:
        """Does the book support taking `target` exposure?

        Long entries want bids ≥ asks (imbalance ≥ +thr); shorts want the
        reverse. Flat is always allowed. A thin/locked book (huge spread)
        vetoes everything.
        """
        if self.spread_bps > 25:           # too wide → bad fills, skip
            return target == 0.0
        if target > 0:
            return self.imbalance >= min_imbalance
        if target < 0:
            return self.imbalance <= -min_imbalance
        return True


def book_features(exchange, symbol: str, depth: int = 20) -> BookFeatures:
    """Compute book features from a live snapshot. `exchange` is a ccxt client."""
    ob = exchange.fetch_order_book(symbol, limit=depth)
    bids = ob.get("bids", [])[:depth]
    asks = ob.get("asks", [])[:depth]
    if not bids or not asks:
        return BookFeatures(0.0, 999.0, 0.0)

    bid_vol = sum(b[1] for b in bids)
    ask_vol = sum(a[1] for a in asks)
    tot = bid_vol + ask_vol
    imbalance = (bid_vol - ask_vol) / tot if tot else 0.0

    best_bid, best_ask = bids[0][0], asks[0][0]
    mid = (best_bid + best_ask) / 2
    spread_bps = (best_ask - best_bid) / mid * 1e4 if mid else 999.0

    # microprice: mid weighted toward the heavier side of the top of book
    bb_v, ba_v = bids[0][1], asks[0][1]
    micro = (best_bid * ba_v + best_ask * bb_v) / (bb_v + ba_v) if (bb_v + ba_v) else mid
    micro_edge = (micro - mid) / mid if mid else 0.0

    return BookFeatures(
        imbalance=round(imbalance, 4),
        spread_bps=round(spread_bps, 3),
        micro_edge=round(micro_edge, 6),
    )
