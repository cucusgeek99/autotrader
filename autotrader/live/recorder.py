"""Live L2 order-book + trade recorder — the honest way to get tick/L2 data.

There is no free historical L2 feed; you have to capture it. This connects to
Binance's public websocket (no API key needed), records top-of-book depth and
aggregated trades, computes the microstructure features we *cannot* get from
historical klines (book imbalance, microprice, spread, signed trade flow), and
writes them to disk. Run it over days/weeks to build a real dataset the existing
research harness can then backtest.

The feature math is split out (`book_snapshot_features`) so it's unit-testable
without a network connection.
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path


def book_snapshot_features(bids: list, asks: list, depth: int = 20) -> dict:
    """Microstructure features from one L2 snapshot. bids/asks are [[price, qty], ...].
    This is the data that has NO free history — only capturable live."""
    bids = [(float(p), float(q)) for p, q in bids[:depth]]
    asks = [(float(p), float(q)) for p, q in asks[:depth]]
    if not bids or not asks:
        return {}
    bid_vol = sum(q for _, q in bids)
    ask_vol = sum(q for _, q in asks)
    tot = bid_vol + ask_vol
    imbalance = (bid_vol - ask_vol) / tot if tot else 0.0
    best_bid, bb_q = bids[0]
    best_ask, ba_q = asks[0]
    mid = (best_bid + best_ask) / 2
    spread_bps = (best_ask - best_bid) / mid * 1e4 if mid else 0.0
    micro = (best_bid * ba_q + best_ask * bb_q) / (bb_q + ba_q) if (bb_q + ba_q) else mid
    return {
        "mid": round(mid, 6),
        "spread_bps": round(spread_bps, 4),
        "imbalance": round(imbalance, 5),       # top-N depth imbalance, [-1, 1]
        "micro_edge": round((micro - mid) / mid, 8) if mid else 0.0,
        "bid_vol": round(bid_vol, 4),
        "ask_vol": round(ask_vol, 4),
    }


async def record(symbol: str, seconds: int, out_dir: str = ".cache/l2",
                 depth: int = 20) -> str:
    """Capture `seconds` of L2 + trades to a CSV. Returns the file path."""
    import websockets  # imported here so the dep is optional

    stream_sym = symbol.replace("/", "").lower()
    url = (f"wss://stream.binance.com:9443/stream?streams="
           f"{stream_sym}@depth{depth}@100ms/{stream_sym}@aggTrade")

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    safe = symbol.replace("/", "-")
    path = Path(out_dir) / f"{safe}_{int(time.time())}.csv"

    rows = []
    buy_flow = sell_flow = 0.0
    n_trades = 0
    deadline = time.monotonic() + seconds

    async with websockets.connect(url, ping_interval=20) as ws:
        while time.monotonic() < deadline:
            try:
                msg = json.loads(await ws.recv())
            except Exception:
                break
            stream = msg.get("stream", "")
            data = msg.get("data", {})
            if stream.endswith("aggTrade"):
                qty = float(data.get("q", 0))
                # m=True → buyer is maker → aggressor is a SELLER
                if data.get("m"):
                    sell_flow += qty
                else:
                    buy_flow += qty
                n_trades += 1
            elif "depth" in stream:
                feats = book_snapshot_features(data.get("bids", []), data.get("asks", []), depth)
                if not feats:
                    continue
                net = buy_flow - sell_flow
                tot = buy_flow + sell_flow
                feats["trade_flow"] = round(net / tot, 5) if tot else 0.0  # signed aggressor flow
                feats["n_trades"] = n_trades
                feats["ts"] = data.get("E", int(time.time() * 1000))
                rows.append(feats)
                buy_flow = sell_flow = 0.0
                n_trades = 0

    if rows:
        cols = ["ts", "mid", "spread_bps", "imbalance", "micro_edge",
                "trade_flow", "n_trades", "bid_vol", "ask_vol"]
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({c: r.get(c, "") for c in cols})
    return str(path)
