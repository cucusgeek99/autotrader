"""Multi-asset paper broker — holds a dollar-neutral cross-sectional book.

Unlike the single-asset PaperBroker, this tracks a position per coin and
rebalances the whole book to a target weight vector (long-top / short-bottom).
Weights are dollar-neutral (sum≈0), so longs and shorts offset and cash holds the
collateral. State persists to JSON. No real money.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class MultiAssetPaperBroker:
    cash: float = 10_000.0
    positions: dict = field(default_factory=dict)      # symbol -> base units (±)
    fee: float = 0.0006
    slippage: float = 0.0005
    fills: list = field(default_factory=list)
    equity_history: list = field(default_factory=list)  # [{time, equity}]
    state_path: str = ".cache/xs_paper_state.json"

    def equity(self, prices: dict) -> float:
        v = self.cash
        for sym, units in self.positions.items():
            if sym in prices:
                v += units * prices[sym]
        return v

    def rebalance_to(self, target_weights: dict, prices: dict, time: str) -> list:
        """Move the book to target weights (fraction of equity per coin)."""
        eq = self.equity(prices)
        cost = self.fee + self.slippage
        fills = []
        for sym, w in target_weights.items():
            price = prices.get(sym)
            if not price:
                continue
            desired = w * eq / price
            cur = self.positions.get(sym, 0.0)
            delta = desired - cur
            if abs(delta * price) < max(1.0, 0.002 * eq):   # ignore dust
                continue
            notional = abs(delta * price)
            fee = notional * cost
            self.cash -= delta * price + fee
            self.positions[sym] = desired
            fills.append({
                "time": time, "symbol": sym,
                "side": "buy" if delta > 0 else "sell",
                "units": round(delta, 8), "price": round(price, 4),
                "weight": round(w, 4), "fee": round(fee, 4),
            })
        self.fills.extend(fills)
        self.equity_history.append({"time": time, "equity": round(self.equity(prices), 2)})
        return fills

    def book(self, prices: dict) -> list:
        """Current positions as a list of {symbol, weight, value} for display."""
        eq = self.equity(prices) or 1.0
        rows = []
        for sym, units in self.positions.items():
            if sym in prices and abs(units) > 1e-12:
                val = units * prices[sym]
                rows.append({"symbol": sym, "units": round(units, 8),
                             "value": round(val, 2), "weight": round(val / eq, 4)})
        return sorted(rows, key=lambda r: r["weight"], reverse=True)

    def save(self) -> None:
        p = Path(self.state_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load_or_create(cls, **defaults) -> "MultiAssetPaperBroker":
        path = defaults.get("state_path", ".cache/xs_paper_state.json")
        p = Path(path)
        if p.exists():
            return cls(**json.loads(p.read_text()))
        return cls(**defaults)
