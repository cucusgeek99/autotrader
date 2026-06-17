"""Paper trading broker — simulated portfolio, real prices. No real money.

Holds USDT cash plus a (possibly negative, i.e. short) base-asset position.
`rebalance_to` moves the portfolio to a target exposure fraction in [-1, 1]
and charges fees + slippage on the traded notional. State is persisted to
JSON so a restarted live loop resumes where it left off.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Fill:
    time: str
    symbol: str
    side: str          # buy / sell
    units: float
    price: float
    fee: float
    target_fraction: float


@dataclass
class PaperBroker:
    symbol: str
    cash: float = 10_000.0
    position: float = 0.0          # base-asset units (negative = short)
    fee: float = 0.0006
    slippage: float = 0.0005
    fills: list[Fill] = field(default_factory=list)
    state_path: str = ".cache/paper_state.json"

    def equity(self, price: float) -> float:
        return self.cash + self.position * price

    def current_fraction(self, price: float) -> float:
        eq = self.equity(price)
        return (self.position * price / eq) if eq else 0.0

    def rebalance_to(self, target_fraction: float, price: float, time: str) -> Fill | None:
        """Adjust holdings so exposure == target_fraction of equity."""
        target_fraction = max(-1.0, min(1.0, target_fraction))
        cost = self.fee + self.slippage
        eq = self.equity(price)
        desired_units = target_fraction * eq / price
        delta = desired_units - self.position

        # Reserve fee headroom: when *increasing* a long, the cash spent plus
        # its fee must fit in available cash, else cash would go negative at
        # full exposure. Scale the buy down to what's actually affordable.
        if delta > 0:
            affordable = self.cash / (price * (1 + cost))
            if delta > affordable:
                delta = max(0.0, affordable)
                desired_units = self.position + delta

        # ignore dust trades
        if abs(delta * price) < max(1e-6, 0.001 * eq):
            return None

        notional = abs(delta * price)
        fee = notional * cost
        self.cash -= delta * price + fee
        self.position = desired_units

        fill = Fill(
            time=time,
            symbol=self.symbol,
            side="buy" if delta > 0 else "sell",
            units=round(delta, 8),
            price=round(price, 2),
            fee=round(fee, 4),
            target_fraction=round(target_fraction, 3),
        )
        self.fills.append(fill)
        return fill

    # ---- persistence -------------------------------------------------
    def save(self) -> None:
        p = Path(self.state_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        p.write_text(json.dumps(data, indent=2))

    @classmethod
    def load_or_create(cls, symbol: str, **defaults) -> "PaperBroker":
        path = defaults.get("state_path", ".cache/paper_state.json")
        p = Path(path)
        if p.exists():
            data = json.loads(p.read_text())
            if data.get("symbol") == symbol:
                fills = [Fill(**f) for f in data.pop("fills", [])]
                return cls(**{**data, "fills": fills})
        return cls(symbol=symbol, **defaults)
