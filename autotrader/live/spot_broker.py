"""Spot execution for the core book — long-only, delta-based, SAFE BY DEFAULT.

Unlike `LiveBroker` (which sizes each order as the full target), this broker:
  * values equity as USDT + every held coin at market (real spot accounts hold coins)
  * trades only the DELTA between current holdings and the target, so a daily run
    that changes nothing sends nothing
  * skips deltas below `band` × equity (fee saver) and below the exchange's
    min-notional; rounds amounts to the lot step
  * never shorts (spot): sells are capped at what is actually held
  * passes every target through the RiskManager (kill-switch flattens to USDT)

dry_run=True by default. Mainnet real orders require dry_run=False explicitly.
Keys: BINANCE_SPOT_API_KEY / BINANCE_SPOT_SECRET (kept separate from the futures
testnet keys on purpose).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from autotrader.risk import RiskManager

QUOTE = "USDT"


@dataclass
class SpotBroker:
    api_key: str = ""
    secret: str = ""
    dry_run: bool = True
    testnet: bool = False
    band: float = 0.02
    risk: RiskManager = field(default_factory=RiskManager)
    _ex: object = None

    def _exchange(self):
        if self._ex is None:
            import ccxt
            cfg = {"enableRateLimit": True, "options": {"defaultType": "spot"}}
            if self.api_key:
                cfg.update(apiKey=self.api_key, secret=self.secret)
            self._ex = ccxt.binance(cfg)
            if self.testnet:
                self._ex.set_sandbox_mode(True)
            self._ex.load_markets()
        return self._ex

    def mode(self) -> str:
        if self.dry_run:
            return "DRY-RUN (no orders sent)"
        return "TESTNET (sandbox)" if self.testnet else "LIVE (REAL MONEY)"

    def prices(self, symbols: list[str]) -> dict:
        ex = self._exchange()
        return {s: float(ex.fetch_ticker(s)["last"]) for s in symbols}

    def holdings(self, symbols: list[str]) -> tuple[float, dict]:
        """(free+locked USDT, {symbol: base qty}) — needs keys."""
        bal = self._exchange().fetch_balance()
        total = bal.get("total", {})
        qty = {s: float(total.get(s.split("/")[0], 0.0) or 0.0) for s in symbols}
        return float(total.get(QUOTE, 0.0) or 0.0), qty

    def rebalance_to(self, target: dict, cash: float, qty: dict, prices: dict,
                     day_key: str = "") -> tuple[float, list[dict]]:
        """Move holdings toward `target` weights. Returns (equity, intents)."""
        equity = cash + sum(qty.get(s, 0.0) * prices[s] for s in target)
        self.risk.update_equity(equity, day_key)
        adjusted, notes = self.risk.check_weights(target)
        adjusted = {s: max(0.0, w) for s, w in adjusted.items()}   # spot: no shorts

        ex = self._exchange()
        intents = []
        # sells first so their proceeds fund the buys
        order = sorted(adjusted, key=lambda s: adjusted[s] * equity - qty.get(s, 0.0) * prices[s])
        for s in order:
            px = prices[s]
            held_val = qty.get(s, 0.0) * px
            delta = adjusted[s] * equity - held_val
            side = "buy" if delta > 0 else "sell"
            amount = abs(delta) / px
            if side == "sell":
                amount = min(amount, qty.get(s, 0.0))
            try:
                amount = float(ex.amount_to_precision(s, amount)) if amount > 0 else 0.0
            except Exception:   # ccxt raises when the delta is below one lot step → nothing to trade
                amount = 0.0
            notional = amount * px
            min_cost = ((ex.markets[s].get("limits") or {}).get("cost") or {}).get("min") or 5.0
            intent = {"symbol": s, "side": side, "weight": round(adjusted[s], 4),
                      "amount": amount, "notional": round(notional, 2)}
            if notional < max(min_cost, self.band * equity) and not self.risk.halted:
                intent["status"] = "skipped(band)"
            elif notional < min_cost:
                intent["status"] = "skipped(dust)"
            elif self.dry_run:
                intent["status"] = "simulated"
            else:
                try:
                    ex.create_order(s, "market", side, amount)
                    intent["status"] = "sent"
                except Exception as e:  # keep going: one failed leg shouldn't block the other
                    intent["status"] = f"error:{str(e)[:80]}"
            intents.append(intent)
        if notes:
            intents.append({"risk_notes": notes})
        return equity, intents
