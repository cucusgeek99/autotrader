"""Live exchange execution — SAFE BY DEFAULT.

This is the only component that can move real money, so it is wrapped in guards:

  * **dry_run=True by default** — it logs the orders it *would* place and sends
    nothing. You must pass dry_run=False explicitly to trade.
  * **testnet supported** — `testnet=True` routes to the exchange sandbox (fake
    money, real API) so you can validate execution before risking a cent.
  * **risk-gated** — every rebalance passes through a RiskManager; a tripped
    kill-switch flattens and blocks new risk.
  * **needs explicit API keys** — no keys → read-only (prices/limits) only.

Recommended path to real money: dry_run on mainnet → testnet live → tiny real
size → scale. Never skip a step.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from autotrader.risk import RiskManager


@dataclass
class LiveBroker:
    exchange_id: str = "binance"
    api_key: str = ""
    secret: str = ""
    dry_run: bool = True          # MUST be set False to place real orders
    testnet: bool = False
    risk: RiskManager = field(default_factory=RiskManager)
    _ex: object = None

    def _exchange(self):
        if self._ex is None:
            import ccxt
            cfg = {"enableRateLimit": True}
            if self.api_key:
                cfg.update(apiKey=self.api_key, secret=self.secret)
            self._ex = getattr(ccxt, self.exchange_id)(cfg)
            if self.testnet and hasattr(self._ex, "set_sandbox_mode"):
                self._ex.set_sandbox_mode(True)
            self._ex.load_markets()
        return self._ex

    def mode(self) -> str:
        if self.dry_run:
            return "DRY-RUN (no orders sent)"
        return "TESTNET (sandbox)" if self.testnet else "LIVE (REAL MONEY)"

    def equity(self) -> float:
        return self.equity_usdt()

    def equity_usdt(self) -> float:
        """Total USDT-equivalent balance (needs keys). 0.0 if unavailable."""
        try:
            bal = self._exchange().fetch_balance()
            return float(bal.get("total", {}).get("USDT", 0.0))
        except Exception:
            return 0.0

    def prices(self, symbols: list[str]) -> dict:
        ex = self._exchange()
        out = {}
        for s in symbols:
            try:
                out[s] = float(ex.fetch_ticker(s)["last"])
            except Exception:
                pass
        return out

    def rebalance_to(self, target_weights: dict, equity: float, prices: dict) -> list[dict]:
        """Risk-gate the targets, then place (or simulate) market orders.

        Returns the list of order intents (each tagged sent/simulated/blocked).
        """
        self.risk.update_equity(equity)
        adjusted, notes = self.risk.check_weights(target_weights)
        intents = []
        ex = self._exchange() if not self.dry_run else None
        for sym, w in adjusted.items():
            price = prices.get(sym)
            if not price:
                continue
            notional = w * equity
            amount = abs(notional) / price
            side = "buy" if w > 0 else "sell"
            intent = {"symbol": sym, "side": side, "weight": round(w, 4),
                      "notional": round(notional, 2), "amount": round(amount, 8)}
            if amount * price < 5:           # below typical min-notional → skip
                intent["status"] = "skipped(dust)"
            elif self.dry_run:
                intent["status"] = "simulated"
            else:
                try:
                    ex.create_order(sym, "market", side, amount)
                    intent["status"] = "sent"
                except Exception as e:
                    intent["status"] = f"error:{e}"
            intents.append(intent)
        intent_notes = notes
        if intent_notes:
            intents.append({"risk_notes": intent_notes})
        return intents
