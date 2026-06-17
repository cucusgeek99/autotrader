"""Minimal Binance USDⓈ-M futures REST client + broker.

ccxt deprecated its futures-testnet sandbox, so we talk to the futures API
directly (signed HMAC requests). Same code serves testnet and mainnet — only the
base URL differs. The book is long/short market-neutral, which is why this is
futures, not spot.

SAFE BY DEFAULT: the broker is dry_run unless explicitly armed, every rebalance
passes the RiskManager, and orders are sized as the *delta* from current
positions (so repeated ticks don't pyramid).
"""
from __future__ import annotations

import hashlib
import hmac
import json
import math
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

from autotrader.risk import RiskManager

TESTNET = "https://testnet.binancefuture.com"
MAINNET = "https://fapi.binance.com"


class FuturesClient:
    def __init__(self, api_key: str, secret: str, testnet: bool = True):
        self.key, self.secret = api_key, secret
        self.base = TESTNET if testnet else MAINNET
        self._filters = None

    def _get(self, path: str, params: dict | None = None):
        url = self.base + path + ("?" + urllib.parse.urlencode(params) if params else "")
        return json.loads(urllib.request.urlopen(url, timeout=15).read())

    def _signed(self, method: str, path: str, params: dict):
        params = {**params, "timestamp": int(time.time() * 1000), "recvWindow": 5000}
        q = urllib.parse.urlencode(params)
        sig = hmac.new(self.secret.encode(), q.encode(), hashlib.sha256).hexdigest()
        url = f"{self.base}{path}?{q}&signature={sig}"
        req = urllib.request.Request(url, method=method, headers={"X-MBX-APIKEY": self.key})
        return json.loads(urllib.request.urlopen(req, timeout=15).read())

    def balance_usdt(self) -> float:
        for a in self._signed("GET", "/fapi/v2/balance", {}):
            if a["asset"] == "USDT":
                return float(a["balance"])
        return 0.0

    def price(self, sym: str) -> float:
        return float(self._get("/fapi/v1/ticker/price", {"symbol": sym})["price"])

    def positions(self) -> dict:
        return {p["symbol"]: float(p["positionAmt"])
                for p in self._signed("GET", "/fapi/v2/positionRisk", {})}

    def filters(self) -> dict:
        """symbol -> (step_size, min_notional)."""
        if self._filters is None:
            info = self._get("/fapi/v1/exchangeInfo")
            f = {}
            for s in info["symbols"]:
                step, minnot = None, 5.0
                for fl in s["filters"]:
                    if fl["filterType"] == "LOT_SIZE":
                        step = float(fl["stepSize"])
                    elif fl["filterType"] == "MIN_NOTIONAL":
                        minnot = float(fl.get("notional", 5))
                f[s["symbol"]] = (step, minnot)
            self._filters = f
        return self._filters

    def fmt_qty(self, sym: str, qty: float) -> str:
        step = self.filters().get(sym, (None, 5))[0] or 0.001
        rounded = math.floor(abs(qty) / step) * step
        prec = max(0, round(-math.log10(step)))
        return f"{rounded:.{prec}f}"

    def market_order(self, sym: str, side: str, qty: str) -> dict:
        return self._signed("POST", "/fapi/v1/order",
                            {"symbol": sym, "side": side, "type": "MARKET", "quantity": qty})


@dataclass
class FuturesBroker:
    client: FuturesClient
    risk: RiskManager = field(default_factory=RiskManager)
    dry_run: bool = True
    testnet: bool = True

    def mode(self) -> str:
        if self.dry_run:
            return "DRY-RUN (no orders sent)"
        return "TESTNET (fake money)" if self.testnet else "LIVE (REAL MONEY)"

    def equity(self) -> float:
        try:
            return self.client.balance_usdt()
        except Exception:
            return 0.0

    def rebalance_to(self, target_weights: dict, equity: float) -> list[dict]:
        """Order the DELTA from current positions to reach target weights."""
        self.risk.update_equity(equity)
        adjusted, notes = self.risk.check_weights(target_weights)
        cur = {} if self.dry_run else self.client.positions()
        intents = []
        for symslash, w in adjusted.items():
            sym = symslash.replace("/", "")
            try:
                price = self.client.price(sym)
            except Exception:
                continue
            target_amt = w * equity / price
            delta = target_amt - cur.get(sym, 0.0)
            qty_str = self.client.fmt_qty(sym, delta)
            qty = float(qty_str)
            minnot = self.client.filters().get(sym, (None, 5))[1]
            if qty == 0 or qty * price < minnot:
                continue
            side = "BUY" if delta > 0 else "SELL"
            it = {"symbol": sym, "side": side, "weight": round(w, 4),
                  "qty": qty_str, "notional": round(qty * price, 2)}
            if self.dry_run:
                it["status"] = "simulated"
            else:
                try:
                    self.client.market_order(sym, side, qty_str)
                    it["status"] = "sent"
                except Exception as e:
                    it["status"] = f"error:{str(e)[:80]}"
            intents.append(it)
        if notes:
            intents.append({"risk_notes": notes})
        return intents
