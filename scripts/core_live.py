"""Core book daily runner — vol-target × 4-speed trend, spot, long-only (see autotrader/core.py).

Three modes, safest by default:

  (default)   PAPER  — simulated book on real Binance prices, no keys, fees charged.
  --account   reads your real spot balances (needs BINANCE_SPOT_API_KEY/SECRET) and
              prints the orders it WOULD send (dry-run).
  --account --arm [--testnet]           sends orders on the spot testnet
  --account --arm --confirm-live        sends REAL orders on mainnet

Every run also tracks two benchmarks started the same day with the same capital:
50/50 BTC/ETH buy&hold and BTC buy&hold — so the core is always judged against
"just holding", on real prices.

Scheduled hourly at :20; only the first successful run of each UTC day trades (the
others are no-ops), so a Mac that was asleep or offline at 00:20 catches up at wake.
State: .cache/core_<mode>.json   Ledger: .cache/core_<mode>.jsonl   Log: .cache/core.log
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from autotrader.core import CORE_UNIVERSE, FEE, SMA_WINDOWS, core_weights, closed_daily
from autotrader.data.feed import latest_ohlcv
from autotrader.live.alerts import notify
from autotrader.live.spot_broker import SpotBroker
from autotrader.risk import RiskLimits, RiskManager

START_CASH = 5000.0
LOG_PATH = ".cache/core.log"
# Backtest worst drawdown is ~-38%, so the kill-switch sits beyond it: it is a
# catastrophe brake (exchange/bug/black swan), not a strategy exit.
LIMITS = RiskLimits(max_gross_exposure=1.0, max_position_pct=0.5,
                    max_drawdown_pct=0.45, max_daily_loss_pct=0.15)


def log(msg: str) -> None:
    line = f"{datetime.now(timezone.utc).isoformat()[:19]} {msg}"
    print(line)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def load(path: str) -> dict:
    p = Path(path)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            pass
    return {}


def daily_closes(symbols: list[str]) -> pd.DataFrame:
    """Last ~400 CLOSED daily candles per coin. Raises if any feed comes back empty."""
    cols = {}
    for s in symbols:
        df = closed_daily(latest_ohlcv(s, "1d", 400))
        if df.empty:
            raise RuntimeError(f"no daily candles for {s}")
        cols[s] = df["close"]
    return pd.DataFrame(cols).dropna()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--account", action="store_true", help="use real spot balances")
    ap.add_argument("--arm", action="store_true", help="actually send orders")
    ap.add_argument("--testnet", action="store_true", help="spot testnet (testnet.binance.vision)")
    ap.add_argument("--confirm-live", action="store_true", help="required for REAL mainnet orders")
    ap.add_argument("--force", action="store_true", help="run even if today's rebalance is already done")
    a = ap.parse_args()

    if a.account and a.arm and not a.testnet and not a.confirm_live:
        print("refusing: real mainnet orders need --confirm-live")
        return 2
    mode = "paper" if not a.account else ("testnet" if a.testnet else "mainnet")
    if a.account and not a.arm:
        mode += "_dryrun"
    state_path, ledger_path = f".cache/core_{mode}.json", f".cache/core_{mode}.jsonl"
    Path(".cache").mkdir(exist_ok=True)
    today = datetime.now(timezone.utc).date().isoformat()
    if load(state_path).get("last_run") == today and not a.force:
        return 0   # already done today — the hourly scheduler retries are no-ops
    log(f"=== core book · {mode} · {today} ===")

    try:
        closes = daily_closes(CORE_UNIVERSE)
    except Exception as e:  # network blip on wake etc. — skip the day, never trade blind
        log(f"  skip: data unavailable ({str(e)[:80]})")
        return 1
    target = core_weights(closes).iloc[-1].to_dict()
    bar = closes.index[-1].date().isoformat()

    risk = RiskManager(LIMITS, state_path=f".cache/risk_core_{mode}.json")
    broker = SpotBroker(api_key=os.environ.get("BINANCE_SPOT_API_KEY", ""),
                        secret=os.environ.get("BINANCE_SPOT_SECRET", ""),
                        dry_run=not a.arm, testnet=a.testnet, risk=risk)
    prices = broker.prices(CORE_UNIVERSE)

    st = load(state_path)
    if a.account:
        if not broker.api_key:
            log("  skip: --account needs BINANCE_SPOT_API_KEY / BINANCE_SPOT_SECRET")
            return 2
        cash, qty = broker.holdings(CORE_UNIVERSE)
    else:
        if not st:
            st = {"cash": START_CASH, "qty": {s: 0.0 for s in CORE_UNIVERSE}}
        cash, qty = st["cash"], st["qty"]

    equity, intents = broker.rebalance_to(target, cash, qty, prices, day_key=today)
    if not st.get("bench"):   # benchmarks start on the first run, same capital
        st["start"] = today
        st["start_equity"] = equity
        st["bench"] = {"hold_5050": {s: equity / len(CORE_UNIVERSE) / prices[s] for s in CORE_UNIVERSE},
                       "hold_btc": {"BTC/USDT": equity / prices["BTC/USDT"]}}

    if not a.account:   # paper fills at the current price, fee charged
        for i in intents:
            if i.get("status") != "simulated":
                continue
            s, n = i["symbol"], i["notional"]
            if i["side"] == "buy":
                st["qty"][s] += i["amount"]
                st["cash"] -= n * (1 + FEE)
            else:
                st["qty"][s] -= i["amount"]
                st["cash"] += n * (1 - FEE)
        equity = st["cash"] + sum(st["qty"][s] * prices[s] for s in CORE_UNIVERSE)

    bench = {k: sum(q * prices[s] for s, q in units.items()) for k, units in st["bench"].items()}
    st["last_run"] = today
    Path(state_path).write_text(json.dumps(st, indent=2))

    rec = {"ts": datetime.now(timezone.utc).isoformat()[:19], "bar": bar, "equity": round(equity, 2),
           "bench": {k: round(v, 2) for k, v in bench.items()},
           "weights": {s: round(w, 4) for s, w in target.items()},
           "prices": prices, "orders": [i for i in intents if "symbol" in i], "halted": risk.halted}
    with open(ledger_path, "a") as f:
        f.write(json.dumps(rec) + "\n")

    # ---- report ----
    for s in CORE_UNIVERSE:
        on = [n for n in SMA_WINDOWS if closes[s].iloc[-1] > closes[s].tail(n).mean()]
        trend = f"trend {len(on)}/{len(SMA_WINDOWS)} on" + (f" (above SMA {','.join(map(str, on))})" if on else " → cash")
        log(f"  {s:9} target {target[s]:5.1%}  {trend}")
    for i in intents:
        if "symbol" in i:
            log(f"    {i['side'].upper():4} {i['symbol']:9} ${i['notional']:>9,.2f}  [{i['status']}]")
        else:
            log(f"    risk: {'; '.join(i['risk_notes'])}")
    s0 = st["start_equity"]
    log(f"  EQUITY ${equity:,.2f} ({equity / s0 - 1:+.1%} since {st['start']}) | "
        f"hold 50/50 {bench['hold_5050'] / s0 - 1:+.1%} | hold BTC {bench['hold_btc'] / s0 - 1:+.1%}")
    if risk.halted:
        log(f"  🛑 HALTED: {risk.state.halt_reason} — book flattened to USDT, manual resume required")
        notify("Core book HALTED", risk.state.halt_reason)
    return 0


if __name__ == "__main__":
    sys.exit(main())
