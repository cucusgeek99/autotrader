"""Dip-buyer tactical sleeve — PAPER on live prices (validate before arming).

A mean-reversion entry/exit system that runs alongside (NOT on top of) the live
lowbeta hold. It harvests deep single-coin dislocations: buy when a coin is
oversold AND below its lower Bollinger band, exit at the mean — with a HARD STOP
that the clean delisting-inclusive backtest proved is non-negotiable (without it
the falling knives like LUNA eat the edge; with a ~-25% stop the strategy stays
+EV even with all the dead coins included).

This is PAPER: it tracks a hypothetical book on real Binance daily closes and logs
exactly what it WOULD do, so you can watch it trigger in real time before we wire
real orders / integrate it. Run daily.

  ENTRY : close < lowerBB(20, 2sigma)  AND  RSI(14) < 30
  EXIT  : close >= SMA20 (mean target)  OR  return <= -STOP  OR  held >= TIMEOUT days

Run: .venv/bin/python scripts/dipbuyer_live.py
State: .cache/dipbuyer_paper.json   Log: .cache/dipbuyer_paper.log
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from autotrader.data.feed import fetch_ohlcv_extended

UNIVERSE = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", "ADA/USDT",
            "DOGE/USDT", "AVAX/USDT", "LINK/USDT", "LTC/USDT", "DOT/USDT", "TRX/USDT"]

# --- params (match the clean backtest) ---
RSI_N, BB_N, BB_K = 14, 20, 2.0
STOP = 0.25            # hard stop (falling-knife protection) — backtest: -25% defensible after gap-risk haircut
TIMEOUT_DAYS = 30
SLOT_FRAC = 0.25       # each dip position = 25% of equity at entry
MAX_SLOTS = 4
FEE = 0.0006 + 0.0005  # taker + slippage, per side
START_CASH = 5000.0

STATE_PATH = ".cache/dipbuyer_paper.json"
LOG_PATH = ".cache/dipbuyer_paper.log"


def rsi(s: pd.Series, n: int = RSI_N) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def load_state() -> dict:
    p = Path(STATE_PATH)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            pass
    return {"cash": START_CASH, "positions": {}, "closed": [], "last_run": ""}


def save_state(st: dict) -> None:
    Path(STATE_PATH).write_text(json.dumps(st, indent=2))


def log(msg: str) -> None:
    line = f"{datetime.now(timezone.utc).isoformat()[:19]} {msg}"
    print(line)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def indicators(sym: str):
    """Latest (close, rsi, sma20, lowerBB, %B) for a coin from live daily candles."""
    df = fetch_ohlcv_extended(sym, "1d", 400, "binance", ".cache", use_cache=False)
    c = df["close"].dropna()
    sma = c.rolling(BB_N).mean()
    sd = c.rolling(BB_N).std()
    lo = sma - BB_K * sd
    up = sma + BB_K * sd
    pb = (c - lo) / (up - lo)
    return {"close": float(c.iloc[-1]), "rsi": float(rsi(c).iloc[-1]),
            "sma20": float(sma.iloc[-1]), "lower": float(lo.iloc[-1]),
            "pctb": float(pb.iloc[-1]), "date": c.index[-1].date().isoformat()}


def main() -> None:
    st = load_state()
    today = datetime.now(timezone.utc).date().isoformat()
    log(f"=== dip-buyer PAPER sleeve · run {today} ===")

    ind = {}
    for s in UNIVERSE:
        try:
            ind[s] = indicators(s)
        except Exception as e:
            log(f"  ! {s} fetch failed: {str(e)[:60]}")

    # 1) EXITS first (mean / stop / timeout) on currently-held positions
    for s in list(st["positions"].keys()):
        if s not in ind:
            continue
        pos = st["positions"][s]
        px = ind[s]["close"]
        ret = px / pos["entry_px"] - 1
        held = (pd.Timestamp(today) - pd.Timestamp(pos["entry_date"])).days
        reason = None
        if ret <= -STOP:
            reason = f"STOP {ret:+.1%}"
        elif px >= ind[s]["sma20"]:
            reason = f"MEAN-exit {ret:+.1%}"
        elif held >= TIMEOUT_DAYS:
            reason = f"TIMEOUT {held}d {ret:+.1%}"
        if reason:
            proceeds = pos["qty"] * px * (1 - FEE)
            st["cash"] += proceeds
            st["closed"].append({"sym": s, "entry_px": pos["entry_px"], "exit_px": px,
                                 "ret": ret, "reason": reason, "entry_date": pos["entry_date"],
                                 "exit_date": today})
            log(f"  EXIT  {s:10} {reason:18} entry {pos['entry_px']:.5g} → {px:.5g}")
            del st["positions"][s]

    # 2) mark-to-market equity
    pos_val = sum(st["positions"][s]["qty"] * ind[s]["close"]
                  for s in st["positions"] if s in ind)
    equity = st["cash"] + pos_val

    # 3) ENTRIES: oversold dip, not already held, slot available
    buys, watch = [], []
    for s in UNIVERSE:
        if s not in ind or s in st["positions"]:
            continue
        d = ind[s]
        if d["close"] < d["lower"] and d["rsi"] < 30:  # RSI<30 AND below lower band
            buys.append(s)
        elif d["rsi"] < 35 and d["pctb"] < 0.15:
            watch.append(s)

    for s in buys:
        if len(st["positions"]) >= MAX_SLOTS:
            log(f"  (signal {s} but {MAX_SLOTS} slots full — skip)")
            continue
        budget = min(SLOT_FRAC * equity, st["cash"])
        if budget < 50:
            log(f"  (signal {s} but no cash — skip)")
            continue
        px = ind[s]["close"]
        qty = budget * (1 - FEE) / px
        st["cash"] -= budget
        st["positions"][s] = {"entry_px": px, "qty": qty, "entry_date": today}
        log(f"  BUY   {s:10} oversold dip  px {px:.5g}  ~${budget:,.0f}  (RSI {ind[s]['rsi']:.0f}, %B {ind[s]['pctb']:.2f})")

    # 4) status
    pos_val = sum(st["positions"][s]["qty"] * ind[s]["close"]
                  for s in st["positions"] if s in ind)
    equity = st["cash"] + pos_val
    n_closed = len(st["closed"])
    wins = [t for t in st["closed"] if t["ret"] > 0]
    wr = len(wins) / n_closed if n_closed else 0.0
    log(f"  EQUITY ${equity:,.0f} | cash ${st['cash']:,.0f} | {len(st['positions'])} open | "
        f"{n_closed} closed (win {wr:.0%})")
    if st["positions"]:
        for s, p in st["positions"].items():
            if s in ind:
                r = ind[s]["close"] / p["entry_px"] - 1
                log(f"    HOLD {s:10} {r:+6.1%}  (entry {p['entry_date']}, stop @ -{STOP:.0%})")
    if not buys:
        msg = "no buy-zone today"
        if watch:
            msg += f" · watching {', '.join(w.split('/')[0] for w in watch)}"
        log(f"  {msg}")

    st["last_run"] = today
    save_state(st)


if __name__ == "__main__":
    main()
