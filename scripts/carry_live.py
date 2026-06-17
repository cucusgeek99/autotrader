"""Delta-neutral funding-carry sleeve — PAPER on live prices (validate before arming).

The genuine market-neutral carry trade: for each coin whose perpetual funding is
sufficiently POSITIVE, hold SPOT LONG + PERP SHORT in equal dollar notional. Net
price delta ≈ 0 (the two legs cancel), so you don't bet on direction — you simply
COLLECT the funding the perp longs pay the shorts every 8h.

This is the only sleeve in the lab that earns a *flow* (funding income) rather than
a directional bet, so it's the real candidate for "green most days". But it is NOT a
free lunch — the audit pinned it at a mundane Sharpe ~2-4 / ~5%/yr on short notional,
and the real risks are:
  • BASIS RISK — spot and perp don't track perfectly; the hedge has tracking error.
    We capture it honestly from real spot vs perp prices (net mark-to-market of the
    two legs), not as zero.
  • FUNDING FLIP — funding turns negative → you'd PAY. We exit a pair when its
    funding drops below the exit floor (hysteresis to avoid churn).
  • COSTS — 4 legs round-trip (enter+exit, spot+perp). Funding must out-earn them,
    so we only enter above a threshold and hold (no fast churn).
  • CAPITAL — fully-funded both legs would be 2x capital per unit of carry. Real
    desks margin the hedged short at ~3x (liquidation only on a basis blowout), so
    capital/unit = notional × (1 + 1/PERP_LEV). Return-on-capital ≈ funding / that.

PAPER: tracks a hypothetical book on real Binance spot + perp prices and real settled
funding, logs exactly what it WOULD do, so you can watch it accrue funding in real
time before we wire real orders. Run daily (captures the 3 daily 8h settlements).

  ENTER : annualized funding > ENTER_ANN   (build spot-long / perp-short, equal $)
  HOLD  : accrue realized funding each settlement; mark basis to market
  EXIT  : annualized funding < EXIT_ANN  (or negative)  OR  basis stop

Run: .venv/bin/python scripts/carry_live.py
State: .cache/carry_paper.json   Log: .cache/carry_paper.log
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import ccxt

from autotrader.data.derivatives import fetch_funding_history

UNIVERSE = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", "ADA/USDT",
            "DOGE/USDT", "AVAX/USDT", "LINK/USDT", "LTC/USDT", "DOT/USDT", "TRX/USDT"]

# --- params ---
ENTER_ANN = 0.06       # enter when annualized funding > 6%/yr (must out-earn round-trip cost)
EXIT_ANN = 0.01        # exit when annualized funding < 1%/yr (hysteresis; also exit if negative)
MAX_SLOTS = 5
PERP_LEV = 3.0         # margin on the hedged short leg (desk standard; capital/unit = notional*(1+1/lev))
BASIS_STOP = 0.05      # exit if net basis P&L on a pair drops below -5% of notional (hedge broke)
FEE_LEG = 0.0006 + 0.0002  # taker + slippage, PER LEG (spot legs are liquid)
START_CASH = 5000.0

STATE_PATH = ".cache/carry_paper.json"
LOG_PATH = ".cache/carry_paper.log"


def load_state() -> dict:
    p = Path(STATE_PATH)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            pass
    return {"cash": START_CASH, "positions": {}, "closed": [],
            "lifetime_funding": 0.0, "last_run": ""}


def save_state(st: dict) -> None:
    Path(STATE_PATH).write_text(json.dumps(st, indent=2))


def log(msg: str) -> None:
    line = f"{datetime.now(timezone.utc).isoformat()[:19]} {msg}"
    print(line)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def market_data() -> dict:
    """Per coin: spot price, perp price, current annualized funding."""
    spot = ccxt.binance()
    perp = ccxt.binanceusdm()
    spot.load_markets()
    perp.load_markets()
    out = {}
    for s in UNIVERSE:
        try:
            sp = float(spot.fetch_ticker(s)["last"])
            pp = float(perp.fetch_ticker(s)["last"])
            fr = float(perp.fetch_funding_rate(s)["fundingRate"])  # per 8h
            out[s] = {"spot": sp, "perp": pp, "fund8h": fr, "ann": fr * 3 * 365}
        except Exception as e:
            log(f"  ! {s} data failed: {str(e)[:50]}")
    return out


def accrue_funding(sym: str, since_ms: int, notional: float) -> tuple[float, int]:
    """Sum realized funding (USD) on a SHORT perp notional since `since_ms`.

    We're short, so POSITIVE funding pays us. Returns (usd_collected, latest_ts_ms).
    """
    hist = fetch_funding_history(sym, limit=50, use_cache=False)
    if hist.empty:
        return 0.0, since_ms
    # index unit varies (datetime64[ms] vs [ns]) — go through .timestamp() to be unit-proof
    ts = [int(t.timestamp() * 1000) for t in hist.index]
    rates = hist["funding"].tolist()
    usd, latest = 0.0, since_ms
    for t, r in zip(ts, rates):
        if t > since_ms:
            usd += r * notional       # short collects positive funding
            latest = max(latest, t)
    return usd, latest


def main() -> None:
    st = load_state()
    today = datetime.now(timezone.utc).date().isoformat()
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    log(f"=== delta-neutral carry PAPER sleeve · run {today} ===")

    md = market_data()

    # 1) ACCRUE funding on open pairs + mark basis to market
    for s, pos in list(st["positions"].items()):
        if s not in md:
            continue
        usd, latest = accrue_funding(s, pos["last_funding_ts"], pos["notional"])
        if usd != 0.0:
            st["cash"] += usd                 # funding settles to wallet
            pos["accrued_funding"] += usd
            st["lifetime_funding"] += usd
            pos["last_funding_ts"] = latest
            log(f"  FUND  {s:10} +${usd:6.3f}  (cumulative ${pos['accrued_funding']:.2f})")

    # 2) EXITS — funding decayed / went negative / basis hedge broke
    for s in list(st["positions"].keys()):
        if s not in md:
            continue
        pos = st["positions"][s]
        ann = md[s]["ann"]
        basis = _basis_pnl(pos, md[s])
        reason = None
        if ann < EXIT_ANN:
            reason = f"funding decayed {ann*100:+.1f}%/yr"
        elif basis < -BASIS_STOP * pos["notional"]:
            reason = f"BASIS-STOP {basis/pos['notional']*100:+.1f}%"
        if reason:
            exit_fee = 2 * FEE_LEG * pos["notional"]   # close both legs
            st["cash"] += pos["capital"] + basis - exit_fee
            held = (datetime.now(timezone.utc).date()
                    - datetime.fromisoformat(pos["entry_date"]).date()).days
            st["closed"].append({"sym": s, "funding": pos["accrued_funding"],
                                 "basis": basis, "held_days": held,
                                 "net": pos["accrued_funding"] + basis - exit_fee
                                        - pos["entry_fee"], "reason": reason})
            log(f"  EXIT  {s:10} {reason:24} funding +${pos['accrued_funding']:.2f} "
                f"basis ${basis:+.2f} held {held}d")
            del st["positions"][s]

    # 3) equity mark
    equity = _equity(st, md)

    # 4) ENTRIES — best positive-funding coins not already held
    cands = sorted(
        [(s, md[s]["ann"]) for s in md
         if s not in st["positions"] and md[s]["ann"] > ENTER_ANN],
        key=lambda x: x[1], reverse=True)
    for s, ann in cands:
        if len(st["positions"]) >= MAX_SLOTS:
            break
        cap_per = equity / MAX_SLOTS              # target capital per slot
        notional = cap_per / (1 + 1 / PERP_LEV)   # spot leg = perp short leg
        capital = notional * (1 + 1 / PERP_LEV)   # spot cash + perp margin
        entry_fee = 2 * FEE_LEG * notional
        if st["cash"] < capital + entry_fee:
            continue
        d = md[s]
        st["cash"] -= capital + entry_fee
        st["positions"][s] = {
            "notional": notional, "capital": capital,
            "spot_entry": d["spot"], "perp_entry": d["perp"],
            "entry_date": today, "last_funding_ts": now_ms,
            "accrued_funding": 0.0, "entry_fee": entry_fee,
        }
        log(f"  ENTER {s:10} funding {ann*100:+.1f}%/yr  notional ${notional:,.0f}/leg "
            f"(cap ${capital:,.0f}, perp {PERP_LEV:g}x)")

    # 5) status
    equity = _equity(st, md)
    n_closed = len(st["closed"])
    closed_funding = sum(c["funding"] for c in st["closed"])
    log(f"  EQUITY ${equity:,.2f} | cash ${st['cash']:,.2f} | {len(st['positions'])} pairs | "
        f"lifetime funding +${st['lifetime_funding']:.2f}")
    for s, p in st["positions"].items():
        if s in md:
            ann = md[s]["ann"]
            basis = _basis_pnl(p, md[s])
            log(f"    HOLD {s:10} funding {ann*100:+5.1f}%/yr  accrued +${p['accrued_funding']:5.2f}  "
                f"basis ${basis:+6.2f}")
    if not st["positions"]:
        best = sorted(md.items(), key=lambda kv: kv[1]["ann"], reverse=True)[:3]
        watch = ", ".join(f"{s.split('/')[0]} {d['ann']*100:+.1f}%" for s, d in best)
        log(f"  no positive-carry pair above {ENTER_ANN*100:.0f}%/yr · top: {watch}")

    st["last_run"] = today
    save_state(st)


def _basis_pnl(pos: dict, d: dict) -> float:
    """Net mark-to-market of spot-long + perp-short (the hedge tracking error)."""
    spot_qty = pos["notional"] / pos["spot_entry"]
    perp_qty = pos["notional"] / pos["perp_entry"]
    spot_pnl = spot_qty * (d["spot"] - pos["spot_entry"])
    perp_pnl = -perp_qty * (d["perp"] - pos["perp_entry"])   # short
    return spot_pnl + perp_pnl


def _equity(st: dict, md: dict) -> float:
    eq = st["cash"]
    for s, p in st["positions"].items():
        eq += p["capital"] + (_basis_pnl(p, md[s]) if s in md else 0.0)
    return eq


if __name__ == "__main__":
    main()
