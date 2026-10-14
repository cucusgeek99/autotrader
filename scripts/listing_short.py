"""New-listing short — do freshly listed Binance perps drift down (low float, unlocks)?

Pre-registered protocol (fixed before results):
  * universe: every USDT perp listed 2023-01-01 → 2026-07-15 INCLUDING delisted
    (status SETTLING) — no survivorship bias
  * trade: short the perp at the close of day D after listing, hold H days, hedged with
    an equal-$ BTC perp long (isolates the listing effect from the market)
  * costs: taker 0.05%/side on both legs + REAL funding paid/received on both legs
  * DEV = listings before 2025-04-01 (choose D, H there), TEST = listings after
  * PASS = TEST mean net > 0 with t > 2, AND not driven by a handful of names
    (median also > 0, and still positive with the top 5% winners removed)

Run: .venv/bin/python scripts/listing_short.py [--fetch]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import ccxt
import numpy as np
import pandas as pd

CACHE = Path(".cache/scalp/listings")
TAKER = 0.0005
DEV_END = pd.Timestamp("2025-04-01")
START, END = pd.Timestamp("2023-01-01"), pd.Timestamp("2026-07-15")
DAYS = 100


def fetch() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    ex = ccxt.binanceusdm()
    lst = pd.read_pickle(".cache/scalp/listings.pkl")
    lst = lst[(lst.date >= START) & (lst.date <= END)]
    syms = list(lst.sym) + ["BTCUSDT"]
    for i, s in enumerate(syms):
        out = CACHE / f"{s}.pkl"
        if out.exists():
            continue
        t0 = int(lst.loc[lst.sym == s, "onboard"].iloc[0]) if s != "BTCUSDT" else int(START.timestamp() * 1000)
        n = DAYS if s != "BTCUSDT" else 1400
        try:
            k = []
            cur = t0
            while len(k) < n:
                b = ex.fapiPublicGetKlines({"symbol": s, "interval": "1d", "startTime": cur, "limit": 1000})
                if not b:
                    break
                k += b
                cur = int(b[-1][0]) + 86_400_000
                if len(b) < 1000:
                    break
            f, cur = [], t0
            end_ms = t0 + n * 86_400_000
            while cur < end_ms:
                b = ex.fapiPublicGetFundingRate({"symbol": s, "startTime": cur, "limit": 1000})
                if not b:
                    break
                f += b
                cur = int(b[-1]["fundingTime"]) + 1
                if len(b) < 1000:
                    break
        except Exception as e:
            print("  skip", s, str(e)[:60])
            continue
        px = pd.Series([float(x[4]) for x in k], index=pd.to_datetime([int(x[0]) for x in k], unit="ms"))
        fr = pd.Series([float(x["fundingRate"]) for x in f],
                       index=pd.to_datetime([int(x["fundingTime"]) for x in f], unit="ms"), dtype=float)
        pd.to_pickle({"close": px, "funding": fr}, out)
        if i % 50 == 0:
            print(f"  {i}/{len(syms)} {s}", flush=True)
        time.sleep(0.05)


def trade(tok: dict, btc: dict, d: int, h: int) -> float | None:
    """Net return of: short token / long BTC, equal $, from close day d to close day d+h."""
    c = tok["close"]
    if len(c) <= d + h:
        return None
    t_in, t_out = c.index[d], c.index[d + h]
    r_tok = c.iloc[d + h] / c.iloc[d] - 1
    b = btc["close"]
    if t_in not in b.index or t_out not in b.index:
        return None
    r_btc = b[t_out] / b[t_in] - 1
    fund = lambda fr: fr[(fr.index > t_in) & (fr.index <= t_out)].sum()
    # short receives positive funding; long BTC pays it
    return -r_tok + r_btc + fund(tok["funding"]) - fund(btc["funding"]) - 4 * TAKER


def stats(x: np.ndarray) -> str:
    if len(x) < 5:
        return f"n={len(x)}"
    t = x.mean() / x.std() * np.sqrt(len(x))
    trim = np.sort(x)[: int(len(x) * 0.95)]
    return (f"n={len(x):3}  mean {x.mean():+6.1%}  median {np.median(x):+6.1%}  win {np.mean(x > 0):4.0%}  "
            f"t {t:+5.2f}  mean w/o top5% {trim.mean():+6.1%}")


def main() -> None:
    if "--fetch" in sys.argv:
        fetch()
    lst = pd.read_pickle(".cache/scalp/listings.pkl").set_index("sym")
    btc = pd.read_pickle(CACHE / "BTCUSDT.pkl")
    toks = {p.stem: pd.read_pickle(p) for p in CACHE.glob("*.pkl") if p.stem != "BTCUSDT"}
    print(f"{len(toks)} listings loaded ({sum(lst.loc[list(toks), 'status'] == 'SETTLING')} later delisted)\n")

    grid = [(d, h) for d in (1, 3, 7, 14) for h in (14, 30, 60)]
    res = {}
    for d, h in grid:
        dev, test = [], []
        for s, tok in toks.items():
            r = trade(tok, btc, d, h)
            if r is None:
                continue
            (dev if lst.loc[s, "date"] < DEV_END else test).append(r)
        res[(d, h)] = (np.array(dev), np.array(test))

    print("DEV grid (short day D after listing, hold H days, BTC-hedged, net of fees+funding)")
    for (d, h), (dev, _) in res.items():
        print(f"   D={d:2} H={h:2}  {stats(dev)}")
    best = max(res, key=lambda k: res[k][0].mean() / res[k][0].std() * np.sqrt(len(res[k][0])))
    print(f"\nbest on DEV: D={best[0]} H={best[1]}")
    print(f"   DEV   {stats(res[best][0])}")
    print(f"   TEST  {stats(res[best][1])}   ← the verdict (listings after {DEV_END.date()}, never used to choose)")
    print("\nTEST for every config (to see whether the effect is broad or a single lucky cell):")
    for (d, h), (_, test) in res.items():
        print(f"   D={d:2} H={h:2}  {stats(test)}")


if __name__ == "__main__":
    main()
