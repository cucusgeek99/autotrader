"""Scalping scan — 4 untested scalping mechanisms on 1 year of Binance 5m data.

Pre-registered protocol (fixed BEFORE looking at results):
  * DEV = first 8 months, TEST = last 4 months. Params are chosen on DEV only.
  * DEV coins BTC/ETH; the other 6 coins are held-out.
  * PASS = TEST net-of-fees mean > 0 at realistic perp fees (maker 0.02% / taker
    0.05%) with t > 2 pooled across coins. Anything else is reported as a fail.

Mechanisms:
  H1 cascade fade   — 5m move beyond k·σ on a volume spike → fade it, market in/out
  H2 maker dip-buy  — z(close−VWAP) < −2 → resting LIMIT buy below close; filled only
                      if the next bar trades THROUGH it (adverse selection included)
  H3 time-of-day    — pick the strongest 30-min UTC slots on DEV, trade them on TEST
  H4 basis reversion— perp/spot basis z-score > 2 → short basis until it normalizes
                      (market-neutral; 4 legs, spot fees 0.075% w/ BNB)

Run: .venv/bin/python scripts/fetch_5m.py && .venv/bin/python scripts/scalp_scan.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

COINS = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "LINK"]
DEV_COINS = ["BTC", "ETH"]
MAKER, TAKER, SPOT = 0.0002, 0.0005, 0.00075   # per side
FEES = {"taker/taker": (TAKER, TAKER), "maker 0.02%": (MAKER, MAKER), "maker 0%": (0.0, 0.0)}


def load(kind: str, coin: str) -> pd.DataFrame:
    return pd.read_pickle(f".cache/scalp/{kind}_{coin}.pkl")


def split(df: pd.DataFrame) -> pd.Timestamp:
    t0, t1 = df.index[0], df.index[-1]
    return t0 + (t1 - t0) * 8 / 12


def no_overlap(idx: np.ndarray, hold: int) -> np.ndarray:
    out, last = [], -10**9
    for i in idx:
        if i >= last + hold:
            out.append(i)
            last = i
    return np.array(out, dtype=int)


def stats(r: pd.Series) -> dict:
    r = r.dropna()
    n = len(r)
    if n < 5:
        return {"n": n, "bps": np.nan, "t": np.nan}
    return {"n": n, "bps": r.mean() * 1e4, "t": r.mean() / r.std() * np.sqrt(n)}


# ---------------- H1 cascade fade ----------------
def h1_trades(df: pd.DataFrame, k: float, hold: int, vol_mult: float = 3.0) -> pd.DataFrame:
    """Gross per-trade return; entry next open, exit close after `hold` bars (taker both)."""
    r = df.close.pct_change()
    sig = r.rolling(288).std().shift(1)
    vmed = df.volume.rolling(288).median().shift(1)
    spike = df.volume > vol_mult * vmed
    side = np.where((r < -k * sig) & spike, 1, np.where((r > k * sig) & spike, -1, 0))
    o, c = df.open.values, df.close.values
    idx = no_overlap(np.flatnonzero(side), hold)
    idx = idx[idx + hold < len(df)]
    g = side[idx] * (c[idx + hold] / o[idx + 1] - 1)
    return pd.DataFrame({"gross": g, "entry_maker": False, "exit_maker": False}, index=df.index[idx])


# ---------------- H2 maker dip-buy ----------------
def h2_trades(df: pd.DataFrame, z_in: float, offset: float, hold: int, tp: float) -> pd.DataFrame:
    tpv = df.close * df.volume
    vwap = tpv.rolling(12).sum() / df.volume.rolling(12).sum()
    dev = (df.close - vwap) / df.close
    z = (dev - dev.rolling(288).mean()) / dev.rolling(288).std()
    lo, hi, c = df.low.values, df.high.values, df.close.values
    cand = no_overlap(np.flatnonzero((z < -z_in).values), hold + 1)
    rows, idxs = [], []
    for i in cand:
        if i + 1 + hold >= len(df):
            break
        L = c[i] * (1 - offset)
        if not lo[i + 1] < L:          # strict trade-through: we only fill if price goes below
            continue
        target = L * (1 + tp)
        hit = np.flatnonzero(hi[i + 2:i + 2 + hold] > target)
        if tp > 0 and len(hit):
            rows.append((target / L - 1, True, True))
        else:
            rows.append((c[i + 1 + hold] / L - 1, True, False))
        idxs.append(df.index[i])
    return pd.DataFrame(rows, columns=["gross", "entry_maker", "exit_maker"], index=idxs)


def net(tr: pd.DataFrame, maker: float, taker: float) -> pd.Series:
    fe = np.where(tr.entry_maker, maker, taker) + np.where(tr.exit_maker, maker, taker)
    return tr.gross - fe


# ---------------- H3 time of day ----------------
def h3(perps: dict) -> list[str]:
    out = []
    pooled_dev, pooled_test = [], []
    for c, df in perps.items():
        r = df.close.pct_change().resample("30min").sum()
        cut = split(df)
        pooled_dev.append(r[r.index < cut])
        pooled_test.append(r[r.index >= cut])
    d, t = pd.concat(pooled_dev), pd.concat(pooled_test)
    slot = lambda s: s.index.strftime("%H:%M")
    g = d.groupby(slot(d)).agg(["mean", "std", "count"])
    g["t"] = g["mean"] / g["std"] * np.sqrt(g["count"])
    top, bot = g.t.nlargest(3).index, g.t.nsmallest(3).index
    out.append(f"   DEV strongest slots  long {list(top)}  short {list(bot)}")
    for name, slots, sgn in (("long", top, 1), ("short", bot, -1)):
        for lab, s in (("DEV", d), ("TEST", t)):
            rr = sgn * s[slot(s).isin(slots)]
            st = stats(rr)
            st_net = stats(rr - 2 * TAKER)
            out.append(f"   {name:5} {lab:4} n={st['n']:5}  gross {st['bps']:+6.1f}bps t={st['t']:+5.2f}  "
                       f"net(taker) {st_net['bps']:+6.1f}bps")
    return out


# ---------------- H4 basis reversion ----------------
def h4_trades(perp: pd.DataFrame, spot: pd.DataFrame, z_in: float, max_hold: int) -> pd.Series:
    j = pd.concat({"p": perp.close, "s": spot.close}, axis=1).dropna()
    b = j.p / j.s - 1
    z = ((b - b.rolling(288).mean()) / b.rolling(288).std()).values
    bv = b.values
    out, idxs, i = [], [], 288
    while i < len(b) - 1:
        if abs(z[i]) > z_in:
            sgn = -np.sign(z[i])            # z>0 → basis rich → short perp / long spot
            e = i + 1
            x = e
            while x < min(e + max_hold, len(b) - 1) and np.sign(z[x]) == np.sign(z[i]):
                x += 1
            out.append(sgn * (bv[x] - bv[e]))
            idxs.append(b.index[e])
            i = x + 1
        else:
            i += 1
    return pd.Series(out, index=idxs, dtype=float)


def report_grid(title: str, make, grid: list[dict], coins: dict, fee_mode: str = "maker 0.02%") -> None:
    """Pick the best grid point on DEV (dev coins, dev period, net at fee_mode), then show it everywhere."""
    print(f"\n{title}")
    mk, tk = FEES[fee_mode]
    best, best_t = None, -np.inf
    for p in grid:
        rs = []
        for c in DEV_COINS:
            tr = make(coins[c], **p)
            tr = tr[tr.index < split(coins[c])]
            rs.append(net(tr, mk, tk))
        s = stats(pd.concat(rs))
        if s["n"] >= 30 and s["t"] > best_t:
            best, best_t = p, s["t"]
    print(f"   best on DEV: {best}  (picked from {len(grid)} configs by DEV t-stat, net @ {fee_mode})")
    for label, cs, per in (("DEV  coins·DEV  period", DEV_COINS, "dev"),
                           ("DEV  coins·TEST period", DEV_COINS, "test"),
                           ("HELD coins·TEST period", [c for c in COINS if c not in DEV_COINS], "test")):
        trs = []
        for c in cs:
            tr = make(coins[c], **best)
            cut = split(coins[c])
            trs.append(tr[tr.index < cut] if per == "dev" else tr[tr.index >= cut])
        tr = pd.concat(trs)
        g = stats(tr.gross)
        nets = "  ".join(f"{k}: {stats(net(tr, *v))['bps']:+6.1f}bps (t {stats(net(tr, *v))['t']:+.1f})"
                         for k, v in FEES.items())
        print(f"   {label}  n={g['n']:5}  gross {g['bps']:+6.1f}bps  |  {nets}")


def main() -> None:
    perps = {c: load("perp", c) for c in COINS}
    spots = {c: load("spot", c) for c in COINS}
    print(f"data: {len(COINS)} coins · {perps['BTC'].index[0].date()} → {perps['BTC'].index[-1].date()} · "
          f"TEST starts {split(perps['BTC']).date()}")

    report_grid("H1 CASCADE FADE (market in/out)", h1_trades,
                [{"k": k, "hold": h} for k in (3, 4, 5, 6) for h in (1, 3, 6, 12, 24)], perps, "taker/taker")
    report_grid("H2 MAKER DIP-BUY (limit entry, trade-through fills)", h2_trades,
                [{"z_in": z, "offset": o, "hold": h, "tp": tp} for z in (2.0, 2.5, 3.0)
                 for o in (0.0, 0.001, 0.002) for h in (3, 6, 12) for tp in (0.0, 0.002, 0.004)], perps)

    print("\nH3 TIME-OF-DAY (30-min slots, all 8 coins pooled, taker in/out)")
    for line in h3(perps):
        print(line)

    print("\nH4 BASIS REVERSION (perp/spot, market-neutral, 4 legs)")
    rt = 2 * (MAKER + SPOT)   # maker on perp, spot 0.075% w/ BNB — round-trip both legs
    best, best_t = None, -np.inf
    for z_in in (2.0, 2.5, 3.0, 4.0):
        for mh in (12, 48, 144):
            s = stats(pd.concat([h4_trades(perps[c], spots[c], z_in, mh).loc[:split(perps[c])]
                                 for c in DEV_COINS]) - rt)
            if s["n"] >= 30 and s["t"] > best_t:
                best, best_t = (z_in, mh), s["t"]
    print(f"   best on DEV: z_in={best[0]} max_hold={best[1]} · round-trip cost {rt * 1e4:.1f}bps")
    for label, cs, per in (("DEV  coins·DEV ", DEV_COINS, "dev"), ("DEV  coins·TEST", DEV_COINS, "test"),
                           ("HELD coins·TEST", [c for c in COINS if c not in DEV_COINS], "test")):
        r = pd.concat([(lambda t, cut: t[t.index < cut] if per == "dev" else t[t.index >= cut])(
            h4_trades(perps[c], spots[c], *best), split(perps[c])) for c in cs])
        g, n_ = stats(r), stats(r - rt)
        print(f"   {label}  n={g['n']:5}  gross {g['bps']:+6.1f}bps (t {g['t']:+.1f})  net {n_['bps']:+6.1f}bps (t {n_['t']:+.1f})")


if __name__ == "__main__":
    main()
