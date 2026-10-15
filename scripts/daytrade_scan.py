"""Day-trading scan — the classic intraday playbooks, flat at the end of every UTC day.

Pre-registered protocol (fixed BEFORE looking at results):
  * data: Binance USDT perps, 1h bars + real funding, 2020-01 → now, 12 coins
  * DEV = before 2024-07-01, TEST = after. DEV coins BTC/ETH, the other 10 are held-out.
  * each mechanism's params are picked on DEV coins · DEV period only (best per-trade t, n ≥ 50)
  * costs: taker 0.05%/side + slippage (1 bp/side BTC/ETH, 3 bps alts) + real funding
  * fills: breakouts fill at the trigger, or at the bar open if it gapped through;
    stop and entry inside the same bar → assume the stop hit (worst case)
  * PASS = TEST, all 12 coins pooled: mean net > 0 with DAY-CLUSTERED t > 2.6 (Bonferroni
    over 5 mechanisms) AND held-out coins on TEST also mean net > 0
    (correction made after the first run: the original per-trade t treated 12 coins
    trading the same day as independent bets — it showed D2 at t 4.25 when the
    day-clustered t is 1.60. The correction only makes the test stricter.)

Mechanisms:
  D1 opening-range breakout — range of the first R hours, trade the first break, stop
  D2 intraday momentum      — return 00:00→T predicts T→24:00 (with or against)
  D3 day-after big move     — |yesterday| > k·σ → trade today open→close (with or against)
  D4 session drift          — hold one session (Asia/EU/US/late/all-day) long or short
  D5 volatility breakout    — Larry Williams / "Korean VBO": open ± k·yesterday's range

Run: .venv/bin/python scripts/fetch_1h.py && .venv/bin/python scripts/daytrade_scan.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

COINS = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "LINK", "AVAX", "LTC", "DOT", "TRX"]
DEV_COINS = ["BTC", "ETH"]
CUT = pd.Timestamp("2024-07-01")
TAKER = 0.0005
SLIP = {"BTC": 0.0001, "ETH": 0.0001}   # others 3 bps
T_PASS = 2.6
H = pd.Timedelta(hours=1)


class Panel:
    """One coin as a (days × 24 hours) grid + funding lookup."""

    def __init__(self, coin: str):
        d = pd.read_pickle(Path(".cache/day") / f"{coin}.pkl")
        b = d["bars"]
        grid = pd.date_range(b.index[0].normalize() + pd.Timedelta(days=1), b.index[-1].normalize(),
                             freq="1h", inclusive="left")
        b = b.reindex(grid)
        days = grid[::24]
        shape = (len(days), 24)
        self.O, self.Hi, self.L, self.C = (b[k].values.reshape(shape) for k in ("open", "high", "low", "close"))
        ok = ~np.isnan(self.C).any(1) & ~np.isnan(self.O).any(1)
        self.days, self.ok = days, ok
        self.cost = 2 * (TAKER + SLIP.get(coin, 0.0003))
        f = d["funding"].sort_index()
        self.ft, self.fc = f.index.values, np.concatenate([[0.0], f.cumsum().values])
        dr = self.C[:, 23] / self.O[:, 0] - 1                    # full-day return
        self.ret = dr
        self.sig = pd.Series(dr).rolling(30, min_periods=20).std().shift(1).values   # yesterday-known
        self.rng = np.concatenate([[np.nan], (self.Hi.max(1) - self.L.min(1))[:-1]])  # yesterday's range
        self.ma5 = pd.Series(self.C[:, 23]).rolling(5).mean().shift(1).values

    def fund(self, t_in: pd.Timestamp, t_out: pd.Timestamp) -> float:
        """Funding paid by a long between entry (exclusive) and exit hour (inclusive)."""
        i = np.searchsorted(self.ft, np.datetime64(t_in), side="right")
        j = np.searchsorted(self.ft, np.datetime64(t_out), side="right")
        return self.fc[j] - self.fc[i]

    def trade(self, d: int, side: int, h_in: int, px_in: float, h_out: int, px_out: float) -> tuple:
        day = self.days[d]
        f = self.fund(day + h_in * H, day + h_out * H)
        net = side * (px_out / px_in - 1) - self.cost - side * f
        return day, net


def _walk(p: Panel, d: int, side: int, h: int, px: float, stop: float | None) -> tuple:
    """From entry bar h at px, hold to the close of hour 23 unless the stop is hit."""
    if stop is not None:
        for j in range(h, 24):
            hit = p.L[d, j] <= stop if side > 0 else p.Hi[d, j] >= stop
            if hit:
                fill = stop if j == h else (min(p.O[d, j], stop) if side > 0 else max(p.O[d, j], stop))
                return p.trade(d, side, h, px, j, fill)
    return p.trade(d, side, h, px, 23, p.C[d, 23])


def _breakout(p: Panel, d: int, start: int, up: float, dn: float, long_only: bool, stop_of) -> tuple | None:
    for h in range(start, 24):
        bu, bd = p.Hi[d, h] > up, (p.L[d, h] < dn) and not long_only
        if bu and bd:
            return None                                        # ambiguous bar → skip the day
        if bu:
            return _walk(p, d, 1, h, max(p.O[d, h], up), stop_of(1))
        if bd:
            return _walk(p, d, -1, h, min(p.O[d, h], dn), stop_of(-1))
    return None


# ---------------- the five mechanisms: each returns a list of (day, net) ----------------
def d1_orb(p: Panel, R: int, stop: str) -> list:
    out = []
    for d in np.flatnonzero(p.ok):
        hi, lo = p.Hi[d, :R].max(), p.L[d, :R].min()
        st = {"none": lambda s: None, "opp": lambda s: lo if s > 0 else hi,
              "mid": lambda s: (hi + lo) / 2}[stop]
        t = _breakout(p, d, R, hi, lo, False, st)
        if t:
            out.append(t)
    return out


def d2_momentum(p: Panel, T: int, k: float, sgn: int) -> list:
    out = []
    for d in np.flatnonzero(p.ok & ~np.isnan(p.sig)):
        r = p.C[d, T - 1] / p.O[d, 0] - 1
        if abs(r) <= k * p.sig[d] or r == 0:
            continue
        side = sgn * int(np.sign(r))
        out.append(p.trade(d, side, T, p.O[d, T], 23, p.C[d, 23]))
    return out


def d3_dayafter(p: Panel, k: float, sgn: int) -> list:
    out = []
    for d in np.flatnonzero(p.ok & ~np.isnan(p.sig)):
        if d == 0 or not p.ok[d - 1]:
            continue
        z = p.ret[d - 1] / p.sig[d]
        if abs(z) <= k:
            continue
        out.append(p.trade(d, sgn * int(np.sign(z)), 0, p.O[d, 0], 23, p.C[d, 23]))
    return out


def d4_session(p: Panel, a: int, b: int, side: int) -> list:
    return [p.trade(d, side, a, p.O[d, a], b - 1, p.C[d, b - 1]) for d in np.flatnonzero(p.ok)]


def d5_vbo(p: Panel, k: float, long_only: bool, trend: bool) -> list:
    out = []
    for d in np.flatnonzero(p.ok & ~np.isnan(p.rng) & ~np.isnan(p.ma5)):
        o = p.O[d, 0]
        if trend and long_only and o < p.ma5[d]:
            continue
        t = _breakout(p, d, 0, o + k * p.rng[d], o - k * p.rng[d], long_only, lambda s: None)
        if t:
            out.append(t)
    return out


MECHS = {
    "D1 opening-range breakout": (d1_orb, [{"R": r, "stop": s} for r in (1, 2, 4, 8) for s in ("none", "opp", "mid")]),
    "D2 intraday momentum": (d2_momentum, [{"T": t, "k": k, "sgn": s} for t in (8, 12, 16, 20, 22)
                                           for k in (0.0, 0.5, 1.0) for s in (1, -1)]),
    "D3 day-after big move": (d3_dayafter, [{"k": k, "sgn": s} for k in (1.0, 1.5, 2.0, 2.5) for s in (1, -1)]),
    "D4 session drift": (d4_session, [{"a": a, "b": b, "side": s} for a, b in ((0, 8), (8, 13), (13, 21), (21, 24), (0, 24))
                                      for s in (1, -1)]),
    "D5 volatility breakout": (d5_vbo, [{"k": k, "long_only": lo, "trend": tr} for k in (0.3, 0.5, 0.7, 1.0)
                                        for lo in (False, True) for tr in (False, True) if not (tr and not lo)]),
}


def stats(x) -> dict:
    x = np.asarray(x, dtype=float)
    if len(x) < 5:
        return {"n": len(x), "bps": np.nan, "t": np.nan}
    return {"n": len(x), "bps": x.mean() * 1e4, "t": x.mean() / x.std() * np.sqrt(len(x))}


def day_t(trades: list) -> float:
    """t-stat over days of the mean net across coins traded that day (same-day trades = one bet)."""
    m = pd.Series([n for _, n in trades], index=[d for d, _ in trades], dtype=float).groupby(level=0).mean()
    return m.mean() / m.std() * np.sqrt(len(m)) if len(m) > 4 else np.nan


def fmt(x) -> str:
    s = stats(x)
    return f"n={s['n']:5}  net {s['bps']:+6.1f}bps/trade  t {s['t']:+5.2f}"


def book_sharpe(trades: dict, coins: list, lo, hi) -> str:
    """Equal-capital book across `coins`: daily return = mean over coins (0 when flat)."""
    frames = []
    for c in coins:
        s = pd.Series({d: r for d, r in trades[c]}, dtype=float)
        frames.append(s[(s.index >= lo) & (s.index < hi)])
    df = pd.concat(frames, axis=1, sort=True).fillna(0.0)
    if df.empty:
        return "—"
    idx = pd.date_range(lo, min(hi, df.index.max()), freq="D")
    r = df.reindex(idx).fillna(0.0).mean(axis=1)
    sh = r.mean() / r.std() * np.sqrt(365) if r.std() > 0 else np.nan
    cagr = (1 + r).prod() ** (365 / len(r)) - 1
    dd = ((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1).min()
    return f"book Sharpe {sh:+.2f}  CAGR {cagr:+6.1%}  DD {dd:+6.1%}"


def main() -> None:
    panels = {c: Panel(c) for c in COINS if (Path(".cache/day") / f"{c}.pkl").exists()}
    coins = list(panels)
    held = [c for c in coins if c not in DEV_COINS]
    first = min(p.days[0] for p in panels.values())
    last = max(p.days[-1] for p in panels.values()) + pd.Timedelta(days=1)
    print(f"{len(coins)} coins · {first.date()} → {last.date()} · DEV < {CUT.date()} ≤ TEST")
    print(f"costs: taker {TAKER:.2%}/side + slippage, round-trip ≈ {2 * (TAKER + 0.0001) * 1e4:.0f} bps "
          f"(BTC/ETH) / {2 * (TAKER + 0.0003) * 1e4:.0f} bps (alts) + funding\n")

    verdicts = []
    for name, (fn, grid) in MECHS.items():
        best, best_t = None, -np.inf
        for prm in grid:
            r = [n for c in DEV_COINS for d, n in fn(panels[c], **prm) if d < CUT]
            s = stats(r)
            if s["n"] >= 50 and s["t"] > best_t:
                best, best_t = prm, s["t"]
        tr = {c: fn(panels[c], **best) for c in coins}
        pick = lambda cs, test: [n for c in cs for d, n in tr[c] if (d >= CUT) == test]
        pick_dn = lambda cs, test: [(d, n) for c in cs for d, n in tr[c] if (d >= CUT) == test]
        print(f"{name}\n   best on DEV: {best}  (from {len(grid)} configs)")
        print(f"   DEV  coins · DEV   {fmt(pick(DEV_COINS, False))}")
        print(f"   HELD coins · DEV   {fmt(pick(held, False))}")
        print(f"   DEV  coins · TEST  {fmt(pick(DEV_COINS, True))}")
        print(f"   HELD coins · TEST  {fmt(pick(held, True))}")
        all_t = stats(pick(coins, True))
        tc = day_t(pick_dn(coins, True))
        print(f"   ALL  coins · TEST  {fmt(pick(coins, True))}  day-clustered t {tc:+5.2f}   ← verdict")
        print(f"   TEST {book_sharpe(tr, coins, CUT, last)}")
        ok = all_t["bps"] > 0 and tc > T_PASS and stats(pick(held, True))["bps"] > 0
        verdicts.append((name, ok))
        print(f"   → {'PASS' if ok else 'FAIL'}\n")

    print("VERDICTS: " + " · ".join(f"{n.split()[0]} {'PASS' if ok else 'fail'}" for n, ok in verdicts))


if __name__ == "__main__":
    main()
