"""Config tournament for the core book — which setup is best, and does picking it pay?

Searches a grid of sensible setups (asset mix × trend filter × vol target) on daily
BTC/ETH, then judges the winner the honest way:
  1. full-period leaderboard (in-sample — optimistic by construction)
  2. walk-forward: each year, pick the best config on PAST data only, trade it next
     year; compare the stitched out-of-sample result to the fixed core and to holding
  3. deflated Sharpe of (winner − fixed core): is the winner's edge over the core
     more than what the best of N random tries would show?
  4. held-out coins: do the top configs keep their ranking on coins never searched?

Run: .venv/bin/python scripts/config_tournament.py [--metric sharpe|calmar]
"""
from __future__ import annotations

import argparse
import itertools

import numpy as np
import pandas as pd

from autotrader.analysis.cross_validation import deflated_sharpe_ratio
from autotrader.core import FEE, closed_daily, summarize
from autotrader.data.feed import fetch_ohlcv

TRENDS = ["none", "sma50", "sma100", "sma150", "sma200", "golden", "half200", "buffer150"]
VOLS = [None, 0.30, 0.40, 0.60]
ASSETS = ["BTC", "5050"]
CORE = ("5050", "sma200", 0.40)
HELD_OUT = ["SOL", "BNB", "XRP", "ADA", "DOGE", "AVAX", "LINK", "LTC", "DOT", "TRX"]


def load(coin: str) -> pd.Series:
    s = closed_daily(fetch_ohlcv(f"{coin}/USDT", "1d", 3400))["close"]
    s.index = s.index.tz_localize(None)
    return s


def trend_mask(c: pd.Series, kind: str) -> pd.Series:
    if kind == "none":
        return pd.Series(1.0, c.index)
    if kind.startswith("sma"):
        return (c > c.rolling(int(kind[3:])).mean()).astype(float)
    if kind == "golden":
        return (c.rolling(50).mean() > c.rolling(200).mean()).astype(float)
    if kind == "half200":   # keep half the position below SMA200 instead of going to cash
        return 0.5 + 0.5 * (c > c.rolling(200).mean()).astype(float)
    if kind == "buffer150":  # enter 2% above SMA150, exit 2% below → fewer whipsaws
        m = c.rolling(150).mean()
        state, out = 0.0, []
        for px, mv in zip(c, m):
            if not np.isnan(mv):
                state = 1.0 if px > mv * 1.02 else (0.0 if px < mv * 0.98 else state)
            out.append(state)
        return pd.Series(out, c.index)
    raise ValueError(kind)


def weights(c: pd.Series, trend: str, vol: float | None) -> pd.Series:
    w = trend_mask(c, trend)
    if vol is not None:
        rv = c.pct_change().rolling(30).std() * np.sqrt(365)
        w = w * (vol / rv).clip(upper=1.0)
    warm = c.rolling(200).mean().isna()   # same warm-up for every config → fair comparison
    return w.where(~warm, 0.0).fillna(0.0)


def returns(prices: dict[str, pd.Series], cfg) -> pd.Series:
    assets, trend, vol = cfg
    coins = ["BTC"] if assets == "BTC" else ["BTC", "ETH"]
    df = pd.DataFrame({k: prices[k] for k in coins}).dropna()
    W = pd.DataFrame({k: weights(df[k], trend, vol) for k in coins}) / len(coins)
    W = W.shift(1).fillna(0.0)
    r = (W * df.pct_change().fillna(0.0)).sum(axis=1) - W.diff().abs().fillna(0.0).sum(axis=1) * FEE
    return r


def score(r: pd.Series, metric: str) -> float:
    m = summarize(r)
    return m["sharpe"] if metric == "sharpe" else m["cagr"] / abs(m["max_dd"] or 1e-9)


def name(cfg) -> str:
    a, t, v = cfg
    return f"{a:4} {t:9} vol{'-' if v is None else int(v * 100)}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="sharpe", choices=["sharpe", "calmar"])
    metric = ap.parse_args().metric
    prices = {k: load(k) for k in ("BTC", "ETH")}
    grid = list(itertools.product(ASSETS, TRENDS, VOLS))
    R = pd.DataFrame({cfg: returns(prices, cfg) for cfg in grid}).dropna()
    R = R[R.index >= R.index[0] + pd.Timedelta(days=200)]   # skip shared warm-up
    hold = R[("5050", "none", None)]
    print(f"{len(grid)} configs · {R.index[0].date()} → {R.index[-1].date()} · metric={metric}\n")

    # 1) in-sample leaderboard
    rows = []
    for cfg in grid:
        m = summarize(R[cfg])
        rows.append({"config": name(cfg), "cagr": m["cagr"], "sharpe": m["sharpe"], "max_dd": m["max_dd"],
                     "calmar": m["cagr"] / abs(m["max_dd"]), "worst_yr": m["worst_year"]})
    lb = pd.DataFrame(rows).sort_values(metric, ascending=False)
    fmt = {"cagr": "{:+.1%}".format, "sharpe": "{:.2f}".format, "max_dd": "{:.0%}".format,
           "calmar": "{:.2f}".format, "worst_yr": "{:+.0%}".format}
    print("1) IN-SAMPLE leaderboard (top 12 + core + hold) — optimistic by construction")
    show = pd.concat([lb.head(12), lb[lb.config.isin([name(CORE), name(("5050", "none", None))])]]).drop_duplicates()
    print(show.to_string(index=False, formatters=fmt), "\n")

    # 2) walk-forward: pick on the past (expanding), trade the next calendar year
    years = sorted({d.year for d in R.index})[2:]   # need ≥2 years of history to pick
    oos, picks = [], []
    for y in years:
        past = R[R.index < f"{y}-01-01"]
        best = max(grid, key=lambda c: score(past[c], metric))
        picks.append((y, name(best)))
        oos.append(R.loc[str(y), best])
    oos = pd.concat(oos)
    span = oos.index
    print("2) WALK-FORWARD — each year uses the config that was best on all PRIOR years")
    for y, p in picks:
        print(f"   {y}: {p}")
    for label, r in [("picked-best (OOS)", oos), ("fixed core", R[CORE].loc[span]), ("hold 50/50", hold.loc[span])]:
        m = summarize(r)
        print(f"   {label:18} CAGR {m['cagr']:+6.1%}  Sharpe {m['sharpe']:.2f}  maxDD {m['max_dd']:6.1%}  "
              f"worst yr {m['worst_year']:+5.0%}  ×{m['multiple']:.1f}")
    print()

    # 3) is the in-sample winner's edge over the core more than selection luck?
    winner = max(grid, key=lambda c: score(R[c], metric))
    diff = (R[winner] - R[CORE])
    trial_sr = [float((R[c] - R[CORE]).mean() / (R[c] - R[CORE]).std()) for c in grid if c != CORE]
    d = deflated_sharpe_ratio(diff, n_trials=len(grid), trial_sharpes=trial_sr)
    print(f"3) DEFLATED: winner [{name(winner)}] minus core → DSR {d['dsr']:.2f} "
          f"(≥0.95 = real improvement; below = could be luck of the best of {len(grid)})\n")

    # 4) held-out coins: rank the top-5 + core + hold on coins never searched (single-coin version)
    top = [c for c in sorted(grid, key=lambda c: -score(R[c], metric)) if c[0] == "BTC"][:5]
    cands = list(dict.fromkeys(top + [("BTC", "sma200", 0.40), ("BTC", "none", None)]))
    print("4) HELD-OUT coins (same rules, single coin) — median over 10 unseen coins")
    hr = {name(c): [] for c in cands}
    for coin in HELD_OUT:
        p = {"BTC": load(coin)}   # reuse the BTC-only code path on another coin
        for c in cands:
            r = returns(p, c)
            r = r[r.index >= r.index[0] + pd.Timedelta(days=200)]
            m = summarize(r)
            hr[name(c)].append((m["sharpe"], m["max_dd"], m["cagr"]))
    for k, v in hr.items():
        a = np.array(v)
        print(f"   {k.replace('BTC ', 'coin'):22} Sharpe {np.median(a[:, 0]):.2f}  maxDD {np.median(a[:, 1]):.0%}  "
              f"CAGR {np.median(a[:, 2]):+.0%}")


if __name__ == "__main__":
    main()
