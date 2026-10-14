"""Reproduce the core-book evidence: BTC/ETH daily 2017→now, core vs buy&hold + param grid.

Run: .venv/bin/python scripts/backtest_core.py
"""
from __future__ import annotations

import pandas as pd

from autotrader.core import CORE_UNIVERSE, closed_daily, core_returns, summarize
from autotrader.data.feed import fetch_ohlcv


def row(name: str, r: pd.Series) -> None:
    m = summarize(r)
    print(f"{name:24s} CAGR {m['cagr']:+6.1%}  Sharpe {m['sharpe']:.2f}  maxDD {m['max_dd']:6.1%}  "
          f"worst yr {m['worst_year']:+6.1%}  x{m['multiple']:.1f}")


def main() -> None:
    closes = pd.DataFrame({s: closed_daily(fetch_ohlcv(s, "1d", 3400))["close"]
                           for s in CORE_UNIVERSE}).dropna()
    closes.index = closes.index.tz_localize(None)
    hold = closes.pct_change().fillna(0).mean(axis=1)
    for start in [None, "2022-01-01"]:
        print(f"=== from {start or closes.index[0].date()}")
        row("hold 50/50", hold[start:] if start else hold)
        c = core_returns(closes)
        row("core (vol40 × SMA200)", c[start:] if start else c)
    print("=== robustness grid (full period): sharpe / maxDD")
    for sma in (100, 150, 200, 250):
        cells = []
        for tv in (0.3, 0.4, 0.6):
            m = summarize(core_returns(closes, target_vol=tv, sma_window=sma))
            cells.append(f"vol{int(tv*100)}: {m['sharpe']:.2f}/{m['max_dd']:.0%}")
        print(f"  SMA{sma:<4} " + "   ".join(cells))


if __name__ == "__main__":
    main()
