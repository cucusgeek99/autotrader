"""Download Binance USDT-perp 1h klines + funding history since 2020 for the day-trading scan.

Output: .cache/day/{COIN}.pkl  → {"bars": DataFrame(open,high,low,close,volume), "funding": Series}
"""
from __future__ import annotations

import time
from pathlib import Path

import ccxt
import pandas as pd

COINS = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "LINK", "AVAX", "LTC", "DOT", "TRX"]
START = int(pd.Timestamp("2020-01-01").timestamp() * 1000)
OUT = Path(".cache/day")


def call(fn, params):
    for attempt in range(5):
        try:
            return fn(params)
        except Exception:
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"failed: {params}")


def bars(ex, sym: str, end: int) -> pd.DataFrame:
    rows, cur = [], START
    while cur < end:
        b = call(ex.fapiPublicGetKlines, {"symbol": sym, "interval": "1h", "startTime": cur, "limit": 1500})
        if not b:
            break
        rows += b
        cur = int(b[-1][0]) + 3_600_000
    df = pd.DataFrame([r[:6] for r in rows], columns=["t", "open", "high", "low", "close", "volume"])
    df.index = pd.to_datetime(df.t.astype("int64"), unit="ms")
    return df[~df.index.duplicated()].drop(columns="t").astype(float)


def funding(ex, sym: str, end: int) -> pd.Series:
    rows, cur = [], START
    while cur < end:
        b = call(ex.fapiPublicGetFundingRate, {"symbol": sym, "startTime": cur, "limit": 1000})
        if not b:
            break
        rows += b
        cur = int(b[-1]["fundingTime"]) + 1
        if len(b) < 1000:
            break
    s = pd.Series([float(r["fundingRate"]) for r in rows],
                  index=pd.to_datetime([int(r["fundingTime"]) for r in rows], unit="ms"), dtype=float)
    return s[~s.index.duplicated()]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ex = ccxt.binanceusdm()
    end = ex.milliseconds()
    for c in COINS:
        b, f = bars(ex, f"{c}USDT", end), funding(ex, f"{c}USDT", end)
        pd.to_pickle({"bars": b, "funding": f}, OUT / f"{c}.pkl")
        print(c, len(b), b.index[0], "→", b.index[-1], f"| funding {len(f)}", flush=True)


if __name__ == "__main__":
    main()
