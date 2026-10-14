"""Download ~1 year of Binance 5m klines (perp + spot, with taker-buy volume) for the scalping scan.

Output: .cache/scalp/{perp|spot}_{COIN}.pkl
"""
from __future__ import annotations

import sys
import time

import ccxt
import pandas as pd

COINS = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "LINK"]
COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "num_trades", "taker_buy_base", "taker_buy_quote", "ignore"]


def pull(getter, symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    rows, cur = [], start_ms
    while cur < end_ms:
        for attempt in range(5):
            try:
                b = getter({"symbol": symbol, "interval": "5m", "startTime": cur, "limit": 1000})
                break
            except Exception:
                time.sleep(2 * (attempt + 1))
        else:
            raise RuntimeError(f"{symbol}: failed at {cur}")
        if not b:
            break
        rows += b
        cur = int(b[-1][0]) + 300_000
    df = pd.DataFrame(rows, columns=COLS)
    df.index = pd.to_datetime(df.open_time.astype("int64"), unit="ms")
    df = df[~df.index.duplicated()].drop(columns=["open_time", "close_time", "ignore", "taker_buy_quote"])
    return df.astype(float)


def main() -> None:
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 365
    spot, perp = ccxt.binance(), ccxt.binanceusdm()
    end = spot.milliseconds()
    start = end - days * 86_400_000
    for c in COINS:
        for kind, getter in (("perp", perp.fapiPublicGetKlines), ("spot", spot.publicGetKlines)):
            df = pull(getter, f"{c}USDT", start, end)
            df.to_pickle(f".cache/scalp/{kind}_{c}.pkl")
            print(kind, c, len(df), df.index[0], "→", df.index[-1], flush=True)


if __name__ == "__main__":
    main()
