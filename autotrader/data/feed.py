"""Market data: paginated OHLCV fetch via ccxt, cached to disk.

Public data only — no API keys required for OHLCV. The cache makes
repeated backtests/optimization runs fast and offline-friendly.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def _exchange(exchange_id: str):
    import ccxt

    klass = getattr(ccxt, exchange_id)
    return klass({"enableRateLimit": True})


def _cache_path(cache_dir: str, exchange_id: str, symbol: str, timeframe: str) -> Path:
    safe = symbol.replace("/", "-")
    d = Path(cache_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{exchange_id}_{safe}_{timeframe}.csv"


def fetch_ohlcv(
    symbol: str,
    timeframe: str = "1h",
    limit: int = 6000,
    exchange_id: str = "binance",
    cache_dir: str = ".cache",
    use_cache: bool = True,
) -> pd.DataFrame:
    """Return a DataFrame indexed by UTC datetime with OHLCV columns.

    Pulls `limit` most-recent bars, paginating in chunks of 1000.
    Results are cached; a second call refreshes only the tail.
    """
    path = _cache_path(cache_dir, exchange_id, symbol, timeframe)
    cached = pd.DataFrame()
    if use_cache and path.exists():
        cached = pd.read_csv(path, parse_dates=["datetime"]).set_index("datetime")

    ex = _exchange(exchange_id)
    tf_ms = ex.parse_timeframe(timeframe) * 1000
    now_ms = ex.milliseconds()
    since = now_ms - limit * tf_ms

    rows: list[list] = []
    cursor = since
    while cursor < now_ms:
        batch = ex.fetch_ohlcv(symbol, timeframe, since=cursor, limit=1000)
        if not batch:
            break
        rows.extend(batch)
        cursor = batch[-1][0] + tf_ms
        if len(batch) < 1000:
            break
        time.sleep(ex.rateLimit / 1000)

    df = pd.DataFrame(rows, columns=_COLUMNS)
    if df.empty and not cached.empty:
        return cached.tail(limit)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("datetime").drop(columns=["timestamp"])
    df = df[~df.index.duplicated(keep="last")].sort_index()

    if not cached.empty:
        df = pd.concat([cached, df])
        df = df[~df.index.duplicated(keep="last")].sort_index()

    if use_cache:
        df.to_csv(path)
    return df.tail(limit)


def latest_ohlcv(
    symbol: str,
    timeframe: str = "1h",
    bars: int = 500,
    exchange_id: str = "binance",
) -> pd.DataFrame:
    """Fetch the most recent `bars` without touching the cache — for live polling."""
    ex = _exchange(exchange_id)
    raw = ex.fetch_ohlcv(symbol, timeframe, limit=bars)
    df = pd.DataFrame(raw, columns=_COLUMNS)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df.set_index("datetime").drop(columns=["timestamp"]).sort_index()


# Binance kline = 12 fields, vs the 6 that fetch_ohlcv exposes. Fields 7-9 carry
# the order-flow data (quote volume, trade count, taker-buy volume) that lets us
# reconstruct aggressor imbalance from purely historical bars — no look-ahead.
_EXT_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "num_trades", "taker_buy_base", "taker_buy_quote", "ignore",
]


def fetch_ohlcv_extended(
    symbol: str,
    timeframe: str = "1h",
    limit: int = 6000,
    exchange_id: str = "binance",
    cache_dir: str = ".cache",
    use_cache: bool = True,
) -> pd.DataFrame:
    """OHLCV plus order-flow columns (taker_buy_base, quote_volume, num_trades).

    Only Binance(-style) exchanges expose these via the raw klines endpoint. For
    others, falls back to plain OHLCV with the extra columns absent — strategies
    that need them degrade to flat.
    """
    path = _cache_path(cache_dir, exchange_id + "_ext", symbol, timeframe)
    if use_cache and path.exists():
        cached = pd.read_csv(path, parse_dates=["datetime"]).set_index("datetime")
        # refresh only if the cache is short of what's requested
        if len(cached) >= limit:
            return cached.tail(limit)

    ex = _exchange(exchange_id)
    # locate the raw klines implicit endpoint (spot vs usdm futures)
    getter = getattr(ex, "publicGetKlines", None) or getattr(ex, "fapiPublicGetKlines", None)
    if getter is None:
        return fetch_ohlcv(symbol, timeframe, limit, exchange_id, cache_dir, use_cache)

    ex.load_markets()
    market_id = ex.market(symbol)["id"]
    tf_ms = ex.parse_timeframe(timeframe) * 1000
    now_ms = ex.milliseconds()
    cursor = now_ms - limit * tf_ms
    rows: list[list] = []
    while cursor < now_ms:
        batch = getter({
            "symbol": market_id, "interval": timeframe,
            "startTime": int(cursor), "limit": 1000,
        })
        if not batch:
            break
        rows.extend(batch)
        cursor = int(batch[-1][0]) + tf_ms
        if len(batch) < 1000:
            break
        time.sleep(ex.rateLimit / 1000)

    df = pd.DataFrame(rows, columns=_EXT_COLUMNS)
    for col in ("open_time", "open", "high", "low", "close", "volume",
                "quote_volume", "num_trades", "taker_buy_base"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["datetime"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = (
        df.set_index("datetime")[
            ["open", "high", "low", "close", "volume",
             "quote_volume", "num_trades", "taker_buy_base"]
        ]
        .sort_index()
    )
    df = df[~df.index.duplicated(keep="last")]
    if use_cache:
        df.to_csv(path)
    return df.tail(limit)
