"""Perpetual-futures data: funding-rate history + perp klines.

Funding fade is a *perp* signal — it doesn't exist on spot. We fetch both the
funding history and the price series from the same USDⓈ-M futures market
(binanceusdm) so they're consistent, then forward-fill the realized funding onto
each bar (causal: at bar t you know the most recent settled funding).
"""
from __future__ import annotations

import time

import pandas as pd

from autotrader.data.feed import _exchange, fetch_ohlcv_extended


def fetch_funding_history(
    symbol: str,
    exchange_id: str = "binanceusdm",
    limit: int = 3000,
    cache_dir: str = ".cache",
    use_cache: bool = True,
) -> pd.DataFrame:
    """Realized funding rate per settlement (~every 8h on Binance), datetime-indexed."""
    from pathlib import Path

    safe = symbol.replace("/", "-").replace(":", "_")
    path = Path(cache_dir) / f"{exchange_id}_funding_{safe}.csv"
    if use_cache and path.exists():
        cached = pd.read_csv(path, parse_dates=["datetime"]).set_index("datetime")
        if len(cached) >= limit:
            return cached.tail(limit)

    ex = _exchange(exchange_id)
    ex.load_markets()
    # ~3 settlements/day → pull enough calendar span to cover `limit` settlements
    span_ms = int(limit / 3 * 86_400_000)
    since = ex.milliseconds() - span_ms
    rows = []
    cursor = since
    while True:
        batch = ex.fetch_funding_rate_history(symbol, since=cursor, limit=1000)
        if not batch:
            break
        rows.extend(batch)
        cursor = batch[-1]["timestamp"] + 1
        if len(batch) < 1000:
            break
        time.sleep(ex.rateLimit / 1000)

    if not rows:
        return pd.DataFrame(columns=["funding"])
    df = pd.DataFrame({
        "datetime": pd.to_datetime([r["timestamp"] for r in rows], unit="ms", utc=True),
        "funding": [float(r["fundingRate"]) for r in rows],
    }).set_index("datetime").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    if use_cache:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        df.to_csv(path)
    return df.tail(limit)


def fetch_perp_with_funding(
    symbol: str,
    timeframe: str = "1h",
    limit: int = 6000,
    exchange_id: str = "binanceusdm",
    cache_dir: str = ".cache",
    use_cache: bool = True,
) -> pd.DataFrame:
    """Perp OHLCV (+ order-flow) with a forward-filled `funding` column."""
    px = fetch_ohlcv_extended(symbol, timeframe, limit, exchange_id, cache_dir, use_cache)
    fund = fetch_funding_history(symbol, exchange_id, limit=max(1000, limit // 4),
                                 cache_dir=cache_dir, use_cache=use_cache)
    if fund.empty:
        px["funding"] = 0.0
        return px
    # coerce indexes to UTC datetime (cache round-trips can stringify them) and
    # align realized funding onto each bar causally via a backward as-of join
    px = px.copy()
    px.index = pd.to_datetime(px.index, utc=True, format="ISO8601")
    fund = fund.copy()
    fund.index = pd.to_datetime(fund.index, utc=True, format="ISO8601")
    merged = pd.merge_asof(
        px.sort_index(), fund.sort_index(),
        left_index=True, right_index=True, direction="backward",
    )
    merged["funding"] = merged["funding"].fillna(0.0)
    return merged
