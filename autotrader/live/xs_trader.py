"""Live cross-sectional paper trader for the house strategy (findings_xs).

Each tick: pull recent bars for the whole universe, compute the dollar-neutral
target weights from the FindingsStrategy, and rebalance the multi-asset paper
book. Market-neutral, low-turnover. Paper only.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone

from rich.console import Console
from rich.table import Table

from autotrader.analysis import cross_sectional as xs
from autotrader.config import Config
from autotrader.data.feed import fetch_ohlcv_extended
from autotrader.live.multi_broker import MultiAssetPaperBroker
from autotrader.portfolio import DEFAULT_FACTORS, EXPANDED_FACTORS, FindingsStrategy
from autotrader.risk import RiskLimits, RiskManager

console = Console()

DEFAULT_UNIVERSE = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", "ADA/USDT",
    "DOGE/USDT", "AVAX/USDT", "LINK/USDT", "LTC/USDT", "DOT/USDT", "TRX/USDT",
]


def run_live_xs(cfg: Config, universe=None, once=False, expanded=True) -> None:
    universe = universe or DEFAULT_UNIVERSE
    fees, bpy, tf = cfg["fees"], cfg.bars_per_year, cfg["timeframe"]
    factors = EXPANDED_FACTORS if expanded else DEFAULT_FACTORS
    strat = FindingsStrategy(factors=list(factors))
    broker = MultiAssetPaperBroker.load_or_create(
        cash=float(cfg["paper"]["starting_cash"]),
        fee=fees["taker"], slippage=fees["slippage"],
    )
    risk = RiskManager(RiskLimits(**cfg.get("risk", {})))
    poll = int(cfg["paper"]["poll_seconds"])
    # need enough history for the slowest factor (longrev 720) + buffer
    bars = 1500

    console.print(f"[bold green]Cross-sectional paper book[/] — {len(universe)} coins @ {tf}, "
                  f"market-neutral. Ctrl-C to stop.")
    while True:
        try:
            dfs = {}
            for s in universe:
                try:
                    dfs[s] = fetch_ohlcv_extended(s, tf, bars, cfg["exchange"],
                                                  cfg["data_cache_dir"], use_cache=False)
                except Exception:
                    pass
            closes = xs.build_panels(dfs)["closes"]
            prices = {s: float(closes[s].iloc[-1]) for s in closes.columns}
            now = datetime.now(timezone.utc).isoformat()

            eq = broker.equity(prices)
            risk.update_equity(eq, day_key=now[:10])
            target = strat.target_weights(closes, fees, bpy)
            latest = target.iloc[-1].to_dict()
            # RISK GATE: clamp/kill-switch before anything reaches the book
            gated, notes = risk.check_weights(latest)
            if notes:
                console.print(f"[yellow]risk:[/] {'; '.join(notes)}")
            fills = broker.rebalance_to(gated, prices, now)
            broker.save()

            eq = broker.equity(prices)
            book = broker.book(prices)
            if risk.halted:
                console.print(f"[bold red]TRADING HALTED — {risk.state.halt_reason}. "
                              "Run `autotrader risk --resume` after review.[/]")
            longs = [b for b in book if b["weight"] > 0][:3]
            shorts = [b for b in book if b["weight"] < 0][-3:]
            console.print(
                f"{now[:19]} | equity ${eq:,.2f} | {len(fills)} trades | "
                f"long {', '.join(b['symbol'].split('/')[0] for b in longs)} / "
                f"short {', '.join(b['symbol'].split('/')[0] for b in shorts)}"
            )
        except Exception as exc:
            console.print(f"[red]tick error:[/] {exc}")
        if once:
            break
        time.sleep(poll)
