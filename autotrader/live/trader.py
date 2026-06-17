"""Live paper-trading loop.

Each tick: pull the latest bars, detect the current regime, ask the knowledge
base for the champion strategy+params for that regime, compute the target
position for the most recent bar, and rebalance the paper portfolio.
The champion is re-selected every tick, so as the regime shifts the bot
swaps to whatever it learned works in the new regime.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone

from rich.console import Console

from autotrader.config import Config
from autotrader.data.feed import fetch_ohlcv_extended
from autotrader.ensemble import build_from_knowledge
from autotrader.knowledge.regime import classify
from autotrader.knowledge.store import KnowledgeStore
from autotrader.live.orderbook import book_features
from autotrader.live.paper_broker import PaperBroker
from autotrader.strategies.registry import get

console = Console()


def _select_champion(store: KnowledgeStore, symbol: str, timeframe: str, regime: str):
    champ = store.best_for(symbol, timeframe, regime)
    if champ is None:
        champ = store.best_any_regime(symbol, timeframe)
    return champ


def _build_strategy(store, symbol, tf, regime, ensemble_k):
    """Return (strategy, label) for the current regime, or (None, None)."""
    if ensemble_k and ensemble_k > 1:
        ens = build_from_knowledge(store, symbol, tf, regime, k=ensemble_k)
        if ens is not None:
            return ens, f"ensemble[{len(ens.members)}]"
    champ = _select_champion(store, symbol, tf, regime)
    if champ is None:
        return None, None
    return get(champ.strategy)(**champ.params), champ.strategy


def run_live(cfg: Config, symbol: str, once: bool = False, ensemble_k: int = 0) -> None:
    store = KnowledgeStore(cfg["knowledge_db"])
    fees = cfg["fees"]
    broker = PaperBroker.load_or_create(
        symbol,
        cash=float(cfg["paper"]["starting_cash"]),
        fee=fees["taker"],
        slippage=fees["slippage"],
    )
    poll = int(cfg["paper"]["poll_seconds"])
    tf = cfg["timeframe"]
    gate_on = bool(cfg.get("orderbook_gate", True))
    depth = int(cfg.get("orderbook_depth", 20))
    import ccxt
    ex = getattr(ccxt, cfg["exchange"])({"enableRateLimit": True})

    console.print(f"[bold green]Paper trading[/] {symbol} @ {tf} — Ctrl-C to stop")
    while True:
        try:
            # extended klines give order-flow strategies their columns live too
            df = fetch_ohlcv_extended(symbol, tf, limit=400,
                                      exchange_id=cfg["exchange"], use_cache=False)
            price = float(df["close"].iloc[-1])
            now = datetime.now(timezone.utc).isoformat()
            regime = classify(df)
            strat, label = _build_strategy(store, symbol, tf, regime, ensemble_k)

            if strat is None:
                console.print(
                    f"[yellow]No champion yet for {symbol}. Run `autotrader lab` first.[/]"
                )
            else:
                target = float(strat.generate_signals(df).iloc[-1])
                note = ""
                if gate_on and target != 0.0:
                    bf = book_features(ex, symbol, depth)
                    if not bf.confirms(target):
                        # book disagrees — hold current exposure instead of entering
                        target = broker.current_fraction(price)
                        note = f"  [magenta]book-veto imb={bf.imbalance:+.2f}[/]"
                fill = broker.rebalance_to(target, price, now)
                broker.save()
                eq = broker.equity(price)
                line = (
                    f"{now[:19]} | {regime:16} | {label:18} "
                    f"target={target:+.2f} price={price:,.2f} "
                    f"equity=${eq:,.2f}"
                )
                if fill:
                    line += f"  [cyan]{fill.side.upper()} {abs(fill.units):.6f}[/]"
                console.print(line + note)
        except Exception as exc:  # keep the loop alive on transient errors
            console.print(f"[red]tick error:[/] {exc}")

        if once:
            break
        time.sleep(poll)

    store.close()
