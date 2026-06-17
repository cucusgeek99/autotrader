"""The lab loop — the bot setting itself up.

For each symbol it: pulls data, detects the current regime, walk-forward tunes
every strategy in the library, and records each result to the knowledge base
tagged with the regime. The best-by-OOS-score strategy becomes the champion
for that symbol+regime. Run it on a schedule and the knowledge compounds.
"""
from __future__ import annotations

from datetime import datetime, timezone

from rich.console import Console
from rich.table import Table

from autotrader.config import Config
from autotrader.data.feed import fetch_ohlcv, fetch_ohlcv_extended
from autotrader.knowledge.regime import classify
from autotrader.knowledge.store import KnowledgeStore
from autotrader.optimize.optimizer import tune
from autotrader.strategies.registry import all_strategies

console = Console()


def load_history(cfg: Config, symbol: str):
    """Pull history, preferring extended order-flow klines when available."""
    if cfg.get("microstructure", True):
        return fetch_ohlcv_extended(
            symbol, cfg["timeframe"], cfg["history_limit"],
            cfg["exchange"], cfg["data_cache_dir"],
        )
    return fetch_ohlcv(
        symbol, cfg["timeframe"], cfg["history_limit"],
        cfg["exchange"], cfg["data_cache_dir"],
    )


def run_lab(cfg: Config, symbols: list[str] | None = None) -> None:
    store = KnowledgeStore(cfg["knowledge_db"])
    symbols = symbols or cfg["symbols"]
    strategies = all_strategies()

    for symbol in symbols:
        console.rule(f"[bold]{symbol}[/]  ({cfg['timeframe']})")
        df = load_history(cfg, symbol)
        regime = classify(df)
        console.print(f"current regime: [cyan]{regime}[/]   bars: {len(df)}")

        table = Table(show_header=True, header_style="bold")
        for col in ("strategy", "OOS score", "in-sample", "overfit gap", "OOS sharpe", "params"):
            table.add_column(col)

        results = []
        for name, cls in strategies.items():
            res = tune(
                cls,
                df,
                fees=cfg["fees"],
                bars_per_year=cfg.bars_per_year,
                wf=cfg["walkforward"],
                trials=cfg["optimize"]["trials"],
                min_trades=cfg["optimize"]["min_trades"],
            )
            store.record(
                created_at=datetime.now(timezone.utc).isoformat(),
                symbol=symbol,
                timeframe=cfg["timeframe"],
                regime=regime,
                strategy=res.strategy,
                params=res.deploy_params,
                insample_score=res.insample_score,
                oos_score=res.oos_score,
                overfit_gap=res.overfit_gap,
                n_folds=res.n_folds,
                metrics=res.oos_metrics,
            )
            results.append(res)
            sharpe = res.oos_metrics.get("sharpe", 0.0)
            table.add_row(
                name,
                f"{res.oos_score:+.3f}",
                f"{res.insample_score:+.3f}",
                f"{res.overfit_gap:+.3f}",
                f"{sharpe:+.2f}",
                str(res.deploy_params),
            )

        console.print(table)
        champ = max(results, key=lambda r: r.oos_score, default=None)
        if champ:
            console.print(
                f"[bold green]champion[/] for {symbol}/{regime}: "
                f"{champ.strategy} (OOS {champ.oos_score:+.3f}) {champ.deploy_params}"
            )

    store.close()
