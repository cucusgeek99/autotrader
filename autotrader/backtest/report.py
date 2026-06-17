"""Human-readable backtest reporting: metrics table, ASCII equity sparkline,
and optional CSV export of the equity curve. Zero plotting dependencies."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

from autotrader.backtest.engine import BacktestResult

_BLOCKS = "▁▂▃▄▅▆▇█"


def sparkline(series: pd.Series, width: int = 60) -> str:
    """Downsample a series to `width` points and render as block characters."""
    s = series.dropna()
    if len(s) == 0:
        return ""
    if len(s) > width:
        # bucket-average down to width points
        idx = pd.cut(range(len(s)), bins=width, labels=False)
        s = s.groupby(idx).mean()
    lo, hi = float(s.min()), float(s.max())
    rng = hi - lo or 1.0
    return "".join(_BLOCKS[min(len(_BLOCKS) - 1, int((v - lo) / rng * (len(_BLOCKS) - 1)))] for v in s)


def render(result: BacktestResult, title: str, console: Console | None = None) -> None:
    console = console or Console()
    m = result.metrics
    table = Table(title=title, header_style="bold", show_header=False)
    table.add_column("metric")
    table.add_column("value", justify="right")
    table.add_row("total return", f"{m.total_return:+.2%}")
    table.add_row("CAGR", f"{m.cagr:+.2%}")
    table.add_row("Sharpe", f"{m.sharpe:+.2f}")
    table.add_row("Sortino", f"{m.sortino:+.2f}")
    table.add_row("max drawdown", f"{m.max_drawdown:+.2%}")
    table.add_row("Calmar", f"{m.calmar:+.2f}")
    table.add_row("win rate", f"{m.win_rate:.1%}")
    table.add_row("trades", str(m.num_trades))
    table.add_row("exposure", f"{m.exposure:.1%}")
    table.add_row("score", f"{m.score:+.3f}")
    console.print(table)

    eq = result.equity
    dd = eq / eq.cummax() - 1.0
    console.print(f"equity   {sparkline(eq)}  {eq.iloc[-1]:.3f}x")
    console.print(f"drawdown {sparkline(dd)}  {dd.min():.1%} worst")


def export_csv(result: BacktestResult, path: str) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(
        {
            "equity": result.equity,
            "position": result.positions,
            "net_return": result.net_returns,
        }
    )
    out.to_csv(p)
    return str(p)
