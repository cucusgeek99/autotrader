"""Vectorized backtest engine.

Model: a strategy emits a target position for each bar. We assume the order
to reach that target is placed at the *next* bar's open-ish price (we use
close-to-close returns with a one-bar lag), and we charge fees + slippage on
every change in position size. This is transparent and fast — ideal for
running thousands of optimizer trials.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from autotrader.backtest.metrics import Metrics, compute
from autotrader.strategies.base import Strategy


@dataclass
class BacktestResult:
    equity: pd.Series
    positions: pd.Series
    net_returns: pd.Series
    metrics: Metrics


def run_backtest(
    strategy: Strategy,
    df: pd.DataFrame,
    taker_fee: float = 0.0006,
    slippage: float = 0.0005,
    bars_per_year: float = 8760,
) -> BacktestResult:
    """Run a single strategy over an OHLCV frame and return equity + metrics."""
    raw_target = strategy.generate_signals(df).clip(-1.0, 1.0)
    # enter on the next bar -> shift the target forward by one
    positions = raw_target.shift(1).fillna(0.0)

    bar_returns = df["close"].pct_change().fillna(0.0)
    gross = positions * bar_returns

    # cost charged proportionally to the traded size (per side)
    traded = positions.diff().abs().fillna(positions.abs())
    cost = traded * (taker_fee + slippage)

    net = gross - cost
    equity = (1.0 + net).cumprod()
    metrics = compute(net, positions, bars_per_year)

    return BacktestResult(
        equity=equity,
        positions=positions,
        net_returns=net,
        metrics=metrics,
    )
