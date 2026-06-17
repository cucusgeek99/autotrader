"""Performance metrics computed from a per-bar net-return series."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass
class Metrics:
    total_return: float      # cumulative, e.g. 0.42 = +42%
    cagr: float
    sharpe: float
    sortino: float
    max_drawdown: float      # negative, e.g. -0.18
    calmar: float
    win_rate: float
    num_trades: int
    exposure: float          # fraction of bars with a non-zero position
    score: float             # composite used for ranking/optimization

    def as_dict(self) -> dict:
        return asdict(self)


def _max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min()) if len(dd) else 0.0


def compute(
    net_returns: pd.Series,
    positions: pd.Series,
    bars_per_year: float,
) -> Metrics:
    """Build a Metrics bundle from per-bar net returns and the position series."""
    net = net_returns.fillna(0.0)
    n = len(net)
    equity = (1.0 + net).cumprod()

    total_return = float(equity.iloc[-1] - 1.0) if n else 0.0
    years = n / bars_per_year if bars_per_year else 0.0
    cagr = float(equity.iloc[-1] ** (1 / years) - 1.0) if years > 0 and equity.iloc[-1] > 0 else 0.0

    mean, std = net.mean(), net.std()
    sharpe = float(mean / std * np.sqrt(bars_per_year)) if std > 0 else 0.0
    downside = net[net < 0].std()
    sortino = float(mean / downside * np.sqrt(bars_per_year)) if downside > 0 else 0.0

    mdd = _max_drawdown(equity)
    calmar = float(cagr / abs(mdd)) if mdd < 0 else 0.0

    trade_changes = positions.fillna(0.0).diff().fillna(positions.iloc[0] if n else 0.0)
    num_trades = int((trade_changes != 0).sum())
    exposure = float((positions.fillna(0.0) != 0).mean()) if n else 0.0

    # win rate per trade: returns grouped by holding segment
    win_rate = _segment_win_rate(net, positions)

    # composite score: reward risk-adjusted return, penalize deep drawdowns.
    score = sharpe * (1.0 + total_return) / (1.0 + 3.0 * abs(mdd))

    return Metrics(
        total_return=round(total_return, 4),
        cagr=round(cagr, 4),
        sharpe=round(sharpe, 3),
        sortino=round(sortino, 3),
        max_drawdown=round(mdd, 4),
        calmar=round(calmar, 3),
        win_rate=round(win_rate, 3),
        num_trades=num_trades,
        exposure=round(exposure, 3),
        score=round(float(score), 4),
    )


def _segment_win_rate(net: pd.Series, positions: pd.Series) -> float:
    pos = positions.fillna(0.0)
    seg_id = (pos != pos.shift(1)).cumsum()
    active = pos != 0
    if not active.any():
        return 0.0
    seg_returns = net[active].groupby(seg_id[active]).sum()
    if len(seg_returns) == 0:
        return 0.0
    return float((seg_returns > 0).mean())
