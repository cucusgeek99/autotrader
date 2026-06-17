"""Strategy parameter optimization with anti-overfit walk-forward validation.

Two numbers come out of `tune`:
  * deploy_params  — params fit on the *entire* history; what to run live now.
  * oos_score      — mean score across walk-forward *test* windows; the honest
                     estimate of how much to trust those params on unseen data.

A wide gap between in-sample and OOS score is the overfitting alarm.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import optuna
import pandas as pd

from autotrader.backtest.engine import run_backtest
from autotrader.optimize.walkforward import make_folds
from autotrader.strategies.base import Strategy

optuna.logging.set_verbosity(optuna.logging.WARNING)
log = logging.getLogger("autotrader.optimize")


@dataclass
class TuneResult:
    strategy: str
    deploy_params: dict[str, Any]
    insample_score: float
    oos_score: float
    oos_metrics: dict[str, float] = field(default_factory=dict)
    n_folds: int = 0
    overfit_gap: float = 0.0


def _suggest(trial: optuna.Trial, name: str, spec: tuple) -> Any:
    kind = spec[0]
    if kind == "int":
        return trial.suggest_int(name, spec[1], spec[2])
    if kind == "float":
        return trial.suggest_float(name, spec[1], spec[2])
    if kind == "categorical":
        return trial.suggest_categorical(name, spec[1])
    raise ValueError(f"unknown spec for {name}: {spec}")


def _score_params(
    strategy_cls: type[Strategy],
    params: dict,
    df: pd.DataFrame,
    fees: dict,
    bars_per_year: float,
    min_trades: int,
) -> tuple[float, dict]:
    strat = strategy_cls(**params)
    res = run_backtest(
        strat,
        df,
        taker_fee=fees["taker"],
        slippage=fees["slippage"],
        bars_per_year=bars_per_year,
    )
    m = res.metrics
    if m.num_trades < min_trades:
        return -1e9, m.as_dict()
    return m.score, m.as_dict()


def _optimize_window(
    strategy_cls: type[Strategy],
    df: pd.DataFrame,
    fees: dict,
    bars_per_year: float,
    trials: int,
    min_trades: int,
) -> tuple[dict, float]:
    space = strategy_cls.param_space

    def objective(trial: optuna.Trial) -> float:
        params = {name: _suggest(trial, name, spec) for name, spec in space.items()}
        score, _ = _score_params(
            strategy_cls, params, df, fees, bars_per_year, min_trades
        )
        return score

    if not space:  # parameterless strategy
        score, _ = _score_params(strategy_cls, {}, df, fees, bars_per_year, min_trades)
        return {}, score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    return study.best_params, study.best_value


def tune(
    strategy_cls: type[Strategy],
    df: pd.DataFrame,
    fees: dict,
    bars_per_year: float,
    wf: dict,
    trials: int = 60,
    min_trades: int = 10,
) -> TuneResult:
    """Walk-forward tune a strategy over `df`."""
    folds = make_folds(df, wf["train"], wf["test"], wf["step"])

    oos_scores: list[float] = []
    oos_metric_rows: list[dict] = []
    for fold in folds:
        best_params, _ = _optimize_window(
            strategy_cls, fold.train, fees, bars_per_year, trials, min_trades
        )
        oos_score, oos_metrics = _score_params(
            strategy_cls, best_params, fold.test, fees, bars_per_year, min_trades
        )
        # ignore folds that couldn't trade enough to be meaningful
        if oos_score > -1e8:
            oos_scores.append(oos_score)
            oos_metric_rows.append(oos_metrics)

    # params to actually deploy: fit on the full history
    deploy_params, insample_score = _optimize_window(
        strategy_cls, df, fees, bars_per_year, trials, min_trades
    )

    oos_score = float(sum(oos_scores) / len(oos_scores)) if oos_scores else 0.0
    oos_metrics = _avg_metrics(oos_metric_rows)

    return TuneResult(
        strategy=strategy_cls.name,
        deploy_params=deploy_params,
        insample_score=round(float(insample_score), 4),
        oos_score=round(oos_score, 4),
        oos_metrics=oos_metrics,
        n_folds=len(oos_scores),
        overfit_gap=round(float(insample_score) - oos_score, 4),
    )


def _avg_metrics(rows: list[dict]) -> dict[str, float]:
    if not rows:
        return {}
    keys = rows[0].keys()
    return {k: round(float(sum(r[k] for r in rows) / len(rows)), 4) for k in keys}
