"""Strategy abstraction.

A Strategy turns an OHLCV DataFrame into a *target position* series in
[-1, 1]:  +1 = fully long, 0 = flat, -1 = fully short. The backtest engine
handles entry timing (next-bar fills), fees, and slippage — strategies stay
pure and declarative.

Each strategy declares a `param_space`: a dict of name -> spec that the
optimizer samples from. Specs:
    ("int", low, high)            integer in [low, high]
    ("float", low, high)          float in [low, high]
    ("categorical", [a, b, ...])  one of the choices
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

ParamSpace = dict[str, tuple]


class Strategy(ABC):
    name: str = "base"
    param_space: ParamSpace = {}

    def __init__(self, **params: Any):
        # fill defaults for any unspecified params (midpoint / first choice)
        merged = {k: self._default(spec) for k, spec in self.param_space.items()}
        merged.update(params)
        self.params = merged

    @staticmethod
    def _default(spec: tuple) -> Any:
        kind = spec[0]
        if kind == "int":
            return int((spec[1] + spec[2]) // 2)
        if kind == "float":
            return (spec[1] + spec[2]) / 2
        if kind == "categorical":
            return spec[1][0]
        raise ValueError(f"unknown param spec: {spec}")

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Return target position in [-1, 1] aligned to df.index."""

    def __repr__(self) -> str:
        ps = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.name}({ps})"
