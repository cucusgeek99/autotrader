"""Walk-forward splitting — rolling train/test windows over time."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class Fold:
    train: pd.DataFrame
    test: pd.DataFrame
    index: int


def make_folds(
    df: pd.DataFrame, train: int, test: int, step: int
) -> list[Fold]:
    """Generate rolling folds. Each fold trains on `train` bars and tests on
    the following `test` bars; the window then rolls forward by `step` bars."""
    folds: list[Fold] = []
    start = 0
    i = 0
    n = len(df)
    while start + train + test <= n:
        tr = df.iloc[start : start + train]
        te = df.iloc[start + train : start + train + test]
        folds.append(Fold(train=tr, test=te, index=i))
        start += step
        i += 1
    return folds
