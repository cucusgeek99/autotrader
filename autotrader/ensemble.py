"""Ensemble strategy — blend several champions into one position.

Rather than betting on a single best strategy, combine the top-k champions for
the current regime, weighting each by its out-of-sample score. Diversification
across uncorrelated strategy ideas is one of the few real free lunches.
"""
from __future__ import annotations

import pandas as pd

from autotrader.knowledge.store import Champion, KnowledgeStore
from autotrader.strategies.base import Strategy
from autotrader.strategies.registry import get


class EnsembleStrategy(Strategy):
    """Weighted blend of member strategies. Not registered — built at runtime."""

    name = "ensemble"

    def __init__(self, members: list[tuple[Strategy, float]]):
        # members: list of (strategy_instance, weight)
        self.members = members
        total = sum(w for _, w in members) or 1.0
        self.weights = [w / total for _, w in members]
        self.params = {
            "members": [repr(s) for s, _ in members],
            "weights": [round(w, 3) for w in self.weights],
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        blended = pd.Series(0.0, index=df.index)
        for (strat, _), w in zip(self.members, self.weights):
            blended = blended.add(strat.generate_signals(df) * w, fill_value=0.0)
        return blended.clip(-1.0, 1.0)


def build_from_champions(champions: list[Champion]) -> EnsembleStrategy | None:
    """Build an ensemble from knowledge-base champions, weighted by OOS score.

    Only positive-OOS champions contribute (a losing strategy shouldn't get
    a vote). Returns None if nothing qualifies.
    """
    members: list[tuple[Strategy, float]] = []
    for champ in champions:
        if champ.oos_score <= 0:
            continue
        strat = get(champ.strategy)(**champ.params)
        members.append((strat, champ.oos_score))
    if not members:
        return None
    return EnsembleStrategy(members)


def build_from_knowledge(
    store: KnowledgeStore,
    symbol: str,
    timeframe: str,
    regime: str,
    k: int = 3,
) -> EnsembleStrategy | None:
    champions = store.top_for(symbol, timeframe, regime, k=k, min_oos=0.0)
    return build_from_champions(champions)
