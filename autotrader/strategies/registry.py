"""Strategy registry — decorator-based auto-registration."""
from __future__ import annotations

from autotrader.strategies.base import Strategy

_REGISTRY: dict[str, type[Strategy]] = {}


def register(cls: type[Strategy]) -> type[Strategy]:
    _REGISTRY[cls.name] = cls
    return cls


def get(name: str) -> type[Strategy]:
    if name not in _REGISTRY:
        raise KeyError(f"unknown strategy '{name}'. available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def all_strategies() -> dict[str, type[Strategy]]:
    return dict(_REGISTRY)
