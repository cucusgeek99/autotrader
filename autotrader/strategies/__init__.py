"""Strategy package. Importing it registers all built-in strategies."""
from autotrader.strategies import library  # noqa: F401  (triggers registration)
from autotrader.strategies import orderflow  # noqa: F401  (order-flow strategies)
from autotrader.strategies.base import Strategy
from autotrader.strategies.registry import all_strategies, get, register

__all__ = ["Strategy", "register", "get", "all_strategies"]
