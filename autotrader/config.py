"""Configuration loading with sane defaults."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# bars per year, used to annualize Sharpe/Sortino/CAGR per timeframe
BARS_PER_YEAR = {
    "1m": 525_600,
    "5m": 105_120,
    "15m": 35_040,
    "30m": 17_520,
    "1h": 8_760,
    "2h": 4_380,
    "4h": 2_190,
    "6h": 1_460,
    "12h": 730,
    "1d": 365,
}

DEFAULTS: dict[str, Any] = {
    "exchange": "binance",
    "symbols": ["BTC/USDT"],
    "timeframe": "1h",
    "history_limit": 6000,
    "fees": {"taker": 0.0006, "slippage": 0.0005},
    "walkforward": {"train": 1500, "test": 500, "step": 500},
    "optimize": {"trials": 60, "min_trades": 10, "objective": "score"},
    "paper": {"starting_cash": 10000, "poll_seconds": 60},
    "knowledge_db": "knowledge.db",
    "data_cache_dir": ".cache",
    "microstructure": True,   # use extended klines (order-flow) when available
    "orderbook_gate": True,   # live-only: veto entries the L2 book disagrees with
    "orderbook_depth": 20,    # levels to sum for book imbalance
    "risk": {
        "max_gross_exposure": 1.0,
        "max_position_pct": 0.30,
        "max_drawdown_pct": 0.20,   # kill-switch
        "max_daily_loss_pct": 0.08,
    },
    # strategy book. long_only = the walk-forward-confirmed smart-beta book (hold
    # the favored low-beta coins, no shorts/leverage; pair with regime gate to sit
    # out bears). Empty/False = legacy market-neutral long-short.
    "strategy": {
        "long_only": False,
        "factor": "lowbeta",
        "lookback": 720,
        "max_weight": 0.0,   # per-coin cap (0 = none)
    },
    # directional overlay: stack market beta on the neutral book to aim bigger.
    # weight 0 = pure neutral; tuned default (0.5 / scale 2.0) gave +return AND
    # +Sharpe in backtest. leverage scales the whole book for size (more DD too).
    "directional": {
        "weight": 0.0,
        "trend_fast": 24,
        "trend_slow": 168,
        "trend_scale": 2.0,
        "leverage": 1.0,
    },
    # risk-on/off gate from stablecoin supply (the validated new edge): when supply
    # contracts (capital leaving), cut the directional beta + de-lever to sit out
    # crashes. The market-neutral factor book keeps running underneath.
    "regime": {
        "enabled": False,
        "lookback": 30,
        # risk-off behavior: "flat" = go market-neutral (conservative, validated);
        # "tsmom_short" = keep the directional overlay but short-only, so it profits
        # from confirmed downtrends while sitting out flat risk-off chop (hybrid).
        "mode": "flat",
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


@dataclass
class Config:
    raw: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path = "config.yaml") -> "Config":
        path = Path(path)
        loaded = {}
        if path.exists():
            loaded = yaml.safe_load(path.read_text()) or {}
        return cls(raw=_deep_merge(DEFAULTS, loaded))

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    @property
    def bars_per_year(self) -> float:
        return BARS_PER_YEAR.get(self["timeframe"], 8_760)
