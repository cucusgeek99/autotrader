"""Risk management — the gate every order must pass before it reaches a broker.

A trading bot without hard risk limits is just an automated way to lose money.
This enforces, regardless of what the strategy wants:
  * per-position cap (no single coin can dominate the book)
  * gross-exposure cap (total long+short ≤ a multiple of equity)
  * a drawdown KILL-SWITCH (equity falls X% from peak → halt + flatten, sticky
    until a human resumes)
  * a daily-loss limit (resets each day)

The kill-switch is sticky on purpose: a deep drawdown should require a human to
look at *why* before trading resumes. State persists to disk so a restart can't
silently un-halt a tripped bot.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class RiskLimits:
    max_gross_exposure: float = 1.0    # total |weights| ≤ 1.0 × equity
    max_position_pct: float = 0.30     # any single coin ≤ 30% of equity
    max_drawdown_pct: float = 0.20     # kill-switch: −20% from peak → halt
    max_daily_loss_pct: float = 0.08   # −8% in a day → halt for the day


@dataclass
class RiskState:
    peak_equity: float = 0.0
    day_start_equity: float = 0.0
    day_key: str = ""
    halted: bool = False
    halt_reason: str = ""


class RiskManager:
    def __init__(self, limits: RiskLimits | None = None,
                 state_path: str = ".cache/risk_state.json"):
        self.limits = limits or RiskLimits()
        self.state_path = state_path
        self.state = self._load()

    # ---- equity tracking + kill-switch ------------------------------
    def update_equity(self, equity: float, day_key: str = "") -> RiskState:
        s = self.state
        if s.peak_equity <= 0:
            s.peak_equity = equity
        if s.day_start_equity <= 0 or (day_key and day_key != s.day_key):
            s.day_start_equity = equity
            s.day_key = day_key
            # a new day clears a *daily* halt, but NOT a drawdown halt
            if s.halted and s.halt_reason.startswith("daily"):
                s.halted, s.halt_reason = False, ""
        s.peak_equity = max(s.peak_equity, equity)

        dd = equity / s.peak_equity - 1.0 if s.peak_equity else 0.0
        daily = equity / s.day_start_equity - 1.0 if s.day_start_equity else 0.0
        if not s.halted:
            if dd <= -self.limits.max_drawdown_pct:
                s.halted = True
                s.halt_reason = f"drawdown {dd:.1%} ≤ −{self.limits.max_drawdown_pct:.0%} (KILL-SWITCH)"
            elif daily <= -self.limits.max_daily_loss_pct:
                s.halted = True
                s.halt_reason = f"daily loss {daily:.1%} ≤ −{self.limits.max_daily_loss_pct:.0%}"
        self._save()
        return s

    def resume(self) -> None:
        """Manual override — clear a tripped kill-switch after human review."""
        self.state.halted = False
        self.state.halt_reason = ""
        self._save()

    @property
    def halted(self) -> bool:
        return self.state.halted

    # ---- order gating -----------------------------------------------
    def check_weights(self, target: dict) -> tuple[dict, list[str]]:
        """Clamp a target weight vector to the limits. Returns (adjusted, notes)."""
        notes = []
        if self.state.halted:
            return ({k: 0.0 for k in target}, [f"HALTED: {self.state.halt_reason} → flatten all"])

        cap = self.limits.max_position_pct
        adj = {}
        for k, w in target.items():
            cw = max(-cap, min(cap, w))
            if cw != w:
                notes.append(f"{k}: capped {w:+.2f}→{cw:+.2f} (max {cap:.0%})")
            adj[k] = cw

        gross = sum(abs(w) for w in adj.values())
        if gross > self.limits.max_gross_exposure and gross > 0:
            scale = self.limits.max_gross_exposure / gross
            adj = {k: w * scale for k, w in adj.items()}
            notes.append(f"gross {gross:.2f}→{self.limits.max_gross_exposure:.2f} (scaled ×{scale:.2f})")
        return adj, notes

    # ---- persistence ------------------------------------------------
    def _load(self) -> RiskState:
        p = Path(self.state_path)
        if p.exists():
            return RiskState(**json.loads(p.read_text()))
        return RiskState()

    def _save(self) -> None:
        p = Path(self.state_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self.state), indent=2))
