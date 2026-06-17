"""Live trade ledger + decision-rule tracker.

A backtest Sharpe and a single live run don't talk to each other. This records
every rebalance's equity and turnover so we can compute a *live* out-of-sample
Sharpe and compare it to the backtest — with a decision rule committed BEFORE we
see the results, so we can't rationalize a lucky streak after the fact.

This is what "consolidating a gain" actually means in quant: not protecting one
run's dollars, but accumulating enough samples to tell skill from luck.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass

# --- pre-committed decision rule (chosen BEFORE seeing live results) ---
BACKTEST_SHARPE = 1.4        # findings_xs OOS purged walk-forward Sharpe
MIN_REBALANCES = 30          # samples before a live Sharpe means anything
LIVE_SHARPE_FLOOR = 0.7      # live must hold ~half the backtest to "survive"
MAX_TURNOVER_RATIO = 0.50    # avg traded notional / equity per tick; above = churn


@dataclass
class Ledger:
    path: str
    interval_s: float = 3600.0   # tick spacing, used to annualize the Sharpe

    def record(self, stamp: str, equity: float, target: dict,
               intents: list, mode: str) -> dict:
        sent = [i for i in intents if i.get("status") in ("sent", "simulated")]
        traded_notional = sum(abs(i.get("notional", 0) or 0) for i in sent)
        rec = {
            "ts": stamp,
            "equity": round(float(equity), 4),
            "n_orders": len(sent),
            "traded_notional": round(traded_notional, 2),
            "weights": {k.replace("/", ""): round(float(v), 4)
                        for k, v in target.items()},
            "mode": mode.split()[0],
        }
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        return rec

    def records(self) -> list:
        if not os.path.exists(self.path):
            return []
        out = []
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return out

    def summary(self) -> dict:
        recs = self.records()
        eq = [r["equity"] for r in recs if r.get("equity", 0) > 0]
        n = len(eq)
        base = {
            "n_rebalances": n,
            "min_required": MIN_REBALANCES,
            "backtest_sharpe": BACKTEST_SHARPE,
        }
        if n < 2:
            return {**base, "verdict": "insufficient — need ≥2 ticks",
                    "live_sharpe": None, "total_return_pct": None,
                    "hit_rate": None, "avg_turnover_ratio": None}

        rets = [eq[i] / eq[i - 1] - 1 for i in range(1, len(eq))]
        mean = sum(rets) / len(rets)
        var = sum((r - mean) ** 2 for r in rets) / max(1, len(rets) - 1)
        sd = math.sqrt(var)
        ppy = (365 * 24 * 3600) / self.interval_s   # periods per year
        live_sharpe = (mean / sd * math.sqrt(ppy)) if sd > 0 else 0.0
        total_ret = eq[-1] / eq[0] - 1
        wins = sum(1 for r in rets if r > 0)
        hit_rate = wins / len(rets)

        turns = [r["traded_notional"] / r["equity"]
                 for r in recs if r.get("equity", 0) > 0 and "traded_notional" in r]
        avg_turn = sum(turns) / len(turns) if turns else 0.0

        if n < MIN_REBALANCES:
            verdict = f"BUILDING — {n}/{MIN_REBALANCES} samples; too early to judge"
        elif live_sharpe >= LIVE_SHARPE_FLOOR and avg_turn <= MAX_TURNOVER_RATIO:
            verdict = ("SURVIVED — edge held live; tiny real capital is now "
                       "a defensible experiment")
        elif live_sharpe >= LIVE_SHARPE_FLOOR:
            verdict = ("MARGINAL — Sharpe held but turnover is high; costs may "
                       "erode it at size. Stay paper.")
        else:
            verdict = ("DECAYED — edge did not survive live costs. Stay paper; "
                       "we learned it cheaply, exactly as designed.")

        return {
            **base,
            "live_sharpe": round(live_sharpe, 3),
            "total_return_pct": round(total_ret * 100, 3),
            "hit_rate": round(hit_rate, 3),
            "avg_turnover_ratio": round(avg_turn, 3),
            "equity_start": eq[0],
            "equity_now": eq[-1],
            "verdict": verdict,
        }
