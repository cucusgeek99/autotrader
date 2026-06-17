"""Persistent knowledge base — every tuning experiment is recorded.

This is what makes the bot "set itself up from knowledge each time": instead
of re-deriving everything, it queries past experiments for the champion
strategy+params that worked in the *current* regime for a given symbol.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

_SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at    TEXT NOT NULL,
    symbol        TEXT NOT NULL,
    timeframe     TEXT NOT NULL,
    regime        TEXT NOT NULL,
    strategy      TEXT NOT NULL,
    params        TEXT NOT NULL,
    insample_score REAL NOT NULL,
    oos_score     REAL NOT NULL,
    overfit_gap   REAL NOT NULL,
    n_folds       INTEGER NOT NULL,
    metrics       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_lookup
    ON experiments (symbol, timeframe, regime, oos_score);
"""


@dataclass
class Champion:
    strategy: str
    params: dict[str, Any]
    oos_score: float
    regime: str
    created_at: str
    metrics: dict[str, float]


class KnowledgeStore:
    def __init__(self, path: str = "knowledge.db"):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def record(
        self,
        *,
        created_at: str,
        symbol: str,
        timeframe: str,
        regime: str,
        strategy: str,
        params: dict,
        insample_score: float,
        oos_score: float,
        overfit_gap: float,
        n_folds: int,
        metrics: dict,
    ) -> int:
        cur = self._conn.execute(
            """INSERT INTO experiments
               (created_at, symbol, timeframe, regime, strategy, params,
                insample_score, oos_score, overfit_gap, n_folds, metrics)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                created_at, symbol, timeframe, regime, strategy,
                json.dumps(params), insample_score, oos_score,
                overfit_gap, n_folds, json.dumps(metrics),
            ),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def best_for(
        self, symbol: str, timeframe: str, regime: str, min_oos: float = 0.0
    ) -> Optional[Champion]:
        """Best champion known for this symbol+timeframe+regime."""
        row = self._conn.execute(
            """SELECT * FROM experiments
               WHERE symbol=? AND timeframe=? AND regime=? AND oos_score>=?
               ORDER BY oos_score DESC, created_at DESC LIMIT 1""",
            (symbol, timeframe, regime, min_oos),
        ).fetchone()
        return self._to_champion(row) if row else None

    def top_for(
        self, symbol: str, timeframe: str, regime: str, k: int = 3, min_oos: float = 0.0
    ) -> list[Champion]:
        """Top-k distinct champions for a regime, best OOS first.

        De-duplicates by strategy so the ensemble blends *different* ideas
        rather than k variants of the same one.
        """
        rows = self._conn.execute(
            """SELECT * FROM experiments
               WHERE symbol=? AND timeframe=? AND regime=? AND oos_score>=?
               ORDER BY oos_score DESC, created_at DESC""",
            (symbol, timeframe, regime, min_oos),
        ).fetchall()
        out: list[Champion] = []
        seen: set[str] = set()
        for r in rows:
            if r["strategy"] in seen:
                continue
            seen.add(r["strategy"])
            out.append(self._to_champion(r))
            if len(out) >= k:
                break
        return out

    def best_any_regime(
        self, symbol: str, timeframe: str, min_oos: float = 0.0
    ) -> Optional[Champion]:
        """Fallback champion ignoring regime, if none matched the current one."""
        row = self._conn.execute(
            """SELECT * FROM experiments
               WHERE symbol=? AND timeframe=? AND oos_score>=?
               ORDER BY oos_score DESC, created_at DESC LIMIT 1""",
            (symbol, timeframe, min_oos),
        ).fetchone()
        return self._to_champion(row) if row else None

    def leaderboard(self, limit: int = 20) -> list[dict]:
        rows = self._conn.execute(
            """SELECT symbol, timeframe, regime, strategy, params,
                      oos_score, insample_score, overfit_gap, created_at
               FROM experiments ORDER BY oos_score DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["params"] = json.loads(d["params"])
            out.append(d)
        return out

    @staticmethod
    def _to_champion(row: sqlite3.Row) -> Champion:
        return Champion(
            strategy=row["strategy"],
            params=json.loads(row["params"]),
            oos_score=row["oos_score"],
            regime=row["regime"],
            created_at=row["created_at"],
            metrics=json.loads(row["metrics"]),
        )

    def close(self) -> None:
        self._conn.close()
