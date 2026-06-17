"""Desktop alerts — the bot taps you on the shoulder so you don't have to babysit.

Fires a macOS notification only on state *transitions* (not every tick), so it
never spams: kill-switch halt, verdict ready (≥30 samples), or a big equity move
since the last alert. State is persisted to dedupe across the launchd one-shots.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path


def notify(title: str, message: str) -> None:
    """Best-effort macOS notification (no-op if osascript missing/headless)."""
    try:
        subprocess.run(
            ["osascript", "-e",
             f'display notification "{message}" with title "{title}" sound name "Glass"'],
            capture_output=True, timeout=5,
        )
    except Exception:
        pass


def check_and_alert(state_path: str, *, equity: float, halted: bool,
                    halt_reason: str, n_samples: int, verdict: str,
                    target_samples: int = 30, move_pct: float = 0.03,
                    risk_on: bool | None = None) -> list:
    """Compare current state to last-seen; fire alerts on transitions. Returns the
    list of alert messages sent (for logging/tests)."""
    p = Path(state_path)
    prev = {}
    if p.exists():
        try:
            prev = json.loads(p.read_text())
        except json.JSONDecodeError:
            prev = {}
    sent = []

    # 0) regime gate flipped (risk-off→on = book redeploys into the long basket;
    #    risk-on→off = book de-risks to cash). Only fires on an actual transition,
    #    and only once we have a prior reading to compare against (no first-run spam).
    if risk_on is not None and "risk_on" in prev and prev["risk_on"] != risk_on:
        if risk_on:
            notify("🟢 autotrader RISK-ON",
                   "regime flipped risk-on — deploying lowbeta long basket")
            sent.append("REGIME: risk-on (deploying)")
        else:
            notify("🔴 autotrader RISK-OFF",
                   "regime flipped risk-off — going to cash")
            sent.append("REGIME: risk-off (cash)")

    # 1) kill-switch just tripped
    if halted and not prev.get("halted", False):
        notify("⛔ autotrader HALTED", halt_reason or "kill-switch tripped")
        sent.append(f"HALTED: {halt_reason}")

    # 2) verdict just became available (crossed the sample threshold, real verdict)
    crossed = n_samples >= target_samples and prev.get("n_samples", 0) < target_samples
    if crossed and "BUILDING" not in verdict:
        notify("🎯 autotrader verdict ready", f"{n_samples} samples — {verdict}")
        sent.append(f"VERDICT: {verdict}")

    # 3) big equity move since last alert-check
    last_eq = prev.get("equity")
    if last_eq and equity and abs(equity / last_eq - 1) >= move_pct:
        d = equity / last_eq - 1
        notify("📈 autotrader move" if d > 0 else "📉 autotrader move",
               f"equity {d:+.1%} → ${equity:,.0f}")
        sent.append(f"MOVE: {d:+.1%}")

    state = {"equity": equity, "halted": halted, "n_samples": n_samples}
    if risk_on is not None:
        state["risk_on"] = risk_on
    p.write_text(json.dumps(state))
    return sent
