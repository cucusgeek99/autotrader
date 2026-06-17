# autotrader — project summary

A self-tuning crypto trading **research lab and paper-trading system**, built from
an ambitious idea ("a bot that sets itself up with strategies based on testing,
tuning, and knowledge") into a rigorous, honest quant research pipeline.

> ⚠️ **Paper only.** No real money, no API keys. A good backtest is not a promise.

## Two halves

### 1. The system (a working product)
A self-tuning lab: pulls market data, tunes a 13-strategy library with Optuna,
validates walk-forward, classifies market regime, stores champions in a SQLite
knowledge base, and paper-trades the champion live — with an LLM advisor, a web
dashboard, and a scheduler. `autotrader lab` → `autotrader dashboard`.

### 2. The research (a rigorous edge hunt)
A genuine quant research program that hunted for tradeable edge and — crucially —
**refused to fool itself**. Six signal families were vetted with the tools real
quants use:

- **Rank IC** analysis (does the signal predict forward returns?)
- **Purged + embargoed** walk-forward (no label leakage)
- **Nested** hyperparameter tuning (no selection peek)
- **Cross-symbol panel** validation (held-out instruments)
- **Deflated Sharpe Ratio** (multiple-testing correction)
- **Cross-sectional, market-neutral** portfolio construction (removes beta)

## What we found (full honesty in `EDGE.md`)

| Signal | Verdict |
|---|---|
| Order-flow / microstructure @1h | Ruled out — real IC, no net edge across instruments |
| Funding fade / momentum @1h | Ruled out — near break-even, doesn't generalize |
| Cross-sectional momentum @1h | Significantly *negative* (crypto reversal regime) |
| Cross-sectional reversal | Real IC, cost-killed (turnover) |
| **Cross-sectional low-vol + breadth** | **Real, OOS-stable, OOS Sharpe ~1.3 — sub-threshold on DSR** |

**The house strategy `findings_xs`** synthesizes the survivors: a market-neutral,
low-turnover, cross-sectional book (low-vol core + breadth factors, cost-aware,
banded). Out-of-sample Sharpe ~1.3 — the best evidence-backed result — but it does
not clear the deflated-Sharpe significance bar alone. It is a **component, not a
proven strategy.**

## The honest conclusion

After leak-free, peek-free, cost-aware, multiple-testing-corrected validation, the
1h OHLCV/funding data yields one real-but-weak market-neutral sleeve and no
significant standalone edge. That is a **legitimate, common, and valuable research
outcome** — most "profitable backtests" are leaks this pipeline would have caught.
Finding genuine edge from here requires *new information* (finer 1–5m bars,
on-chain, basis/term-structure), not more reprocessing.

The lasting deliverables: a working self-tuning system, a reusable validation
harness that won't ship false positives, and `findings_xs` — a real, if modest,
market-neutral sleeve you can watch trade live on the dashboard.

## Run it

```bash
cd autotrader && python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,llm,web]"

autotrader lab                  # tune the single-asset lab
autotrader dashboard            # watch it (http://127.0.0.1:8000)
autotrader findings --expanded  # characterize the house strategy
autotrader paper-xs             # trade the market-neutral book live on paper
```

43 tests passing. See `README.md` for the full command surface and `EDGE.md` for
the complete research write-up.
