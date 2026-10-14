# autotrader — project summary

A self-tuning crypto trading **research lab and paper-trading system**, built from
an ambitious idea ("a bot that sets itself up with strategies based on testing,
tuning, and knowledge") into a rigorous, honest quant research pipeline.

> ⚠️ Paper/testnet by default; the core book is the only real-capital component (explicit `--arm --confirm-live`).

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

## 4-month live forward test (2026-06-18 → 2026-10-14) — the verdict that redirected the project

| Sleeve | Result | vs just holding |
|---|---|---|
| lowbeta long-only (testnet) | testnet +16.8%; **same weights on mainnet prices +20.2%**, Sharpe 2.2, DD −8.6% | BTC hold **+26.7%**, eq-wt 12 +28.0% → no alpha, just lower beta |
| delta-neutral carry (paper) | **−1.3%** — 54 entries, ~$120 fees vs +$48 funding | turnover wall again → stopped |
| daily dip-buyer (paper) | −0.3%, 4 trades in 4 months | inconclusive |

Lessons: testnet prices flatter (Sharpe 6.6 there vs 2.2 on mainnet) — always
replay on mainnet; and nothing beat holding the market. Hence the pivot to the
**core book** (README): capture market upside with vol-targeting + SMA200 so a
trader never sits through −80%. Backtest 2017→2026: maxDD −38% vs −88% for
holding, Sharpe 0.99 vs 0.80; since 2022 +21.8%/yr vs +3.9%/yr.

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
