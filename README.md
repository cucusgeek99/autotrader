# autotrader — a self-tuning crypto trading lab

A trading bot that **sets itself up**: it tunes a library of strategies against
real market data, validates them walk-forward to avoid overfitting, remembers
which strategy+params worked in which market regime, and paper-trades the
champion live. Each run compounds its knowledge.

> ⚠️ Everything is paper/testnet by default. The **core book** (below) is the only
> component meant for real capital, and only via explicit `--arm --confirm-live`.
> A good backtest is not a promise.

## ⭐ The core book — the real-capital strategy (start here)

After the research arc (below) found no candle-based alpha that survives
out-of-sample, and 4 months of live forward-testing confirmed it, the project's
answer to *"help a trader actually make money"* is **disciplined risk-taking on
the market itself**, not a secret edge:

```
weight_i = ½ × min(1, 40% / realized_vol_30d) × [close > SMA200]     # BTC, ETH; rest in USDT
```

Spot, long-only, **no leverage, no liquidation risk**, one decision per day, a
2%-of-equity no-trade band, and a −45% catastrophe kill-switch.

| BTC/ETH daily, fees incl. | CAGR | Sharpe | Max DD | Worst year |
|---|---|---|---|---|
| hold 50/50, 2017→2026 | +36.6% | 0.80 | **−87.9%** | −76.9% |
| **core**, 2017→2026 | +25.9% | **0.99** | **−37.8%** | −17.1% |
| hold 50/50, 2022→2026 | +3.9% | 0.35 | −68.3% | −65.1% |
| **core**, 2022→2026 | **+21.8%** | **0.88** | **−21.1%** | −5.3% |

Textbook parameters, not tuned: every cell of a 12-combo grid (SMA 100–250 ×
vol 30–60%) has Sharpe 0.88–1.16 and maxDD −24%..−53%. The price of the
insurance: it lags buy&hold in a straight-up bull market.

**Held-out check (10 coins never used to design it, each traded alone):** the
drawdown cut is robust — smaller maxDD on **10/10** coins (median −49% vs −93%
holding). The Sharpe gain is NOT: better on only 5/10. So the promise is
"−40% instead of −90%", not "more return"; and asset choice matters — on alts
the rule still loses (2022+ median ≈ +1%/yr vs −11% holding). Keep it on BTC/ETH.

```bash
.venv/bin/python scripts/backtest_core.py        # reproduce the table + grid
.venv/bin/python scripts/core_live.py            # PAPER run (daily via launchd: com.autotrader.core)
# real account, in this order — keys: BINANCE_SPOT_API_KEY / BINANCE_SPOT_SECRET
.venv/bin/python scripts/core_live.py --account                       # dry-run on real balances
.venv/bin/python scripts/core_live.py --account --arm --confirm-live  # REAL orders
```
Every run tracks "hold 50/50" and "hold BTC" from the same start, on real
prices; the dashboard's top card shows all three curves.

## The loop

```
 data → regime → tune strategies (Optuna) → walk-forward validate
      → record to knowledge base (SQLite, keyed by regime)
      → select champion for current regime → paper-trade → repeat
```

The **knowledge base** is the memory: every experiment is stored with its
out-of-sample (OOS) score and the regime it was measured in. The live trader
asks it, each tick, for the best strategy for *current* conditions.

## Quick start

```bash
cd autotrader
python -m venv .venv && source .venv/bin/activate
pip install -e .

# 1. teach the bot — tune every strategy, store what works
autotrader lab

# 2. see what it learned
autotrader leaderboard

# 3. paper-trade the champion (or an ensemble of the top 3) on live data
autotrader paper BTC/USDT
autotrader paper BTC/USDT --ensemble 3

# 4. watch it in the browser (equity curve, leaderboard, fills)
autotrader dashboard            # → http://127.0.0.1:8000   (needs autotrader[web])

# 5. keep it tuning itself on a schedule
autotrader schedule --every 3600

# 6. risk + going live (SAFE BY DEFAULT — see TESTNET.md)
autotrader preflight --live          # check prerequisites
autotrader risk                      # inspect limits + kill-switch
autotrader live                      # dry-run the market-neutral book (sends nothing)
autotrader live --testnet --arm --loop   # continuous futures-testnet loop (fake money)

# explore
autotrader strategies
autotrader report supertrend BTC/USDT --param period=10 --param multiplier=3 --csv eq.csv
autotrader advise               # LLM proposes next experiments (needs autotrader[llm] + API key)
```

Configure markets, timeframe, fees, and walk-forward windows in `config.yaml`.
Install optional layers: `pip install -e ".[llm,web]"` for the Claude advisor and dashboard.

## Architecture

| Module | Role |
|---|---|
| `data/feed.py` | Paginated OHLCV via ccxt, cached to disk |
| `strategies/` | Pluggable strategy library + indicators |
| `backtest/` | Vectorized engine (fees + slippage) and metrics |
| `optimize/` | Optuna tuning wrapped in walk-forward validation |
| `knowledge/` | Regime classifier + SQLite experiment memory |
| `live/` | Paper broker + champion-selecting live loop |
| `lab.py` | Orchestrates tune → record → select |

## Anti-overfitting

Every tuned strategy reports an **in-sample** score and an **out-of-sample**
score (mean across walk-forward test windows). The `overfit_gap` between them
is the alarm: a strategy that looks great in-sample but collapses OOS is
curve-fit, and the leaderboard ranks by OOS so it won't be chosen.

## The house strategy & the research

The single-asset lab is the playground; the real result is **`findings_xs`** — a
cross-sectional, market-neutral book built from rigorously-validated findings
(see `EDGE.md`). Every design choice is evidence-backed: low-vol core + breadth
factors, inverse-turnover weighted, no-trade band, dollar-neutral.

```bash
autotrader findings --expanded     # characterize it: in-sample + OOS + deflated Sharpe
autotrader xsection                # cross-sectional factor research (rank IC, long-short)
autotrader xsection --oos          # walk-forward OOS + deflated Sharpe per factor
autotrader paper-xs                # trade the book live on paper, market-neutral
```

The research harness — `features` (rank IC), `purged` (purged/nested walk-forward),
`panel` (cross-symbol validation), `findings` (deflated Sharpe) — is reusable to
vet *any* future signal. Honest standing result: `findings_xs` is a real,
low-turnover, market-neutral sleeve (OOS Sharpe ~1.3) that does **not** clear the
deflated-Sharpe significance bar alone — a component, not a proven strategy.
Finding more requires new data (finer bars / on-chain / basis), not more
reprocessing. **The harness refuses to ship false positives — that is the point.**

## What's built

- **13 strategies**: MA crossover, RSI / Bollinger / stochastic reversion,
  breakout, MACD, momentum, supertrend, order-flow (momentum/reversion/divergence,
  edge composite), funding fade — all long/flat/short and Optuna-tunable.
- **Ensembles**: `--ensemble K` blends the top-K champions for the current
  regime, weighted by OOS score (`autotrader/ensemble.py`).
- **LLM advisor** (`knowledge/llm_advisor.py`): Claude reads the leaderboard +
  regime and proposes tighter param ranges and new hypotheses, with structured
  output and prompt caching. Degrades gracefully with no API key.
- **Web dashboard** (`dashboard/`): live equity curve, leaderboard, fills.
- **Scheduler**: `autotrader schedule` re-tunes on an interval.
- **Hardened broker**: fees are reserved so cash never goes negative.

## Roadmap (designed-for extension points)

- **RL agent** — drop a learned policy in as another `Strategy` subclass.
- **Multi-symbol portfolio allocation** across simultaneous champions.
- **Live execution adapter** — swap `PaperBroker` for a real ccxt broker
  behind the same interface (only after extensive paper validation).

## Adding a strategy

Subclass `Strategy`, declare a `param_space`, decorate with `@register`:

```python
@register
class MyStrategy(Strategy):
    name = "my_strategy"
    param_space = {"window": ("int", 5, 100)}
    def generate_signals(self, df):
        ...  # return target position in [-1, 1]
```

It's automatically picked up by the optimizer, lab, and live trader.
