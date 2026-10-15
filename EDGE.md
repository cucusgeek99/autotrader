# Edge research — order-flow & microstructure

This documents the **measured** search for genuine predictive edge, not assumed
edge. The methodology is deliberately skeptical: measure feature predictiveness
first (information coefficient), then build strategies only on what survives,
then judge those strategies walk-forward out-of-sample.

## What's backtestable vs live-only (the honest split)

| Signal | Source | Historical? | Status |
|---|---|---|---|
| Order-flow imbalance (taker buy/sell) | Binance extended klines (12-field) | ✅ | Backtested |
| VWAP deviation, Amihud illiquidity, realized vol, trade intensity | Extended klines | ✅ | Backtested |
| Funding rate / open interest | Perp history endpoints | ✅ | Available (perps) |
| **L2 book imbalance / microprice / spread** | Live `fetch_order_book` | ❌ **no history** | **Live-only execution gate** |

`fetch_order_book` returns only the *current* snapshot — there is no historical
L2 depth via ccxt. So book imbalance/microprice are used as a **live entry
filter** (`live/orderbook.py`), never backtested. Pretending otherwise would
fabricate history.

## Feature IC results (BTC/USDT 1h, 4000 bars, run `autotrader features BTC/USDT`)

IC = Spearman rank correlation of the feature vs the forward N-bar return.
`|t| > 2` ≈ unlikely to be noise. In liquid markets, a *stable* IC of even
0.03–0.07 is genuine, exploitable edge — anyone reporting IC of 0.3 is fitting
noise.

| Feature | h=1 | h=4 | h=12 |
|---|---|---|---|
| Amihud illiquidity | 0.030 (noise) | **+0.057 (t 3.6)** | **+0.075 (t 4.7)** |
| Trade intensity | 0.020 (noise) | **+0.048 (t 3.1)** | **+0.043 (t 2.7)** |
| VWAP deviation | −0.030 (noise) | **−0.039 (t −2.5)** | −0.022 (noise) |
| Order-flow momentum (12) | −0.004 | 0.005 | **+0.046 (t 2.9)** |
| Order-flow imbalance (6) | −0.004 | −0.015 | **+0.037 (t 2.3)** |

**Findings:**
1. **No bar-to-bar edge.** At horizon 1, *every* feature is noise — hourly
   order flow doesn't predict the next single hour. HFT arbs that away.
2. **Edge emerges at the multi-hour horizon.** At 4–12 bars, illiquidity, trade
   intensity, VWAP deviation, and order-flow momentum/imbalance all become
   statistically significant. The signal is real but slow.
3. **Implication for strategy design:** hold for *hours*, not minutes, and
   combine features — a single feature's IC is too small to trade alone after
   fees.

## Strategy results (walk-forward OOS)

The naive fast order-flow strategies overfit exactly as the IC predicted —
trading the (non-existent) short-horizon signal:

| Strategy | OOS score | Overfit gap |
|---|---|---|
| `orderflow_momentum` | −5.04 | **+5.98** (severe overfit) |
| `flow_divergence` | −3.51 | +5.15 |
| `orderflow_reversion` | −2.68 | +1.62 |
| **`edge_composite`** | **+0.74** | **+0.29** (tightest of any momentum strategy) |

`edge_composite` combines the IC-significant features at a multi-hour hold. Its
in-sample (+1.03) and OOS (+0.74) scores are close — the signature of a *real,
modest* signal rather than a curve-fit. It's the only flow/momentum strategy
that doesn't blow up out-of-sample.

> ⚠️ **Honest caveat:** `edge_composite` derives its feature *directions* from
> the full-sample IC, a mild in-sample peek. See the purged result below — once
> that peek is removed, the edge disappears.

## Purged walk-forward result — the leak-free verdict (`autotrader purged BTC/USDT`)

`edge_composite`'s +0.74 OOS above was an artifact: it fixed feature *directions*
from the full sample, so it had already seen the future. The purged walk-forward
removes that entirely — feature weights are fit **only on each training fold**,
labels overlapping the test set are **purged**, and a `horizon + embargo` gap
separates train from test.

Result on real BTC/USDT 1h (train 1500 / test 500 / step 500):

| Horizon | Mean OOS return | Folds positive | Return t-stat | Verdict |
|---|---|---|---|---|
| 12 bars | **−6.3%** | 1/3 | −1.78 | **no leak-free edge** |
| 4 bars  | **−10.4%** | 0/3 | −2.80 | **no leak-free edge** |

**The edge does not survive a leak-free test.** The honest conclusion: the
full-sample IC was real, but **not stationary** — and the smoking gun is in the
per-fold weights:

```
horizon 12:  fold 0  flow_mom=+0.48   →   fold 2  flow_mom=-0.39
```

The sign of the most important feature **flips between folds**. A weight fit on
the recent past points the *wrong way* in the near future. A static IC measured
over the whole sample averages these flips into a number that looks predictive
but isn't tradeable, because you never know which regime you're in *ex ante*.

**This is the framework working as designed.** A naive backtest would have
shipped a losing strategy; the purged walk-forward caught the false positive.
The deliverable here is not an edge — it's a research harness rigorous enough to
*tell you when you don't have one*, which is the more valuable thing.

## Regime-conditional weighting — the hypothesis the weight-flips pointed to

If the global weights flip because the relationship is *regime-dependent*, then
fitting weights **per regime** (still leak-free: per training fold, purged,
embargoed) should recover stability. Run: `autotrader purged BTC/USDT --regime`.

**1. The signal IS more stationary within regimes.** The stability diagnostic
(does each regime's dominant feature keep its sign across folds?):

| Regime | Strongest feature | Across folds |
|---|---|---|
| trending_down | trade_z (+0.31) | stable |
| ranging_calm | trade_z (+0.38) | stable |
| ranging_volatile | flow_mom (-0.36) | stable |
| trending_up | vwap_dev (-0.20) | flips |

3 of 4 regimes are now sign-stable — versus the global fit where the dominant
feature flipped +0.48 -> -0.39. **The non-stationarity was largely regime-driven,
as hypothesized.**

**2. The signal is real GROSS, but turnover kills it net.** Isolating costs by
re-running with zero fees:

| | Gross (0 fees) | Net (real 0.11% round-trip) |
|---|---|---|
| Regime-conditional, threshold 0.5 | **+3.16%** OOS, Sharpe +1.18 | **-5.44%** |

The leak-free signal makes money *before costs*. At ~40 trades/fold, fees impose
an ~8.6%/fold drag that swamps it. "Real alpha, wrong turnover."

**3. Cutting turnover (raise the conviction threshold) lifts net positive:**

| Threshold | Trades/fold | Net OOS | Gross OOS |
|---|---|---|---|
| 0.5 | 37 | -5.44% | +3.16% |
| 1.0 | 32 | **+1.54%** | +7.36% |
| 1.5 | 29 | **+3.45%** | +6.80% |
| **2.0** | **16** | **+3.85%** | +5.68% |
| 2.5 | 10 | +2.00% | +3.05% |

Net OOS is **positive across the whole 1.0-2.5 range** — robustness, not a single
lucky point. At threshold 2.0: mean OOS **+3.85%**, mean Sharpe **+1.82**, folds
+7.1% / +9.0% / -4.6%.

**Honest verdict — promising but NOT yet significant.** Return t-stat at the
sweet spot is **0.91** (one of three folds lost money). With only 3 folds this is
encouraging, not proven. Two caveats remain:
  * **Sample size**: 3 folds / 4000 bars is too few for significance. Needs more
    history and more symbols.
  * **Threshold selection**: the threshold was picked on the OOS panel — a small
    second-order peek. Range-robustness mitigates it; the clean fix is **nested**
    per-fold threshold tuning.

The arc is real and leak-free: **regime conditioning + turnover control turned a
look-ahead artifact into a genuinely net-positive (if not-yet-significant)
out-of-sample signal**, and we can name which feature drives each regime.

## Nested threshold tuning — removing the last peek

At threshold 2.0 above, the threshold was picked by *looking at the outer OOS
panel* — a small second-order leak. Nested tuning removes it: for each outer
fold, the threshold is selected on an **inner walk-forward carved from that
fold's training data only** (`autotrader purged BTC/USDT --nested`). The outer
test set is never seen during selection.

**Validation — the inner WF independently chose high thresholds: [2.0, 2.5, 2.5].**
It converged on low-turnover settings *without* seeing the outer test sets — so
the earlier manual pick of 2.0 was **not** cherry-picked. The cost/turnover story
holds up under a clean protocol.

**Peek-free result (the cleanest number in this doc):**

| | Mean OOS | Sharpe | Folds + | Return t-stat |
|---|---|---|---|---|
| Nested (per-fold threshold) | **+4.29%** | **+2.28** | 2/3 | **1.47** |

Folds: +7.08% / +7.35% / −1.56%. Removing the threshold peek **didn't break the
edge — it slightly improved it** (vs +3.85% at the hand-picked 2.0), because the
inner selection adapts per fold.

**Verdict: robustly promising, methodologically clean, still not significant.**
t-stat 1.47 < 2; one of three folds loses. Every *look-ahead* peek is now gone —
weights, standardization, and threshold are all fit inside the training window.
What remains is purely a **sample-size** problem.

> **The one honest caveat left:** feature set, lookback (24), horizon (12), and
> the threshold grid were all chosen from the *same BTC sample* used here —
> study-level researcher degrees of freedom. The truly clean test is on symbols
> and time periods never touched during development. That is exactly what
> multi-symbol / panel validation provides, and it's the binding next step.

## Panel validation — the definitive verdict (`autotrader panel`)

The nested BTC result (+4.29%, t=1.47) was on **3 folds**. The honest test: run
the identical peek-free nested pipeline on **5 coins never used in development**
(ETH, SOL, BNB, XRP, ADA), with **8000 bars / 21 folds** each for real power, and
pool the result. Crypto is cross-correlated, so the headline t-stat collapses
each period into an equal-weight portfolio across the held-out coins (absorbing
cross-sectional correlation), then takes the t-stat across periods.

| Symbol | Role | Folds | Mean OOS | Sharpe | t-stat | Pos |
|---|---|---|---|---|---|---|
| BTC | dev | 21 | −1.93% | −2.28 | −1.67 | 4/21 |
| ETH | held-out | 21 | −2.51% | −0.96 | −1.27 | 7/21 |
| SOL | held-out | 21 | −1.74% | −0.69 | −0.79 | 8/21 |
| BNB | held-out | 21 | −4.62% | −2.21 | −2.63 | 8/21 |
| XRP | held-out | 21 | −7.05% | −3.61 | −2.85 | 6/21 |
| ADA | held-out | 21 | −3.13% | −1.28 | −1.65 | 6/21 |

- **Held-out symbols net-positive: 0/5.**
- **Portfolio (honest): mean −3.81%, t = −3.58** — significantly *negative*.
- **Even BTC turns negative** (−1.93%) once measured on 21 folds instead of 3.

### Conclusion: there is no tradeable edge here.

The promising +4.29% was a **small-sample fluke** on 3 BTC folds. The framework
flagged it correctly at the time (t=1.47, "not significant"); proper statistical
power confirmed it was noise. The strategy does **not** generalize across
instruments or survive a larger sample.

What survived and what didn't:
- ✅ **True:** order-flow features carry weak, statistically-significant IC at the
  4–12 bar horizon (replicated, leak-free). The *raw signal* is real.
- ❌ **False:** that a tradeable, net-of-cost strategy can be built on them at 1h.
  After fees, across instruments, with adequate folds — it loses.

**This is the project's most valuable result, not its failure.** A naive
backtest would have shipped a confident loser. Each escalating layer of rigor —
purging, regime-conditioning, nested tuning, then cross-symbol panel — stripped
away apparent edge until, properly measured, it was clearly absent. The
deliverable is a research harness that **refuses to let you fool yourself.**

## Funding-rate fade (and momentum) — second candidate, tested & ruled out

A structurally different signal: fade crowded perp positioning. Run on
binanceusdm perps with realized funding history (`autotrader funding`,
`autotrader panel --funding`). Two genuine advantages over order flow: it trades
**rarely** (low turnover → costs barely bite) and a short on high-positive funding
**receives the carry**. We model price-only (conservative) and carry-inclusive
(realistic).

**Funding IC on BTC (dev):** funding z-score has *positive* IC vs forward returns
(+0.029, t=2.61 at 8 bars) — high funding precedes *higher* returns here
(momentum), the opposite of the classic fade prior. So the fade is the wrong sign
on this sample, and indeed it loses.

**Cross-symbol panel (5 held-out coins, 21 folds each, carry included):**

| Direction (chosen on BTC) | Held-out portfolio mean | t-stat | Symbols + |
|---|---|---|---|
| **Fade** (short high funding) | −1.30% | −1.53 | 0/5 |
| **Momentum** (long high funding) | −1.83% | −1.78 | 1/5 |

**Both directions fail out-of-sample.** Momentum looked marginally positive on
BTC (+0.31%) — but that's the dev symbol where its IC was measured (the in-sample
peek), and it did **not** transfer to held-out coins. The dev/held-out split
caught the peek exactly as intended.

Two honest nuances:
- Funding losses are **much milder** than order flow (−1 to −3% vs −2 to −7%) —
  the low-turnover + carry structure keeps it near break-even. It's the
  *least-far-from-edge* signal tested, just not over the line.
- "Fade extreme funding" is itself a crowded, well-known trade; on recent data
  it's arbed flat-to-slightly-adverse, consistent with this result.

**Ruled out** as tested. A real version would need finer event timing (act only
on the most extreme funding spikes), better execution, or a cross-sectional
construction — each of which must still clear the `panel` bar.

## Cross-sectional (market-neutral) factors — the first survivor

Single-asset timing kept drowning in beta. The cross-sectional approach ranks a
signal *across* a 12-coin universe each period and goes long-top / short-bottom,
dollar-neutral — beta cancels, and every (period × coin) is an observation, so
the rank-IC t-stat finally has real power (hundreds of points, not 3 folds).
Run: `autotrader xsection`.

**Results (12 coins, 8000×1h bars, horizon 12, net of costs):**

| Factor | mean IC | IC t-stat | LS return | LS Sharpe | ann. turnover |
|---|---|---|---|---|---|
| momentum | −0.045 | **−3.21** | −53.5% | −3.70 | 533× |
| reversal | +0.035 | **+2.94** | −65.8% | −5.48 | 1027× |
| **lowvol** | **+0.059** | **+3.27** | **+6.6%** | **+0.40** | **180×** |
| orderflow | +0.017 | +0.40 | −47.5% | −4.40 | 822× |
| funding_xs | −0.005 | +0.06 | +6.0% | +0.46 | 101× |

Three real findings:
1. **Cross-sectional momentum is significantly *negative* in crypto at 12h**
   (t=−3.21) — recent relative winners mean-revert. Short-horizon reversal, not
   momentum, is the cross-sectional regime here.
2. **Reversal has significant IC (+2.94) but loses −66% net** — 1027× turnover.
   The same lesson as everywhere, now cross-sectional: real signal, killed by
   costs. (`funding_xs` LS is +0.46 Sharpe but IC t=0.06 — no real skill, so the
   positive book is not trustworthy.)
3. **`lowvol` is the first signal in this entire program to clear IC t>2 AND
   survive costs net-positive** — t=3.27, long-short Sharpe **+0.40**, return
   +6.6%, and **low turnover (180×)** so costs don't eat it. It's the
   **low-volatility anomaly**: low-vol coins outperform high-vol risk-adjusted —
   one of the most robust, economically-grounded factors in *all* of finance
   (documented across equities, bonds, commodities, FX for decades).

**Why this one is different from every prior candidate:** significant IC, net of
costs *positive*, sustainable turnover, and a strong economic prior that the
effect persists. That combination did not occur for order flow or funding.

**Honest caveats (in-sample):** whole-period, and five factors tried. Validated
below.

### Out-of-sample validation (`autotrader xsection --oos`)

Walk-forward, 7 folds, lookback chosen on each TRAIN window only (purged +
embargoed), OOS stitched from test windows; then the **Deflated Sharpe Ratio**
(Bailey & López de Prado) haircuts for the number of factors tried and return
skew/kurtosis.

| Factor | OOS IC | OOS IC-t | OOS Sharpe | OOS return |
|---|---|---|---|---|
| momentum | −0.065 | −4.17 | −6.65 | −53.3% |
| reversal | +0.044 | +3.31 | −4.25 | −39.2% (turnover) |
| **lowvol** | **+0.073** | **+2.63** | **+0.61** | **+7.3%** |
| orderflow | +0.034 | +3.21 | −2.54 | −22.7% |

**lowvol passed the walk-forward** — OOS IC still significant (t=2.63), Sharpe
*improved* to +0.61, lookbacks stable (96–168). Temporal stability is real.

**But it fails the Deflated Sharpe: DSR = 0.03.** After deflation, only ~3%
probability the true Sharpe is positive — *not distinguishable from selection
luck*. The reason is magnitude: OOS Sharpe 0.61 is small, and the haircut
(SR0=0.119, per-observation) sits above the per-observation Sharpe.

Two honest caveats, in both directions:
  * The DSR here is **somewhat conservative** — its null dispersion is estimated
    from all four trial Sharpes, two of which (momentum, orderflow) are
    *genuinely* strongly negative, not noise, which inflates the assumed
    luck-spread. A fairer DSR would be higher.
  * But that doesn't rescue it: **Sharpe 0.61 over 7 folds is thin evidence**,
    and IC t=2.63 across 4 factors is only marginal after Bonferroni (~p 0.04).

**Verdict: `lowvol` is a real, economically-grounded, OOS-stable factor — but too
*weak* to clear the strict multiple-testing bar alone.** Directionally confirmed,
not deployable solo.

### Why this is the case for multi-factor combination

A single factor (IC 0.07, Sharpe 0.6) failing the deflated-Sharpe bar is the
*empirical* argument for combining factors. Skill ≈ IC × √breadth — stacking
several weak-but-real, low-correlated factors raises the combined Sharpe past the
bar one factor can't clear. `lowvol` (confirmed) + `reversal` (significant IC,
cost-killed) is the natural pairing.

### The combination result (`autotrader xsection --combo`)

`lowvol` and `reversal` are genuinely uncorrelated (per-period rank corr −0.04 —
real breadth). Walk-forward OOS:

| Book | OOS IC-t | OOS Sharpe | OOS return |
|---|---|---|---|
| lowvol (alone) | +2.80 | +0.90 | +11.8% |
| reversal (alone) | +3.03 | **−4.63** | −39.6% |
| COMPOSITE (equal-weight) | **+3.66** | −2.76 | −28.8% |
| COMPOSITE (cost-aware, inverse-turnover) | +2.87 | **+0.91** | +12.0% |

What this shows, in order:
1. **Breadth helps the *signal*** — the equal-weight composite has the highest
   IC t-stat (3.66 > either alone). Combining uncorrelated predictors works,
   exactly as the fundamental law says.
2. **But naive equal-weighting wrecks the *book*** (Sharpe −2.76) — it imports
   reversal's turnover. You must combine *net-of-cost* contributions, not raw
   signals.
3. **Cost-aware (inverse-turnover) weighting fixes the net P&L** (+0.91) — but
   only by shrinking reversal's allocation to ~zero. The composite collapses back
   into essentially lowvol-alone (+0.91 vs +0.90). We dodged reversal's cost by
   not trading it, so we never captured its uncorrelated alpha.
4. **Still below the bar: DSR 0.016.** Two factors weren't enough — and the
   second one (reversal) is *fundamentally untradeable net* at 1h regardless of
   weighting, because its alpha lives below the transaction-cost floor.

### Honest endpoint of the order-flow / cross-sectional research arc

After leak-free, peek-free, cost-aware, multiple-testing-corrected validation
across **six signal families** and a combination attempt, the standing result is:

> **One real-but-weak factor (cross-sectional low-vol, OOS Sharpe ~0.9) that does
> not individually clear the deflated-Sharpe bar, and no second *tradeable*
> low-turnover factor available at 1h to supply the breadth that would.**

This is a legitimate, common place for a research program to land: a genuine
sub-threshold edge, with no cheap way to reach significance from the data on
hand. Harvesting more requires *new information*, not more reprocessing of 1h
OHLCV/funding:
  * **Finer-frequency data (1–5m tick/L2)** — order-flow and reversal ICs are far
    larger intraday, where they're not yet cost-killed. Needs real market-data
    infra.
  * **Different signal families** — on-chain flows, basis/term-structure,
    cross-exchange, sentiment. Genuinely uncorrelated to price/vol.
  * **More breadth** — a larger, longer universe (more coins, more history) raises
    √breadth and tightens every t-stat directly.

`lowvol` is worth keeping as one sleeve of a future book; it is not, on its own,
a strategy. That is the honest, cost-and-significance-aware conclusion.

## The house strategy: `findings_xs` (`autotrader findings`)

Synthesis of everything above into one coherent book (`autotrader/portfolio.py`).
Every design choice is a finding: cross-sectional dollar-neutral (beta-free) ·
low-vol core + slow-reversal tilt (the real factors) · inverse-turnover factor
weighting (cost-aware) · no-trade band + 12h rebalance (turnover was the binding
constraint everywhere).

Building it *properly* — with turnover control — gave the best numbers in the
project. The cost-aware rule auto-allocated **90% lowvol / 10% reversal** from the
data, and turnover collapsed to **58×/yr** (vs 180–1027× for the raw factors).

| | In-sample | Out-of-sample (peek-free) |
|---|---|---|
| Return | +18.6% | +13.1% |
| Sharpe | +0.87 | **+0.97** |
| Turnover | 58×/yr | — |
| OOS IC t | — | **+2.96** |
| Deflated Sharpe | — | **DSR 0.39** |

**DSR 0.39 is the best deflation result here** (up from 0.02–0.03) because low
turnover lifted the real Sharpe relative to the haircut — but it is still below
the 0.95 bar. Verdict: `findings_xs` is a **real, market-neutral, low-turnover
sleeve (OOS Sharpe ~1.0, significant OOS IC)** — the best evidence-backed book we
have, not a *proven* standalone strategy. Run `autotrader findings` to reproduce.

### Adding breadth factors (`autotrader findings --expanded`)

Per the fundamental law (skill ≈ IC × √breadth), adding uncorrelated factors
should lift the combined Sharpe. Added three economically-grounded, low-turnover
candidates: lottery/MAX, betting-against-beta, long-term reversal.

| | 2-factor | 5-factor (expanded) |
|---|---|---|
| OOS Sharpe | +0.97 | **+1.26** |
| OOS IC t | 2.96 | 3.13 |
| Deflated Sharpe | 0.39 | **0.57** |

**Breadth genuinely helped** — OOS Sharpe and DSR both rose, confirming the
theory. But it still does not clear the 0.95 bar, and the honest reasons:
  * The strong additions (lowbeta corr +0.94, lottery +0.73 to lowvol) are
    largely **re-expressing the low-vol anomaly**, not adding independent breadth.
  * The genuinely uncorrelated ones (longrev −0.14, reversal −0.07) are
    individually weak (longrev IC t=1.12) or cost-killed (reversal).
  * DSR 0.57 is mildly **optimistic** — the true project-wide trial count exceeds
    the n_trials used.

**Diminishing returns on this data.** We are re-expressing one real anomaly
(low-vol) several ways. Clearing the bar needs *new information* — finer bars
(1–5m) or genuinely different families (on-chain, basis, sentiment) — not more
reprocessing of 1h OHLCV. `findings_xs(expanded)` is the best honest book: a
market-neutral, low-turnover sleeve, OOS Sharpe ~1.3, sub-threshold on DSR.

It is now tradeable live on paper, market-neutral across the universe, via
`autotrader paper-xs` (visible on the dashboard's "findings_xs" panel).

## Funding carry & statistical arbitrage — the deep dive

Two more genuinely-distinct mechanisms, tested to be exhaustive:

**Funding carry** (`autotrader carry`) — market-neutral book that *collects* the
funding spread (short high-funding, long low-funding perps). The carry is real
(+4.6% collected over the sample) but small, and harvesting it is dominated by
either turnover cost (516× → 52% drag) or, at low turnover, the price/beta noise
of the legs (price P&L ±29% swamps the carry). OOS Sharpe up to ~1.5 but
**DSR 0.29** — not clean carry, and sub-threshold.

**Statistical arbitrage** (`autotrader statarb`) — cointegration pairs,
relative-value mean reversion. **This is the definitive illustration of why money
printers are illusions:**

| | In-sample | Out-of-sample |
|---|---|---|
| Book equity | **1.80×** (smooth, rising) | — |
| Per-pair Sharpe | +2.4 to +3.9 | — |
| OOS Sharpe | — | **+0.54** |
| Deflated Sharpe | — | **0.42** |

The in-sample book *looks exactly like a money printer* — 1.8×, a beautiful
rising equity curve, Sharpes near 4. Out-of-sample it collapses to Sharpe 0.54,
DSR 0.42. The pairs that mean-reverted in the past do not reliably keep doing so;
the cointegration breaks down. **Every "money printer" backtest looks like that
in-sample row. The OOS row is the truth.**

## Liquidity screening — a textbook self-deception trap (`autotrader findings --top N`)

Hypothesis: dropping the "junk" coins should lift `findings_xs`. Tested with an
ex-ante dollar-volume liquidity screen (no return-snooping):

| Universe | OOS Sharpe | DSR |
|---|---|---|
| Full 12 | 0.97 | 0.38 |
| Top-8 | **1.47** | **0.52** |
| Top-6 | 0.62 | 0.28 |

Top-8 looks like a big win — until you see it's **non-monotonic**: top-6 *hurts*,
top-8 *helps*, full sits in between. A real effect trends with universe size;
this zig-zags, so the top-8 "win" is **noise from a freshly-added knob**, and
selecting top-8 on its OOS DSR is the exact multiple-testing trap the deflated
Sharpe guards against (and the reported DSR doesn't even count the universe-size
search, so it's overstated).

Two concrete failures of the screen itself:
* It **keeps DOGE** — the actual edge-killer (negative 5m IC) is *high* volume, so
  a liquidity screen can't catch it.
* It **drops good contributors** (e.g. LINK, a real 5m edge), cutting breadth —
  and breadth is exactly what a cross-sectional book lives on (IC × √breadth).

First read (premature): "top-8 is a fishing artifact." Then we did the rigorous
test — and it **refuted that call.**

### Nested universe-size selection (`autotrader findings --nest-universe`)

Choose N on each *training* window only (by train Sharpe), apply to the disjoint
test window — N never sees the test set.

* Train picks **N=8 in 6 of 7 folds** (one picks 10) — a *stable* preference, not
  random flipping.
* Leak-free OOS: **Sharpe +1.48** (vs 0.97 full), return +22%, IC t=3.21,
  **DSR 0.414** (vs 0.378 full).

**The lift carries over out-of-sample.** I was wrong to call top-8 noise — the
non-monotonicity (top-6 hurts, top-8 helps) isn't randomness, it's a genuine
*interior optimum*: too few coins loses breadth, too many dilutes with noisier
names, ~8 is the sweet spot — and, crucially, it is **discoverable on training
data alone.** Trimming to a train-selected ~8 is a real, leak-free improvement.

It is still **sub-threshold** (DSR 0.41 < 0.95) — a better sleeve, not a printer.
The lesson cuts both ways: skepticism about an added knob was the right instinct,
*and* the honest move was to test it properly rather than dismiss it — and the
test corrected me. The improvement is real; it just isn't enough.

## Production hardening (the bot, not the edge)

The system was hardened for real use — safety rails first, live execution last
and heavily gated:
* **RiskManager** (`risk.py`): per-position cap, gross-exposure cap, a sticky
  drawdown **kill-switch**, and a daily-loss limit. Gates every order; state
  persists so a restart can't silently un-halt.
* **Risk-gated paper book** — limits + kill-switch apply even on paper.
* **LiveBroker** (`live/live_broker.py`): **dry-run by default**, testnet support,
  risk-gated, needs explicit keys. `autotrader live` requires `--arm` AND
  `--confirm-live` AND keys to touch real money; refuses otherwise.
* **`preflight`** (connectivity/config/keys) and **`risk --resume`** (re-arm after
  review).

Note this is orthogonal to edge: hardening makes the bot **safe to run**, not
**profitable**. `findings_xs` remains a modest, sub-threshold sleeve — so "real
use" means dry-run → testnet → tiny size, never skipping a step.

## The money-printer conclusion

Seven distinct mechanisms — time-series microstructure, funding prediction,
cross-sectional factors, multi-factor combination, funding carry, statistical
arbitrage — all tested leak-free, peek-free, cost-aware, multiple-testing-
corrected. **None is a money printer.** The best is a modest market-neutral
low-vol sleeve (OOS Sharpe ~1.3, still sub-threshold on DSR).

This is not pessimism — it is the *point*. A search continued on fixed data until
it "finds a printer" is a process that **guarantees a false positive**: the more
configurations you try, the more certain you are to surface a leak that prints
in-sample and dies live (see the stat-arb row). The discipline that protected us
the whole way would have to be switched off to declare a printer — and then it
wouldn't be one.

Real printers exist, but they come from **capabilities, not more code on this
data**: latency/co-location (HFT, cross-exchange arb), execution at tick/L2 with
real fill modeling, capital/access edges (market-making rebates, OTC, listings),
or genuinely novel alt-data. Those are "acquire a new capability," not "iterate
on 1h OHLCV." What this project delivers instead is rarer and more useful: a
harness that tells you the truth, and the honest knowledge of what is real.

## Going finer: 1–5m bars + live L2 capture — we found the signal, and the wall

The one avenue with a real prior: finer bars, where microstructure isn't yet
cost-killed. Two pieces, both built:

* **Finer-bar klines** (`autotrader micro`) — 1–5m extended klines are free and
  carry aggressor flow. Backtestable now.
* **Live L2 recorder** (`autotrader record`) — true tick/L2 has no free history,
  so we capture it via Binance websocket (book imbalance, microprice, spread,
  signed trade flow) going forward. Verified: 235 real snapshots in 25s.

**The finer-bar hypothesis was correct.** At 5m the order-flow signal that was
*noise* at 1h becomes strongly significant — and contrarian (aggressive flow gets
faded):

| Feature | IC@1 (1h) | IC@1 (5m) |
|---|---|---|
| order-flow imbalance | noise | **t = −4.5** |
| vwap deviation | noise | **t = −7.0** |

And the right-sign strategy (order-flow *reversion*) at 5m:

| | Gross | Net (taker) |
|---|---|---|
| Return | **+23.0%** | −100% |
| Sharpe | **+3.47** | −109 |

**There's the "money printer" — gross.** Sharpe 3.5 at 5m. But the per-trade edge
is **0.28 bps** against an **11 bps** taker+slippage cost — the edge is ~40× too
small to pay the toll. As a taker you are wiped out.

### The honest wall

The 5m edge is real but lives entirely below the taker-cost floor. Capturing it
means **becoming a market maker** — earning the spread/rebate instead of paying
it — which requires:
  * real-time L2 book + **queue position** modeling (what `autotrader record`
    starts capturing, and what klines can *never* provide),
  * sub-second **latency / colocation**,
  * a **fill simulator**, then real low-latency execution.

That is an *infrastructure and capability* problem, not a backtest. The signal is
found; the moat is execution — and that moat is held by firms (Jump, Wintermute,
…) who do exactly this. This is the empirical proof of the earlier claim: at fine
frequency the edge is real, and it is a market-making business, not a strategy
you can backtest on candles. The recorder is step one of that long road.

- ❌ **Order-flow / microstructure composites at 1h** — ruled out. Real IC,
  no tradeable net edge across instruments.
- ❌ **Funding fade / funding momentum at 1h** — ruled out. Near break-even, not
  significant, doesn't generalize across instruments in either direction.
- ⚠️ **Cross-sectional low-volatility factor** — OOS-stable (walk-forward IC
  t=2.63, Sharpe +0.61) and economically grounded, but fails the deflated Sharpe
  (DSR 0.03): real but too weak to deploy alone. A **component**, not a standalone
  edge — feed it into a multi-factor book.
- ➡️ **Cross-sectional reversal** — real IC, but needs turnover control (longer
  hold / wider rebalance / cost-aware weighting) to become net-positive.
- ➡️ **Anything new must clear the `panel` bar**: leak-free, peek-free, net of
  costs, positive across held-out instruments with t > 2. That is the standard
  now, and it is a high one (by design).
- ➡️ **Cross-sectional** — rank a signal across many coins, long top / short
  bottom, market-neutral. Removes the beta that dominates single-symbol noise.
  The most structurally different option left, and untested.
- ➡️ **Finer bars (1–5m)** where order-flow IC is far higher — needs a tick/L2
  feed to backtest honestly.

> The bar to clear is now explicit and enforced in code (`autotrader panel`).
> Most candidate "edges" will not clear it. That is the point.

## The live order-book gate

`live/orderbook.py` computes top-of-book imbalance, spread, and microprice from
the live snapshot. In the paper loop it **vetoes entries the book disagrees
with** (long entry needs bids ≥ asks; wide spread skips entirely). This improves
fills using information we genuinely only have live — and it's labelled as such,
never backtested.

## Next steps toward real edge

1. **Purged walk-forward feature weighting** — remove the residual in-sample
   peek in `edge_composite`.
2. **Funding-rate fade on perps** — fetch funding history, fade crowded
   positioning (a well-documented crypto edge).
3. **Cross-sectional** — rank order-flow across many symbols, long the top /
   short the bottom (market-neutral; removes beta).
4. **Finer bars** — order flow is stronger at 1–5m; needs a tick/L2 data
   subscription to backtest properly.

## Refuted ideas — October 2026 (don't re-test these by accident)

All run with a protocol fixed before looking: parameters chosen on a DEV slice
only, verdict on a never-seen TEST slice, held-out instruments where possible.

| Idea | Script | DEV | TEST (verdict) | Why it fails |
|---|---|---|---|---|
| 5m cascade fade (k·σ move + volume spike → fade) | `scalp_scan.py` H1 | +2.9 bps gross | −11 bps (BTC/ETH), −3 bps (6 held-out) | no edge even at zero fees |
| 5m maker dip-buy (limit below VWAP-z<−2, trade-through fills) | `scalp_scan.py` H2 | +10 bps gross | +0.8 bps / −0.2 bps held-out (n=244) | DEV-period artefact |
| Time-of-day 30-min slots | `scalp_scan.py` H3 | t = 6–8 | t ≈ 1.3, ~1 bp | classic overfit |
| Perp/spot basis reversion (5m) | `scalp_scan.py` H4 | +1.7 bps, t = 6 | +1–3.6 bps, t = 1.7–3 | **real**, but 10× below the 19 bps round-trip — market-maker only |
| Short new perp listings, BTC-hedged, fees + funding | `listing_short.py` | mean +20%, t = 6.6 | mean +0.6%, t = 0.06 | median trade wins (+25%, 70%) but squeezes kill the mean; stop-loss made it worse; 2025-26 listings squeeze harder |
| Funding-crowding overlay on the core (halve when 30d funding > 15%/yr) | inline (see SUMMARY) | Sharpe 1.44 vs 1.40, CAGR 27% vs 39% | Sharpe 1.29 vs 1.23, DD −20% vs −23% | within noise after 12 configs — not adopted |
| Pick the best core config each year | `config_tournament.py` | in-sample winner Sharpe 1.29 | walk-forward 1.05 vs fixed 1.17; DSR 0.00 | selection hurts → core uses a blend of 4 trend speeds instead |
| Day trading — opening-range breakout (first R h, stop) | `daytrade_scan.py` D1 | +10 bps (BTC/ETH), −4 bps held-out | −7.7 bps, day-t −1.0 | costs eat it; doesn't transfer to alts |
| Day trading — intraday momentum (00→16h move > 1σ → ride to close) | `daytrade_scan.py` D2 | +32 bps, book Sharpe 0.7 | +31 bps but day-t 1.6, median −10 bps | **looked like a PASS** with per-trade t 4.25 — 12 coins on the same day are one bet; isolated cell (T12/T20 negative), mostly long-side beta |
| Day trading — fade/follow a >2.5σ day the next day | `daytrade_scan.py` D3 | +83 bps (n=96) | −14 bps | small-n DEV mirage |
| Day trading — session drift (Asia/EU/US/all-day) | `daytrade_scan.py` D4 | all-day long +9 bps | −10 bps | that's just beta minus 12–16 bps/day of fees |
| Day trading — volatility breakout ("Korean VBO", Larry Williams) | `daytrade_scan.py` D5 | +65 bps, t 3.6 | +33 bps, day-t −0.3 | long-only + trend filter = trend-following beta; core already owns that |

Data: `fetch_5m.py` (1y of 5m perp+spot klines, 8 coins); `fetch_1h.py` (1h perp
klines + funding 2020→2026, 12 coins; day trading DEV < 2024-07 ≤ TEST); `listing_short.py --fetch`
(515 USDT perps listed 2023→2026-07 incl. 102 delisted).

**Methodology lesson (day-trading scan).** Pool trades across coins and the t-stat
lies: same-day trades on correlated coins are one bet. The verdict must use the
day-clustered t (or the book's daily t). `daytrade_scan.py` does; the per-trade t in
`scalp_scan.py` / `listing_short.py` is optimistic for the same reason (both failed
anyway).

**Takeaway.** Measurable 5m regularities are worth 1–4 bps; retail pays 4–19 bps
per round trip. Day-trading playbooks (1h, flat overnight) pay 12–16 bps a day in fees
and none survives. Event effects (listings) are real on the median and broken by fat
tails. The core book (README) remains the strategy for real capital.
