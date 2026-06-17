# Going live safely — testnet guide

This walks you from the dry-run bot to live execution **without risking a cent**,
using the Binance **Futures** testnet (fake money, real API). Follow it in order.
Never skip a step.

> **Why futures, not spot?** `findings_xs` is a *market-neutral, long/short* book —
> it shorts the rich coins and longs the cheap ones. **You cannot short on spot.**
> So live execution defaults to perpetual futures (`binanceusdm`). `--spot` exists
> but only does the long legs (not the real strategy).

> ⚠️ Live trading is real risk. The bot has a kill-switch and risk limits, but
> those bound losses, they don't prevent them. The edge is modest and unproven
> (`findings_xs` ~Sharpe 1.5, sub-significance). Treat live as *validation of
> execution*, not as a profit engine. Use tiny size for a long time.

## Step 1 — Get Futures testnet API keys (free, fake money)

1. Go to **https://testnet.binancefuture.com** and log in (GitHub/Google).
2. Bottom of the page → **API Key** panel → copy the **API Key** and **API Secret**.
3. The testnet funds your account with fake USDT automatically.

(Spot-only testing instead? Use https://testnet.binance.vision and run with `--spot`.)

## Step 2 — Set the keys in your environment

```bash
export BINANCE_API_KEY="your_testnet_key"
export BINANCE_SECRET="your_testnet_secret"
```

(Use `! export ...` inside this session if you want it in the conversation.)

## Step 3 — Pre-flight

```bash
autotrader preflight --live
```
You want: exchange reachable ✓, risk limits set ✓, API keys present ✓.

## Step 4 — Dry-run first (no keys needed, sends nothing)

```bash
autotrader live --config config_quick.yaml
```
Confirms the bot computes a sensible long/short book and the risk gate behaves.
Status should read `simulated` for every order.

## Step 5 — One live order on testnet

```bash
autotrader live --config config_quick.yaml --testnet --arm
```
`--testnet --arm` places **real orders on the sandbox** (fake money). Check the
status column reads `sent`, then verify the positions on testnet.binancefuture.com.

## Step 6 — Continuous testnet loop

```bash
autotrader live --config config_quick.yaml --testnet --arm --loop
```
Rebalances every `poll_seconds` (config). Let it run for days; watch:
* the kill-switch (`autotrader risk`) — if a −20% drawdown trips it, the loop
  halts and flattens; review, then `autotrader risk --resume`.
* the dashboard (`autotrader dashboard`) for the book.

## Step 7 — Tiny real money (only after testnet looks right)

```bash
# real keys from binance.com (NOT testnet), tiny balance, then:
autotrader live --config config_quick.yaml --arm --confirm-live --equity 100 --loop
```
`--confirm-live` is the extra lock required for real money. Start with the
smallest size the exchange allows and watch it like a hawk. Scale only if live
fills match the backtest assumptions.

## Safety controls you have

| Control | What it does |
|---|---|
| dry-run (default) | sends nothing; you must `--arm` to trade |
| `--testnet` | sandbox, fake money |
| `--confirm-live` | extra lock required for REAL money |
| risk limits (`config.yaml` → `risk:`) | per-position 30%, gross 100%, daily −8% |
| kill-switch | −20% from peak → halt + flatten, sticky until `risk --resume` |
| `autotrader preflight --live` | refuses if keys/config missing |

## What's still manual / honest gaps

* **Monitoring/alerting**: the loop prints + halts, but there's no push alert yet.
  Watch it, or wire a notifier.
* **Position reconciliation**: it rebalances to target each tick but doesn't yet
  diff intended vs actual fills — check the exchange UI.
* **Order types**: market orders only (pays the taker spread). Maker/limit orders
  would cut cost but need fill-handling logic.
* **The edge is modest**: hardening makes it *safe to run*, not *profitable*.
