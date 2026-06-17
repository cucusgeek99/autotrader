"""autotrader command line.

    autotrader lab            # tune every strategy, record what works
    autotrader paper BTC/USDT # paper-trade the champion live
    autotrader leaderboard    # show best experiments learned so far
    autotrader backtest ma_crossover BTC/USDT --fast 10 --slow 50
    autotrader strategies     # list the strategy library
"""
from __future__ import annotations

import json

import typer
from rich.console import Console
from rich.table import Table

from autotrader.config import Config

app = typer.Typer(add_completion=False, help="Self-tuning crypto trading lab.")
console = Console()


def _cfg(path: str) -> Config:
    return Config.load(path)


@app.command()
def lab(
    config: str = typer.Option("config.yaml", help="config file"),
    symbol: list[str] = typer.Option(None, help="override symbols (repeatable)"),
):
    """Tune every strategy walk-forward and store champions in the knowledge base."""
    from autotrader.lab import run_lab

    cfg = _cfg(config)
    run_lab(cfg, symbols=symbol or None)


@app.command()
def paper(
    symbol: str = typer.Argument(..., help="e.g. BTC/USDT"),
    config: str = typer.Option("config.yaml"),
    once: bool = typer.Option(False, help="run a single tick and exit"),
    ensemble: int = typer.Option(0, help="blend top-K champions instead of just the best"),
):
    """Paper-trade the current champion (or an ensemble) for SYMBOL on live data."""
    from autotrader.live.trader import run_live

    run_live(_cfg(config), symbol, once=once, ensemble_k=ensemble)


@app.command()
def report(
    strategy: str = typer.Argument(...),
    symbol: str = typer.Argument(...),
    config: str = typer.Option("config.yaml"),
    param: list[str] = typer.Option(None, help="key=value override (repeatable)"),
    csv: str = typer.Option(None, help="export equity curve to this CSV path"),
):
    """Full backtest report with equity sparkline + drawdown for one strategy."""
    from autotrader.backtest.engine import run_backtest
    from autotrader.backtest.report import export_csv, render
    from autotrader.data.feed import fetch_ohlcv
    from autotrader.strategies.registry import get

    cfg = _cfg(config)
    params = {}
    for kv in param or []:
        k, v = kv.split("=", 1)
        params[k] = _coerce(v)
    df = fetch_ohlcv(symbol, cfg["timeframe"], cfg["history_limit"],
                     cfg["exchange"], cfg["data_cache_dir"])
    strat = get(strategy)(**params)
    res = run_backtest(strat, df, cfg["fees"]["taker"], cfg["fees"]["slippage"],
                       cfg.bars_per_year)
    render(res, f"{strat} on {symbol} ({len(df)} bars)", console)
    if csv:
        console.print(f"[dim]equity curve → {export_csv(res, csv)}[/]")


@app.command(name="paper-xs")
def paper_xs(
    config: str = typer.Option("config.yaml"),
    symbol: list[str] = typer.Option(None, help="universe (repeatable); default basket"),
    once: bool = typer.Option(False, help="run a single rebalance and exit"),
    simple: bool = typer.Option(False, help="use the 2-factor book instead of expanded"),
):
    """Paper-trade the house cross-sectional book (findings_xs) live, market-neutral."""
    from autotrader.live.xs_trader import run_live_xs

    run_live_xs(_cfg(config), universe=symbol or None, once=once, expanded=not simple)


@app.command()
def leaderboard(config: str = typer.Option("config.yaml"), limit: int = 20):
    """Show the best experiments learned across all lab runs."""
    from autotrader.knowledge.store import KnowledgeStore

    cfg = _cfg(config)
    store = KnowledgeStore(cfg["knowledge_db"])
    rows = store.leaderboard(limit)
    if not rows:
        console.print("[yellow]No experiments yet. Run `autotrader lab` first.[/]")
        return
    table = Table(title="Knowledge leaderboard", header_style="bold")
    for col in ("symbol", "tf", "regime", "strategy", "OOS", "in-sample", "gap", "params"):
        table.add_column(col)
    for r in rows:
        table.add_row(
            r["symbol"], r["timeframe"], r["regime"], r["strategy"],
            f"{r['oos_score']:+.3f}", f"{r['insample_score']:+.3f}",
            f"{r['overfit_gap']:+.3f}", json.dumps(r["params"]),
        )
    console.print(table)
    store.close()


@app.command()
def backtest(
    strategy: str = typer.Argument(...),
    symbol: str = typer.Argument(...),
    config: str = typer.Option("config.yaml"),
    param: list[str] = typer.Option(None, help="key=value override (repeatable)"),
):
    """Backtest one strategy with explicit params on cached/fresh data."""
    from autotrader.backtest.engine import run_backtest
    from autotrader.data.feed import fetch_ohlcv
    from autotrader.strategies.registry import get

    cfg = _cfg(config)
    params = {}
    for kv in param or []:
        k, v = kv.split("=", 1)
        params[k] = _coerce(v)

    df = fetch_ohlcv(symbol, cfg["timeframe"], cfg["history_limit"],
                     cfg["exchange"], cfg["data_cache_dir"])
    strat = get(strategy)(**params)
    res = run_backtest(strat, df, cfg["fees"]["taker"], cfg["fees"]["slippage"],
                       cfg.bars_per_year)
    console.print(f"[bold]{strat}[/] on {symbol} ({len(df)} bars)")
    for k, v in res.metrics.as_dict().items():
        console.print(f"  {k:14} {v}")


@app.command()
def advise(config: str = typer.Option("config.yaml"), symbol: str = typer.Option(None)):
    """Ask Claude (LLM advisor) to analyze the knowledge base and propose experiments."""
    from autotrader.knowledge.llm_advisor import AdvisorUnavailable, advise as run_advise
    from autotrader.knowledge.regime import classify
    from autotrader.knowledge.store import KnowledgeStore
    from autotrader.data.feed import fetch_ohlcv
    from autotrader.strategies.registry import all_strategies

    cfg = _cfg(config)
    sym = symbol or cfg["symbols"][0]
    store = KnowledgeStore(cfg["knowledge_db"])
    df = fetch_ohlcv(sym, cfg["timeframe"], cfg["history_limit"], cfg["exchange"], cfg["data_cache_dir"])
    regime = classify(df)
    leaderboard = [r for r in store.leaderboard(40) if r["symbol"] == sym]
    spaces = {name: cls.param_space for name, cls in all_strategies().items()}

    try:
        advice = run_advise(sym, cfg["timeframe"], regime, leaderboard, spaces)
    except AdvisorUnavailable as e:
        console.print(f"[yellow]LLM advisor unavailable:[/] {e}")
        store.close()
        raise typer.Exit(1)

    console.rule(f"[bold]LLM advice — {sym} ({regime})[/]")
    console.print(f"[bold]Regime read:[/] {advice.regime_read}\n")
    console.print(f"[bold]Summary:[/] {advice.summary}\n")
    console.print("[bold]Hypotheses:[/]")
    for h in advice.hypotheses:
        ranges = ", ".join(f"{p.name}∈[{p.low:g},{p.high:g}]" for p in h.param_ranges)
        console.print(f"  • [{h.conviction}] [cyan]{h.strategy}[/] — {h.rationale}")
        if ranges:
            console.print(f"      ranges: {ranges}")
    if advice.risk_flags:
        console.print("\n[bold red]Risk flags:[/]")
        for r in advice.risk_flags:
            console.print(f"  ! {r}")
    store.close()


@app.command()
def dashboard(
    config: str = typer.Option("config.yaml"),
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8000),
):
    """Launch the web dashboard (equity curve, leaderboard, fills)."""
    try:
        import uvicorn
    except ImportError:
        console.print("[red]Install web deps:[/] pip install 'autotrader[web]'")
        raise typer.Exit(1)
    from autotrader.dashboard.app import create_app

    console.print(f"[green]Dashboard →[/] http://{host}:{port}")
    uvicorn.run(create_app(config), host=host, port=port, log_level="warning")


@app.command()
def schedule(
    config: str = typer.Option("config.yaml"),
    every: int = typer.Option(3600, help="seconds between lab re-tuning runs"),
    symbol: list[str] = typer.Option(None),
):
    """Re-run the lab on an interval so the bot keeps setting itself up."""
    import time
    from datetime import datetime, timezone

    from autotrader.lab import run_lab

    cfg = _cfg(config)
    console.print(f"[green]Scheduler started[/] — re-tuning every {every}s. Ctrl-C to stop.")
    while True:
        console.rule(f"lab run @ {datetime.now(timezone.utc).isoformat()[:19]}")
        try:
            run_lab(cfg, symbols=symbol or None)
        except Exception as exc:
            console.print(f"[red]lab run failed:[/] {exc}")
        time.sleep(every)


@app.command()
def features(
    symbol: str = typer.Argument(...),
    config: str = typer.Option("config.yaml"),
    horizon: int = typer.Option(1, help="forward-return horizon in bars"),
):
    """Measure feature edge: information coefficient vs forward returns."""
    from autotrader.analysis.feature_ic import compute_ic
    from autotrader.data.feed import fetch_ohlcv_extended
    from autotrader.features.microstructure import feature_frame, has_flow

    cfg = _cfg(config)
    df = fetch_ohlcv_extended(symbol, cfg["timeframe"], cfg["history_limit"],
                              cfg["exchange"], cfg["data_cache_dir"])
    if not has_flow(df):
        console.print("[yellow]No order-flow columns for this exchange — price features only.[/]")
    feats = feature_frame(df)
    results = compute_ic(feats, df["close"], horizon=horizon)

    table = Table(title=f"Feature IC — {symbol} ({cfg['timeframe']}, h={horizon}, n={len(df)})",
                  header_style="bold")
    for col in ("feature", "IC", "t-stat", "hit-rate", "signal?"):
        table.add_column(col)
    for r in results:
        # |t|>2 ≈ statistically significant; flag genuine signal
        sig = "[green]yes[/]" if abs(r.t_stat) > 2 else "[dim]noise[/]"
        ic_str = f"[green]{r.ic:+.4f}[/]" if abs(r.t_stat) > 2 else f"{r.ic:+.4f}"
        table.add_row(r.feature, ic_str, f"{r.t_stat:+.2f}", f"{r.hit_rate:.1%}", sig)
    console.print(table)
    console.print("[dim]IC = Spearman corr of feature vs next-h-bar return. "
                  "|t|>2 ≈ unlikely to be noise. Even small stable IC is real edge at scale.[/]")


@app.command()
def purged(
    symbol: str = typer.Argument(...),
    config: str = typer.Option("config.yaml"),
    horizon: int = typer.Option(12, help="forward-return horizon / hold (bars)"),
    lookback: int = typer.Option(24),
    threshold: float = typer.Option(0.5, help="entry threshold in score z-units"),
    embargo: int = typer.Option(12, help="extra purge buffer between train and test"),
    regime: bool = typer.Option(False, "--regime", help="fit weights per regime"),
    nested: bool = typer.Option(False, "--nested", help="tune threshold per fold (nested, regime)"),
):
    """Leak-free read on the order-flow composite via purged walk-forward.

    Fits feature weights per training fold only, purges overlapping labels, and
    embargoes the train/test boundary — so the OOS score has no look-ahead.
    With --regime, fits a separate weight vector per market regime. With
    --nested, the entry threshold is tuned per fold on an inner walk-forward
    (no threshold-selection peek).
    """
    from autotrader.analysis.purged_wf import (
        purged_nested_regime, purged_walk_forward, purged_walk_forward_regime,
    )
    from autotrader.data.feed import fetch_ohlcv_extended

    cfg = _cfg(config)
    df = fetch_ohlcv_extended(symbol, cfg["timeframe"], cfg["history_limit"],
                              cfg["exchange"], cfg["data_cache_dir"])
    wf = cfg["walkforward"]
    if nested:
        out = purged_nested_regime(
            df, horizon=horizon, lookback=lookback,
            train=wf["train"], test=wf["test"], step=wf["step"], embargo=embargo,
            fees=cfg["fees"], bars_per_year=cfg.bars_per_year,
        )
    else:
        runner = purged_walk_forward_regime if regime else purged_walk_forward
        out = runner(
            df, horizon=horizon, lookback=lookback, threshold=threshold,
            train=wf["train"], test=wf["test"], step=wf["step"], embargo=embargo,
            fees=cfg["fees"], bars_per_year=cfg.bars_per_year,
        )

    mode = ("nested regime" if nested else
            "regime-conditional" if regime else "global")
    console.rule(f"[bold]Purged walk-forward ({mode}) — edge_composite on {symbol}[/]")
    console.print(
        f"horizon={horizon}  lookback={lookback}  threshold={threshold}  "
        f"purge+embargo={horizon + embargo} bars\n"
    )
    last_col = "chosen thr (inner scores)" if nested else "weights"
    table = Table(show_header=True, header_style="bold")
    for col in ("fold", "OOS return", "OOS sharpe", "trades", last_col):
        table.add_column(col)
    for f in out.get("folds", []):
        w = f.get("weights", {})
        if nested:
            detail = f"thr={w.get('chosen_threshold')}  inner={w.get('inner', {})}"
        else:
            detail = " ".join(f"{k}={v:+.2f}" for k, v in w.items()
                              if isinstance(v, (int, float)))
        table.add_row(str(f["fold"]), f"{f['return']:+.2%}", f"{f['sharpe']:+.2f}",
                      str(f["trades"]), detail)
    console.print(table)
    if out.get("chosen_thresholds"):
        console.print(f"[bold]thresholds chosen by inner WF:[/] {out['chosen_thresholds']} "
                      "[dim](each picked without seeing its outer test set)[/]")
    console.print(
        f"\n[bold]folds:[/] {out['n_folds']}   "
        f"[bold]mean OOS return:[/] {out['mean_oos_return']:+.2%}   "
        f"[bold]mean sharpe:[/] {out['mean_oos_sharpe']:+.2f}   "
        f"[bold]positive:[/] {out['folds_positive']}   "
        f"[bold]return t-stat:[/] {out['return_t_stat']}"
    )
    color = "green" if "genuine" in out["verdict"] else (
        "yellow" if "promising" in out["verdict"] else "red")
    console.print(f"\n[bold {color}]VERDICT: {out['verdict']}[/]")

    if out.get("regime_stability"):
        console.print("\n[bold]Regime weight stability[/] (is the signal stable WITHIN a regime?):")
        for r, s in out["regime_stability"].items():
            flag = "[green]stable[/]" if s["strongest_sign_stable"] else "[red]flips[/]"
            console.print(
                f"  {r:18} {s['n_folds']} folds  strongest={s['strongest_feature']} "
                f"({s['features'][s['strongest_feature']]['mean']:+.2f}) → {flag}"
            )


@app.command()
def panel(
    config: str = typer.Option("config.yaml"),
    symbol: list[str] = typer.Option(
        None, help="held-out symbols (repeatable); default basket if omitted"),
    dev: str = typer.Option("BTC/USDT", help="dev symbol (excluded from headline)"),
    history: int = typer.Option(8000, help="bars per symbol (more = more folds)"),
    horizon: int = typer.Option(12),
    lookback: int = typer.Option(24),
    train: int = typer.Option(1500),
    test: int = typer.Option(400),
    step: int = typer.Option(300),
    funding: bool = typer.Option(False, "--funding", help="test the funding fade instead"),
    momentum: bool = typer.Option(False, "--momentum", help="funding momentum (follow) not fade"),
):
    """Cross-symbol validation: run the nested pipeline on instruments NOT used
    to design the strategy, then pool folds for an honest significance test."""
    from autotrader.analysis.panel import panel_validation
    from autotrader.data.feed import fetch_ohlcv_extended

    cfg = _cfg(config)
    held = symbol or ["ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", "ADA/USDT"]
    all_syms = [dev] + [s for s in held if s != dev]

    if funding:
        from autotrader.analysis.funding_fade import purged_fade
        from autotrader.data.derivatives import fetch_perp_with_funding
        dfs = {}
        for s in all_syms:
            console.print(f"[dim]fetching {s} perp + funding ({history} bars)...[/]")
            dfs[s] = fetch_perp_with_funding(s, cfg["timeframe"], history,
                                             cache_dir=cfg["data_cache_dir"])
        out = panel_validation(
            dfs, dev_symbols={dev}, runner=purged_fade,
            horizon=max(horizon, 24), train=train, test=test, step=step,
            direction=(1 if momentum else -1),
            fees=cfg["fees"], bars_per_year=cfg.bars_per_year,
        )
    else:
        dfs = {}
        for s in all_syms:
            console.print(f"[dim]fetching {s} ({history} bars)...[/]")
            dfs[s] = fetch_ohlcv_extended(s, cfg["timeframe"], history,
                                          cfg["exchange"], cfg["data_cache_dir"])
        out = panel_validation(
            dfs, dev_symbols={dev}, horizon=horizon, lookback=lookback,
            train=train, test=test, step=step,
            fees=cfg["fees"], bars_per_year=cfg.bars_per_year,
        )

    console.rule("[bold]Panel validation — nested order-flow strategy[/]")
    table = Table(show_header=True, header_style="bold")
    for col in ("symbol", "role", "folds", "mean OOS", "sharpe", "t-stat", "pos", "thresholds"):
        table.add_column(col)
    for s in all_syms:
        ps = out["per_symbol"][s]
        st = ps["stats"]
        role = "[dim]dev[/]" if s == dev else "held-out"
        thr = "".join(str(t).replace(".0", "") for t in ps["chosen_thresholds"])
        table.add_row(
            s, role, str(ps["n_folds"]), f"{st['mean']:+.2%}",
            f"{ps['mean_sharpe']:+.2f}", f"{st['t_stat']:+.2f}", st["pos"], thr,
        )
    console.print(table)

    p, pool = out["portfolio"], out["pooled"]
    console.print(
        f"\n[bold]Held-out:[/] {out['held_out']}   "
        f"[bold]symbols net-positive:[/] {out['symbols_positive']}"
    )
    console.print(
        f"[bold]Pooled[/] (optimistic, n={pool['n']}): "
        f"mean {pool['mean']:+.2%}, t={pool['t_stat']:+.2f}, {pool['pos']}"
    )
    console.print(
        f"[bold]Portfolio[/] (honest, cross-corr absorbed, n={p['n']}): "
        f"mean {p['mean']:+.2%}, [bold]t={p['t_stat']:+.2f}[/], {p['pos']}"
    )
    color = ("green" if "SIGNIFICANT" in out["verdict"] else
             "yellow" if "promising" in out["verdict"] else "red")
    console.print(f"\n[bold {color}]VERDICT: {out['verdict']}[/]")


@app.command()
def funding(
    symbol: str = typer.Argument("BTC/USDT"),
    config: str = typer.Option("config.yaml"),
    horizon: int = typer.Option(24, help="forward-return horizon / hold (bars)"),
    lookback: int = typer.Option(168, help="funding z-score window (bars)"),
    history: int = typer.Option(8000),
):
    """Funding-rate fade: IC of funding z-score + nested purged read (carry vs price-only)."""
    from autotrader.analysis.feature_ic import compute_ic
    from autotrader.analysis.funding_fade import funding_zscore, purged_fade
    from autotrader.data.derivatives import fetch_perp_with_funding
    import pandas as pd

    cfg = _cfg(config)
    df = fetch_perp_with_funding(symbol, cfg["timeframe"], history,
                                 cache_dir=cfg["data_cache_dir"])
    if (df["funding"] == 0).all():
        console.print("[red]No funding data — is this a perp symbol on binanceusdm?[/]")
        raise typer.Exit(1)

    fr = float(df["funding"].mean())
    console.print(f"[dim]{len(df)} bars, mean funding {fr:+.4%}/8h "
                  f"(annualized ~{fr*3*365:+.1%})[/]\n")

    # IC: does funding z-score predict forward returns? (fade = NEGATIVE IC is the edge)
    feats = pd.DataFrame({"funding_z": funding_zscore(df, lookback)}, index=df.index)
    for h in (8, 24, 48):
        ic = compute_ic(feats, df["close"], horizon=h)[0]
        sig = "significant" if abs(ic.t_stat) > 2 else "noise"
        console.print(f"  funding_z vs +{h}b return: IC {ic.ic:+.4f}  t={ic.t_stat:+.2f}  ({sig})")

    wf = cfg["walkforward"]
    out = purged_fade(
        df, horizon=horizon, lookback=lookback,
        train=wf["train"], test=wf["test"], step=wf["step"],
        fees=cfg["fees"], bars_per_year=cfg.bars_per_year,
    )
    console.rule(f"[bold]Funding fade — nested purged WF — {symbol}[/]")
    console.print(
        f"folds: {out['n_folds']}   thresholds chosen: {out.get('chosen_thresholds')}\n"
        f"[bold]with carry[/] (realistic): mean OOS {out['mean_oos_return']:+.2%}, "
        f"Sharpe {out['mean_oos_sharpe']:+.2f}, {out['folds_positive']}, "
        f"t={out['return_t_stat']}\n"
        f"[bold]price-only[/] (conservative): mean OOS {out['mean_price_only_return']:+.2%}"
    )
    color = ("green" if "genuine" in out["verdict"] else
             "yellow" if "promising" in out["verdict"] else "red")
    console.print(f"\n[bold {color}]VERDICT (single symbol): {out['verdict']}[/]")
    console.print("[dim]Then run `autotrader panel --funding` for the cross-symbol test.[/]")


@app.command()
def xsection(
    config: str = typer.Option("config.yaml"),
    symbol: list[str] = typer.Option(None, help="universe (repeatable); default basket"),
    history: int = typer.Option(8000),
    horizon: int = typer.Option(12, help="forward-return horizon / rebalance (bars)"),
    funding: bool = typer.Option(False, "--funding", help="include perp funding factor"),
    oos: bool = typer.Option(False, "--oos", help="walk-forward out-of-sample + deflated Sharpe"),
    combo: bool = typer.Option(False, "--combo", help="combine factors into a multi-factor book"),
):
    """Cross-sectional factor research: rank IC + dollar-neutral long-short
    backtest (net of costs) for each factor across a coin universe."""
    from autotrader.analysis import cross_sectional as xs
    from autotrader.data.feed import fetch_ohlcv_extended

    cfg = _cfg(config)
    universe = symbol or [
        "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", "ADA/USDT",
        "DOGE/USDT", "AVAX/USDT", "LINK/USDT", "LTC/USDT", "DOT/USDT", "TRX/USDT",
    ]
    dfs = {}
    for s in universe:
        console.print(f"[dim]fetching {s}...[/]")
        try:
            if funding:
                from autotrader.data.derivatives import fetch_perp_with_funding
                dfs[s] = fetch_perp_with_funding(s, cfg["timeframe"], history,
                                                 cache_dir=cfg["data_cache_dir"])
            else:
                dfs[s] = fetch_ohlcv_extended(s, cfg["timeframe"], history,
                                              cfg["exchange"], cfg["data_cache_dir"])
        except Exception as e:
            console.print(f"[yellow]skip {s}: {e}[/]")

    panels = xs.build_panels(dfs)
    closes, flow, fund = panels["closes"], panels["flow"], panels["funding"]
    console.rule(f"[bold]Cross-sectional factors — {len(closes.columns)} coins, "
                 f"{len(closes)} bars, h={horizon}[/]")

    if oos:
        from autotrader.analysis.cross_validation import (
            _block_returns, deflated_sharpe_ratio, walk_forward_xs,
        )
        builders = {
            "momentum": (lambda lb: xs.sig_momentum(closes, lb), [12, 24, 48, 96]),
            "reversal": (lambda lb: xs.sig_reversal(closes, lb), [3, 6, 12, 24]),
            "lowvol": (lambda lb: xs.sig_lowvol(closes, lb), [24, 48, 96, 168]),
        }
        if not flow.empty:
            builders["orderflow"] = (lambda lb: xs.sig_flow(flow, lb), [6, 12, 24])
        if not fund.empty:
            builders["funding_xs"] = (lambda lb: xs.sig_funding(fund, lb), [72, 168, 336])

        results, block_rets, trial_sharpes = {}, {}, []
        for name, (fn, grid) in builders.items():
            wf = walk_forward_xs(fn, closes, horizon=horizon, lookback_grid=grid,
                                 fees=cfg["fees"], bars_per_year=cfg.bars_per_year)
            results[name] = wf
            br = _block_returns(wf.oos_returns, horizon)
            block_rets[name] = br
            if br.std() > 0:
                trial_sharpes.append(float(br.mean() / br.std()))

        table = Table(title="Walk-forward OUT-OF-SAMPLE (nested lookback)", header_style="bold")
        for col in ("factor", "OOS IC", "OOS IC-t", "OOS Sharpe", "OOS return", "folds", "lookbacks"):
            table.add_column(col)
        for name, wf in results.items():
            sig = abs(wf.oos_ic_t) > 2
            ics = f"[green]{wf.oos_ic_mean:+.4f}[/]" if sig else f"{wf.oos_ic_mean:+.4f}"
            table.add_row(name, ics, f"{wf.oos_ic_t:+.2f}", f"{wf.oos_sharpe_ann:+.2f}",
                          f"{wf.oos_total_return:+.2%}", str(wf.n_folds),
                          str(wf.chosen_lookbacks))
        console.print(table)

        valid = {n: w for n, w in results.items() if abs(w.oos_ic_t) > 2 and w.oos_sharpe_ann > 0}
        if not valid:
            console.print("\n[bold red]No factor holds up out-of-sample (IC t>2 + positive OOS Sharpe).[/]")
            return
        best = max(valid, key=lambda n: results[n].oos_sharpe_ann)
        dsr = deflated_sharpe_ratio(block_rets[best], n_trials=len(builders),
                                    trial_sharpes=trial_sharpes)
        console.print(
            f"\n[bold]Best OOS factor: {best}[/] — OOS Sharpe {results[best].oos_sharpe_ann:+.2f}, "
            f"IC t={results[best].oos_ic_t}"
        )
        console.print(
            f"[bold]Deflated Sharpe[/] ({len(builders)} factors tried, "
            f"skew/kurt-adjusted): SR0 haircut={dsr['sr0']:.3f}, "
            f"[bold]DSR={dsr['dsr']:.3f}[/] (prob true Sharpe>0 after deflation)"
        )
        if dsr["dsr"] > 0.95:
            console.print(f"\n[bold green]VERDICT: {best} survives OOS + deflation (DSR>0.95). "
                          "Genuine factor.[/]")
        elif dsr["dsr"] > 0.80:
            console.print(f"\n[bold yellow]VERDICT: {best} survives OOS; DSR {dsr['dsr']:.2f} "
                          "— likely real but not conclusive.[/]")
        else:
            console.print(f"\n[bold red]VERDICT: {best} OOS Sharpe positive but DSR {dsr['dsr']:.2f} "
                          "— not distinguishable from selection luck.[/]")
        return

    if combo:
        from autotrader.analysis.cross_validation import (
            _block_returns, deflated_sharpe_ratio, walk_forward_composite, walk_forward_xs,
        )
        # right-signed, low-correlated, economically-grounded components
        lowvol = xs.sig_lowvol(closes, 168)      # weekly vol (economic prior)
        reversal = xs.sig_reversal(closes, 6)    # short-horizon reversal
        corr = xs.factor_correlation(lowvol, reversal)
        console.print(f"factor correlation (lowvol vs reversal): [cyan]{corr:+.2f}[/] "
                      f"{'(good breadth)' if abs(corr) < 0.3 else '(redundant)'}\n")
        naive = xs.combine_factors([lowvol, reversal])

        results, block_rets, trial_sharpes = {}, {}, []
        singles = {
            "lowvol(alone)": (lambda lb: xs.sig_lowvol(closes, lb), [96, 168]),
            "reversal(alone)": (lambda lb: xs.sig_reversal(closes, lb), [6, 12]),
            "COMPOSITE(equal-wt)": (lambda lb: naive, [1]),
        }
        for name, (fn, grid) in singles.items():
            wf = walk_forward_xs(fn, closes, horizon=horizon, lookback_grid=grid,
                                 fees=cfg["fees"], bars_per_year=cfg.bars_per_year)
            results[name] = wf
        # cost-aware composite: inverse-turnover weights fit on train each fold
        results["COMPOSITE(cost-aware)"] = walk_forward_composite(
            [lowvol, reversal], closes, horizon=horizon,
            fees=cfg["fees"], bars_per_year=cfg.bars_per_year,
        )
        for name, wf in results.items():
            br = _block_returns(wf.oos_returns, horizon)
            block_rets[name] = br
            if br.std() > 0:
                trial_sharpes.append(float(br.mean() / br.std()))

        table = Table(title="Multi-factor combination — walk-forward OOS", header_style="bold")
        for col in ("book", "OOS IC", "OOS IC-t", "OOS Sharpe", "OOS return", "folds"):
            table.add_column(col)
        for name, wf in results.items():
            sig = abs(wf.oos_ic_t) > 2
            ics = f"[green]{wf.oos_ic_mean:+.4f}[/]" if sig else f"{wf.oos_ic_mean:+.4f}"
            table.add_row(name, ics, f"{wf.oos_ic_t:+.2f}", f"{wf.oos_sharpe_ann:+.2f}",
                          f"{wf.oos_total_return:+.2%}", str(wf.n_folds))
        console.print(table)

        comp = results["COMPOSITE(cost-aware)"]
        # n_trials reflects the broader factor search (5 singles + combos ~ 7)
        dsr = deflated_sharpe_ratio(block_rets["COMPOSITE(cost-aware)"], n_trials=7,
                                    trial_sharpes=trial_sharpes)
        console.print(
            f"\n[bold]COMPOSITE(cost-aware)[/] OOS Sharpe {comp.oos_sharpe_ann:+.2f}, "
            f"IC t={comp.oos_ic_t}\n"
            f"[bold]Deflated Sharpe[/] (n_trials=7): SR0 haircut={dsr['sr0']:.3f}, "
            f"[bold]DSR={dsr['dsr']:.3f}[/]"
        )
        lv = results["lowvol(alone)"].oos_sharpe_ann
        better = comp.oos_sharpe_ann > lv
        console.print(f"[dim]cost-aware composite Sharpe {comp.oos_sharpe_ann:+.2f} vs lowvol-alone "
                      f"{lv:+.2f} → combination {'helps' if better else 'does not beat lowvol alone'}[/]")
        if dsr["dsr"] > 0.95:
            console.print("\n[bold green]VERDICT: composite clears OOS + deflated Sharpe. "
                          "First deployable book.[/]")
        elif dsr["dsr"] > 0.80:
            console.print(f"\n[bold yellow]VERDICT: composite promising (DSR {dsr['dsr']:.2f}) "
                          "— stronger than any single factor, not yet conclusive.[/]")
        else:
            console.print(f"\n[bold red]VERDICT: composite still below the bar (DSR {dsr['dsr']:.2f}). "
                          "Breadth helped but not enough — need more uncorrelated factors.[/]")
        return

    lib = xs.factor_library(panels, horizon)
    table = Table(show_header=True, header_style="bold")
    for col in ("factor", "mean IC", "IC-IR", "IC t-stat", "LS return", "LS Sharpe", "ann turn"):
        table.add_column(col)
    best = None
    for name, sig in lib.items():
        ic = xs.rank_ic(sig, closes, horizon)
        ps = xs.xs_backtest(sig, closes, horizon, cfg["fees"]["taker"],
                            cfg["fees"]["slippage"], cfg.bars_per_year)
        sig_flag = abs(ic.t_stat) > 2
        ic_str = f"[green]{ic.mean_ic:+.4f}[/]" if sig_flag else f"{ic.mean_ic:+.4f}"
        table.add_row(name, ic_str, f"{ic.ic_ir:+.3f}", f"{ic.t_stat:+.2f}",
                      f"{ps.total_return:+.2%}", f"{ps.sharpe:+.2f}", f"{ps.ann_turnover:.0f}x")
        score = ps.sharpe if sig_flag else -999
        if best is None or score > best[1]:
            best = (name, ps.sharpe, ic.t_stat)
    console.print(table)
    console.print(
        "[dim]rank IC = Spearman(signal, fwd return) ACROSS coins each period; "
        "t-stat on non-overlapping periods. LS = dollar-neutral long-short, net of costs.[/]"
    )
    if best and abs(best[2]) > 2 and best[1] > 0:
        console.print(f"\n[bold green]Best factor: {best[0]} — IC significant (t={best[2]}), "
                      f"long-short Sharpe {best[1]:+.2f} net.[/]")
    else:
        console.print("\n[bold yellow]No factor clears IC t>2 with positive net long-short Sharpe.[/]")
    console.print("[dim]Validate survivors with a purged/panel split before believing them.[/]")


@app.command()
def findings(
    config: str = typer.Option("config.yaml"),
    symbol: list[str] = typer.Option(None, help="universe (repeatable); default basket"),
    history: int = typer.Option(8000),
    horizon: int = typer.Option(12, help="rebalance horizon (bars)"),
    expanded: bool = typer.Option(False, "--expanded", help="add breadth factors (lottery/lowbeta/longrev)"),
    top: int = typer.Option(0, "--top", help="liquidity-screen to top-N coins by dollar volume"),
    nest_universe: bool = typer.Option(False, "--nest-universe", help="pick universe size per fold (leak-free)"),
    directional: float = typer.Option(0.0, "--directional", help="compare neutral vs directional overlay at this k (e.g. 0.5, 1.0)"),
):
    """Our house strategy — the cross-sectional book built from the findings.
    Characterizes it honestly: in-sample equity + out-of-sample + deflated Sharpe."""
    from autotrader.analysis import cross_sectional as xs
    from autotrader.analysis.cross_validation import (
        _block_returns, deflated_sharpe_ratio, walk_forward_composite, walk_forward_xs,
    )
    from autotrader.backtest.report import sparkline
    from autotrader.data.feed import fetch_ohlcv_extended
    from autotrader.portfolio import DEFAULT_FACTORS, EXPANDED_FACTORS, FindingsStrategy

    cfg = _cfg(config)
    universe = symbol or [
        "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", "ADA/USDT",
        "DOGE/USDT", "AVAX/USDT", "LINK/USDT", "LTC/USDT", "DOT/USDT", "TRX/USDT",
    ]
    dfs = {}
    for s in universe:
        console.print(f"[dim]fetching {s}...[/]")
        try:
            dfs[s] = fetch_ohlcv_extended(s, cfg["timeframe"], history,
                                          cfg["exchange"], cfg["data_cache_dir"])
        except Exception as e:
            console.print(f"[yellow]skip {s}: {e}[/]")

    _panels = xs.build_panels(dfs)
    closes, dollar_vol = _panels["closes"], _panels["dollar_vol"]
    factors = EXPANDED_FACTORS if expanded else DEFAULT_FACTORS
    strat = FindingsStrategy(factors=list(factors), rebalance=horizon)
    fees, bpy = cfg["fees"], cfg.bars_per_year

    console.rule(f"[bold]findings_xs{'(expanded)' if expanded else ''} — "
                 f"{len(closes.columns)} coins, {len(closes)} bars[/]")

    if expanded:
        # breadth diagnostics: correlation to lowvol + per-factor OOS Sharpe
        lv = xs.sig_lowvol(closes, 168)
        tbl = Table(title="Breadth factors", header_style="bold")
        for c in ("factor", "corr→lowvol", "OOS IC-t", "OOS Sharpe"):
            tbl.add_column(c)
        for name, builder in factors:
            sig = builder(closes)
            corr = xs.factor_correlation(sig, lv) if name != "lowvol" else 1.0
            wf1 = walk_forward_xs(lambda lb, b=builder: b(closes), closes, horizon=horizon,
                                  lookback_grid=[1], fees=fees, bars_per_year=bpy)
            tbl.add_row(name, f"{corr:+.2f}", f"{wf1.oos_ic_t:+.2f}", f"{wf1.oos_sharpe_ann:+.2f}")
        console.print(tbl)
        console.print()

    # in-sample characterization
    ch = strat.characterize(closes, fees, bpy)
    alloc = " / ".join(f"{n} {w:.0%}" for n, w in ch["factor_weights"].items())
    console.print(f"[bold]factor allocation (inverse-turnover):[/] {alloc}")
    console.print("[bold]In-sample (full period):[/]")
    console.print(f"  return {ch['total_return']:+.1%} · Sharpe {ch['sharpe']:+.2f} · "
                  f"maxDD {ch['max_drawdown']:+.1%} · turnover {ch['ann_turnover']:.0f}x/yr")
    eq = ch["equity"]
    console.print(f"  equity {sparkline(eq)}  {eq.iloc[-1]:.3f}x\n")

    if directional > 0:
        console.rule("[bold]Directional overlay — neutral vs market-beta tilt[/]")
        cmp = Table(header_style="bold")
        for c in ("book", "return", "Sharpe", "maxDD", "turnover", "avg|net|"):
            cmp.add_column(c)
        for k in (0.0, directional):
            d = FindingsStrategy(factors=list(factors), rebalance=horizon,
                                 directional_weight=k).characterize(closes, fees, bpy)
            label = "neutral (k=0)" if k == 0 else f"directional (k={k})"
            cmp.add_row(label, f"{d['total_return']:+.1%}", f"{d['sharpe']:+.2f}",
                        f"{d['max_drawdown']:+.1%}", f"{d['ann_turnover']:.0f}x",
                        f"{d['avg_net_exposure']:.2f}")
        console.print(cmp)
        console.print("[dim]bigger return usually comes with a bigger drawdown — "
                      "that's the beta you're now taking on.[/]\n")

    # out-of-sample: cost-aware walk-forward + deflated Sharpe
    panels = strat._panels(closes)
    wf = walk_forward_composite(panels, closes, horizon=horizon, band=strat.band,
                                fees=fees, bars_per_year=bpy)
    blocks = _block_returns(wf.oos_returns, horizon)
    # honest trial set: each factor's own OOS block Sharpe; n_trials = factors tried
    trial_sharpes = []
    for _, builder in factors:
        w1 = walk_forward_xs(lambda lb, b=builder: b(closes), closes, horizon=horizon,
                             lookback_grid=[1], fees=fees, bars_per_year=bpy)
        br = _block_returns(w1.oos_returns, horizon)
        if br.std() > 0:
            trial_sharpes.append(float(br.mean() / br.std()))
    dsr = deflated_sharpe_ratio(blocks, n_trials=max(len(factors), 5),
                                trial_sharpes=trial_sharpes or None)
    console.print("[bold]Out-of-sample (walk-forward, peek-free):[/]")
    console.print(f"  OOS IC t={wf.oos_ic_t} · OOS Sharpe {wf.oos_sharpe_ann:+.2f} · "
                  f"OOS return {wf.oos_total_return:+.1%} · {wf.n_folds} folds")
    console.print(f"  deflated Sharpe DSR={dsr['dsr']:.3f} (haircut SR0={dsr['sr0']:.3f}, "
                  f"n_trials={max(len(factors), 5)})\n")

    if dsr["dsr"] > 0.95:
        console.print("[bold green]This book clears the deflated-Sharpe bar (DSR>0.95). Deployable.[/]")
    elif dsr["dsr"] > 0.80:
        console.print(f"[bold yellow]Promising: DSR {dsr['dsr']:.2f} — strong but short of the "
                      "0.95 bar. Closest we've gotten.[/]")
    else:
        console.print(f"[bold yellow]Honest status: real market-neutral low-turnover sleeve "
                      f"(OOS Sharpe {wf.oos_sharpe_ann:+.2f}) — DSR {dsr['dsr']:.2f}, does NOT clear "
                      "the bar alone. The best evidence-backed book we have.[/]")


    if top:
        import numpy as np
        from autotrader.analysis.cross_sectional import liquidity_mask, screen_signal, xs_net
        mask = liquidity_mask(dollar_vol, top, 336)
        latest = mask.iloc[-1]
        kept = [c.split("/")[0] for c in latest[latest].index]
        dropped = [c.split("/")[0] for c in closes.columns if c not in latest[latest].index]
        console.rule(f"[bold]Liquidity screen -> top {top} by dollar volume[/]")
        console.print(f"kept (latest): [green]{kept}[/]")
        console.print(f"dropped: [red]{dropped}[/]\n")

        comp, _, _ = strat.composite(closes, fees, bpy)
        comp_s = screen_signal(comp, mask)
        net_s, _ = xs_net(comp_s, closes, strat.rebalance, fees["taker"], fees["slippage"], strat.band)
        net_s = net_s.dropna()
        eq_s = (1 + net_s).cumprod()
        sh_s = float(net_s.mean() / net_s.std() * np.sqrt(bpy)) if net_s.std() > 0 else 0.0
        console.print(f"[bold]Screened in-sample:[/] return {eq_s.iloc[-1]-1:+.1%} - Sharpe {sh_s:+.2f}")
        console.print(f"  equity {sparkline(eq_s)}  {eq_s.iloc[-1]:.3f}x")

        wf_s = walk_forward_composite(panels, closes, horizon=horizon, band=strat.band,
                                      dollar_vol=dollar_vol, top_n=top,
                                      fees=fees, bars_per_year=bpy)
        blk_s = _block_returns(wf_s.oos_returns, horizon)
        dsr_s = deflated_sharpe_ratio(blk_s, n_trials=max(len(factors), 5),
                                      trial_sharpes=trial_sharpes or None)
        console.print(f"[bold]Screened OOS:[/] IC t={wf_s.oos_ic_t} - Sharpe {wf_s.oos_sharpe_ann:+.2f} - "
                      f"return {wf_s.oos_total_return:+.1%} - DSR={dsr_s['dsr']:.3f}")
        d_sh = wf_s.oos_sharpe_ann - wf.oos_sharpe_ann
        d_dsr = dsr_s["dsr"] - dsr["dsr"]
        helps = d_sh > 0.05 and d_dsr > 0
        verdict = ("screening HELPS" if helps else
                   "screening doesn't help" if d_sh <= 0 else "screening ~neutral")
        color = "green" if helps else "yellow"
        console.print(f"\n[bold {color}]vs full universe: dSharpe {d_sh:+.2f}, "
                      f"dDSR {d_dsr:+.3f} -> {verdict}[/]")

    if nest_universe:
        from collections import Counter

        from autotrader.analysis.cross_validation import walk_forward_nested_universe
        console.rule("[bold]Nested universe-size selection (leak-free)[/]")
        wf_n = walk_forward_nested_universe(
            panels, closes, dollar_vol, horizon=horizon, n_grid=(6, 8, 10, 12),
            band=strat.band, fees=fees, bars_per_year=bpy)
        blk_n = _block_returns(wf_n.oos_returns, horizon)
        # n_trials counts the universe-size search honestly (factors + grid sizes)
        dsr_n = deflated_sharpe_ratio(blk_n, n_trials=max(len(factors), 5) + 4,
                                      trial_sharpes=trial_sharpes or None)
        picks = Counter(wf_n.chosen_lookbacks)
        mode_n, mode_count = picks.most_common(1)[0]
        stable = mode_count / max(wf_n.n_folds, 1) >= 0.7
        console.print(f"N chosen per fold: {wf_n.chosen_lookbacks}  → {dict(picks)}")
        console.print(f"[bold]Nested OOS (leak-free):[/] IC t={wf_n.oos_ic_t} · "
                      f"Sharpe {wf_n.oos_sharpe_ann:+.2f} · return {wf_n.oos_total_return:+.1%} · "
                      f"{wf_n.n_folds} folds · DSR={dsr_n['dsr']:.3f}")
        d = wf_n.oos_sharpe_ann - wf.oos_sharpe_ann
        if stable and d > 0.1:
            console.print(f"[bold green]vs full universe: dSharpe {d:+.2f}. Train consistently picks "
                          f"N={mode_n} ({mode_count}/{wf_n.n_folds} folds) and the lift CARRIES OVER "
                          "out-of-sample — a real, leak-free improvement (still sub-DSR-bar).[/]")
        elif d > 0.1:
            console.print(f"[yellow]vs full universe: dSharpe {d:+.2f}, but N selection is unstable "
                          f"({dict(picks)}) — treat the lift with caution.[/]")
        else:
            console.print(f"[yellow]vs full universe: dSharpe {d:+.2f} — no leak-free benefit.[/]")


@app.command()
def carry(
    config: str = typer.Option("config.yaml"),
    symbol: list[str] = typer.Option(None, help="perp universe (repeatable)"),
    history: int = typer.Option(8000),
    lookback: int = typer.Option(24, help="funding smoothing window (bars)"),
    rebalance: int = typer.Option(8, help="rebalance every N bars (funding period)"),
    band: float = typer.Option(0.01, help="no-trade band"),
):
    """Funding-carry harvest: market-neutral book that COLLECTS the funding
    spread. Decomposes carry vs price P&L; OOS + deflated Sharpe."""
    from autotrader.analysis import cross_sectional as xs
    from autotrader.analysis.cross_validation import _block_returns, deflated_sharpe_ratio
    from autotrader.analysis.funding_carry import (
        carry_backtest, carry_signal, walk_forward_carry,
    )
    from autotrader.backtest.report import sparkline
    from autotrader.data.derivatives import fetch_perp_with_funding

    cfg = _cfg(config)
    universe = symbol or [
        "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", "ADA/USDT",
        "DOGE/USDT", "AVAX/USDT", "LINK/USDT", "LTC/USDT", "DOT/USDT", "TRX/USDT",
    ]
    dfs = {}
    for s in universe:
        console.print(f"[dim]fetching {s} perp+funding...[/]")
        try:
            dfs[s] = fetch_perp_with_funding(s, cfg["timeframe"], history,
                                             cache_dir=cfg["data_cache_dir"])
        except Exception as e:
            console.print(f"[yellow]skip {s}: {e}[/]")

    panels = xs.build_panels(dfs)
    closes, funding = panels["closes"], panels["funding"]
    fees, bpy = cfg["fees"], cfg.bars_per_year
    console.rule(f"[bold]Funding carry — {len(closes.columns)} perps, {len(closes)} bars[/]")

    sig = carry_signal(funding, lookback)
    res = carry_backtest(closes, funding, sig, rebalance, fees["taker"],
                         fees["slippage"], bpy, band)
    console.print("[bold]In-sample (full period):[/]")
    console.print(f"  total return {res['total_return']:+.1%} · Sharpe {res['sharpe']:+.2f} · "
                  f"maxDD {res['max_drawdown']:+.1%} · turnover {res['ann_turnover']:.0f}x/yr")
    console.print(f"  decomposition → [green]carry {res['carry_contrib']:+.1%}[/] · "
                  f"price {res['price_contrib']:+.1%} · cost {res['cost_drag']:+.1%}")
    eq = res["equity"]
    console.print(f"  equity {sparkline(eq)}  {eq.iloc[-1]:.3f}x\n")

    wf = walk_forward_carry(closes, funding, lookback=lookback, rebalance=rebalance,
                            band=band, fees=fees, bars_per_year=bpy)
    blocks = _block_returns(wf.oos_returns, rebalance)
    dsr = deflated_sharpe_ratio(blocks, n_trials=8,
                                trial_sharpes=[float(blocks.mean()/blocks.std())] if blocks.std() else None)
    console.print("[bold]Out-of-sample (walk-forward):[/]")
    console.print(f"  OOS Sharpe {wf.oos_sharpe_ann:+.2f} · OOS return {wf.oos_total_return:+.1%} · "
                  f"{wf.n_folds} folds · deflated Sharpe DSR={dsr['dsr']:.3f}")
    if dsr["dsr"] > 0.95:
        console.print("\n[bold green]VERDICT: funding carry clears the bar (DSR>0.95). "
                      "This is a genuine, significant edge.[/]")
    elif dsr["dsr"] > 0.80:
        console.print(f"\n[bold yellow]VERDICT: strong (DSR {dsr['dsr']:.2f}) — best candidate yet, "
                      "short of conclusive.[/]")
    else:
        console.print(f"\n[bold red]VERDICT: DSR {dsr['dsr']:.2f} — carry exists but is "
                      "arbed/risky enough not to clear the bar net.[/]")
    console.print("[dim]Caveat: carry has crash/short-squeeze tail risk a Sharpe doesn't capture, "
                  "and assumes you can actually short each perp at these funding rates.[/]")


@app.command()
def statarb(
    config: str = typer.Option("config.yaml"),
    symbol: list[str] = typer.Option(None, help="universe (repeatable)"),
    history: int = typer.Option(8000),
    k: int = typer.Option(8, help="number of pairs in the book"),
    entry: float = typer.Option(2.0, help="z-score entry threshold"),
    exit: float = typer.Option(0.5, help="z-score exit threshold"),
):
    """Statistical arbitrage: cointegration pairs (relative-value mean reversion),
    leak-free walk-forward + deflated Sharpe."""
    from autotrader.analysis import cross_sectional as xs
    from autotrader.analysis.cross_validation import _block_returns, deflated_sharpe_ratio
    from autotrader.analysis.statarb import (
        book_net, select_pairs, walk_forward_statarb,
    )
    from autotrader.backtest.report import sparkline
    from autotrader.data.feed import fetch_ohlcv_extended

    cfg = _cfg(config)
    universe = symbol or [
        "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", "ADA/USDT",
        "DOGE/USDT", "AVAX/USDT", "LINK/USDT", "LTC/USDT", "DOT/USDT", "TRX/USDT",
    ]
    dfs = {}
    for s in universe:
        console.print(f"[dim]fetching {s}...[/]")
        try:
            dfs[s] = fetch_ohlcv_extended(s, cfg["timeframe"], history,
                                          cfg["exchange"], cfg["data_cache_dir"])
        except Exception as e:
            console.print(f"[yellow]skip {s}: {e}[/]")

    closes = xs.build_panels(dfs)["closes"]
    fees, bpy = cfg["fees"], cfg.bars_per_year
    console.rule(f"[bold]Stat-arb pairs — {len(closes.columns)} coins, {len(closes)} bars[/]")

    # show the top pairs selected on the full sample (illustrative)
    pairs = select_pairs(closes, k, entry, exit, fees, bpy)
    if pairs:
        tbl = Table(title="Top cointegration pairs (full-sample, illustrative)", header_style="bold")
        for c in ("pair", "beta", "half-life(h)", "in-sample Sharpe"):
            tbl.add_column(c)
        for p in pairs[:k]:
            tbl.add_row(f"{p.i.split('/')[0]}/{p.j.split('/')[0]}", f"{p.beta:.2f}",
                        f"{p.half_life:.0f}", f"{p.train_sharpe:+.2f}")
        console.print(tbl)
        net = book_net(closes, pairs, entry, exit, fees).dropna()
        eq = (1 + net).cumprod()
        console.print(f"in-sample book equity {sparkline(eq)}  {eq.iloc[-1]:.3f}x\n")

    wf = walk_forward_statarb(closes, k=k, entry=entry, exit=exit,
                              fees=fees, bars_per_year=bpy)
    blocks = _block_returns(wf.oos_returns, 24)
    dsr = deflated_sharpe_ratio(blocks, n_trials=9,
                                trial_sharpes=[float(blocks.mean()/blocks.std())] if blocks.std() else None)
    console.print("[bold]Out-of-sample (walk-forward, pairs reselected per fold):[/]")
    console.print(f"  OOS Sharpe {wf.oos_sharpe_ann:+.2f} · OOS return {wf.oos_total_return:+.1%} · "
                  f"{wf.n_folds} folds · ~{wf.avg_pairs} pairs/fold · DSR={dsr['dsr']:.3f}")
    if dsr["dsr"] > 0.95:
        console.print("\n[bold green]VERDICT: stat-arb clears the bar (DSR>0.95). Genuine edge.[/]")
    elif dsr["dsr"] > 0.80:
        console.print(f"\n[bold yellow]VERDICT: promising (DSR {dsr['dsr']:.2f}).[/]")
    else:
        console.print(f"\n[bold red]VERDICT: DSR {dsr['dsr']:.2f} — pairs mean-revert in-sample but "
                      "the edge doesn't survive net + OOS. Crypto pairs are heavily arbed.[/]")


@app.command()
def record(
    symbol: str = typer.Argument("BTC/USDT"),
    config: str = typer.Option("config.yaml"),
    seconds: int = typer.Option(30, help="how long to capture"),
    depth: int = typer.Option(20, help="order-book levels"),
):
    """Capture live L2 order book + trades to disk (the only way to get tick/L2
    data — record it going forward). Run for days/weeks to build a dataset."""
    import asyncio

    from autotrader.live.recorder import record as record_l2

    cfg = _cfg(config)
    out = f"{cfg['data_cache_dir']}/l2"
    console.print(f"[bold green]Recording L2 + trades[/] for {symbol} ({seconds}s)...")
    try:
        path = asyncio.run(record_l2(symbol, seconds, out, depth))
    except ImportError:
        console.print("[red]Install the stream dep:[/] pip install 'autotrader[stream]'")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]websocket error:[/] {e}")
        raise typer.Exit(1)

    import pandas as pd
    df = pd.read_csv(path)
    console.print(f"[green]captured {len(df)} L2 snapshots → {path}[/]")
    if len(df):
        console.print(f"  mid {df['mid'].iloc[-1]:,.2f} · "
                      f"spread {df['spread_bps'].mean():.2f}bps avg · "
                      f"imbalance range [{df['imbalance'].min():+.2f}, {df['imbalance'].max():+.2f}] · "
                      f"avg trades/snap {df['n_trades'].mean():.1f}")
        console.print("[dim]This is real L2 data — no free history exists; keep recording to "
                      "accumulate a backtestable dataset, then feed it to the harness.[/]")


@app.command()
def micro(
    symbol: str = typer.Argument("BTC/USDT"),
    config: str = typer.Option("config.yaml"),
    timeframe: str = typer.Option("5m", help="fine bar size (1m/5m/15m)"),
    bars: int = typer.Option(20000, help="how many fine bars to pull"),
):
    """Finer-bar test: is the order-flow edge alive at 1-5m before costs?
    Feature IC at fine frequency + the brutal cost honesty (taker fee cliff)."""
    from autotrader.analysis.feature_ic import compute_ic
    from autotrader.backtest.engine import run_backtest
    from autotrader.config import BARS_PER_YEAR
    from autotrader.data.feed import fetch_ohlcv_extended
    from autotrader.features.microstructure import feature_frame, has_flow
    from autotrader.strategies.registry import get

    cfg = _cfg(config)
    bpy = BARS_PER_YEAR.get(timeframe, 105120)
    df = fetch_ohlcv_extended(symbol, timeframe, bars, cfg["exchange"], cfg["data_cache_dir"])
    console.rule(f"[bold]Finer bars — {symbol} @ {timeframe}, {len(df)} bars[/]")
    if not has_flow(df):
        console.print("[yellow]No order-flow columns.[/]")
        raise typer.Exit(1)

    feats = feature_frame(df)
    tbl = Table(title="Feature IC at fine frequency", header_style="bold")
    for c in ("feature", "IC@1", "IC@3", "IC@6"):
        tbl.add_column(c)
    for name in feats.columns:
        row = [name]
        for h in (1, 3, 6):
            ic = compute_ic(feats[[name]], df["close"], horizon=h)[0]
            sig = abs(ic.t_stat) > 2
            row.append((f"[green]{ic.ic:+.3f}[/]" if sig else f"{ic.ic:+.3f}") + f"(t{ic.t_stat:+.1f})")
        tbl.add_row(*row)
    console.print(tbl)

    # cost honesty: order-flow momentum, gross vs net (taker), breakeven fee
    strat = get("orderflow_momentum")(window=6, threshold=0.05, allow_short=True)
    gross = run_backtest(strat, df, 0.0, 0.0, bpy)
    net = run_backtest(strat, df, cfg["fees"]["taker"], cfg["fees"]["slippage"], bpy)
    traded = float(gross.positions.diff().abs().sum())
    breakeven_bps = (gross.metrics.total_return / traded * 1e4) if traded > 0 else 0.0
    taker_bps = (cfg["fees"]["taker"] + cfg["fees"]["slippage"]) * 1e4
    console.print(f"\n[bold]orderflow_momentum @ {timeframe}:[/]")
    console.print(f"  GROSS: return {gross.metrics.total_return:+.1%}, Sharpe {gross.metrics.sharpe:+.2f}, "
                  f"{gross.metrics.num_trades} trades")
    console.print(f"  NET (taker): return {net.metrics.total_return:+.1%}, Sharpe {net.metrics.sharpe:+.2f}")
    console.print(f"  [bold]breakeven cost {breakeven_bps:.2f} bps/trade  vs  taker+slip {taker_bps:.1f} bps[/]")
    if breakeven_bps < taker_bps:
        console.print(f"\n[bold red]The edge is real gross but smaller than taker costs "
                      f"({breakeven_bps:.1f} < {taker_bps:.0f} bps). Only viable as a MAKER "
                      "(rebates + queue position) — which needs the live L2 data `autotrader record` "
                      "captures. This is the honest finer-bar wall.[/]")
    else:
        console.print(f"\n[bold green]Edge survives taker costs at {timeframe} "
                      f"(breakeven {breakeven_bps:.1f} > {taker_bps:.0f} bps). Worth full validation.[/]")


@app.command()
def live(
    config: str = typer.Option("config.yaml"),
    symbol: list[str] = typer.Option(None, help="universe (repeatable)"),
    history: int = typer.Option(2000),
    equity: float = typer.Option(0.0, help="equity override for dry-run sizing"),
    testnet: bool = typer.Option(False, "--testnet", help="route to exchange sandbox"),
    arm: bool = typer.Option(False, "--arm", help="actually place orders (off = dry-run)"),
    confirm_live: bool = typer.Option(False, "--confirm-live", help="REQUIRED to trade REAL money"),
    spot: bool = typer.Option(False, "--spot", help="trade spot (long-only); default is futures (long/short)"),
    loop: bool = typer.Option(False, "--loop", help="run continuously, rebalancing each interval"),
    interval: int = typer.Option(0, help="loop interval seconds (default: config poll_seconds)"),
):
    """Trade the findings_xs market-neutral book through the live broker — SAFE BY
    DEFAULT (dry-run). The book is long/short, so it needs FUTURES (default);
    --spot is long-only. Real money requires --arm AND --confirm-live AND keys."""
    import os
    import time as _time

    from autotrader.analysis import cross_sectional as xs
    from autotrader.data.feed import fetch_ohlcv_extended
    from autotrader.live.live_broker import LiveBroker
    from autotrader.portfolio import EXPANDED_FACTORS, FindingsStrategy
    from autotrader.risk import RiskLimits, RiskManager

    cfg = _cfg(config)
    key, secret = os.getenv("BINANCE_API_KEY", ""), os.getenv("BINANCE_SECRET", "")
    # market-neutral book needs shorts → futures (binanceusdm). --spot = long-only.
    exec_exchange = cfg["exchange"] if spot else "binanceusdm"

    # ---- safety gauntlet ----
    if arm and not testnet and not confirm_live:
        console.print("[bold red]REFUSED:[/] real-money trading needs --confirm-live (and you should "
                      "go dry-run → testnet → tiny real first). Not arming.")
        raise typer.Exit(1)
    if arm and not key:
        console.print("[bold red]REFUSED:[/] no BINANCE_API_KEY/SECRET in env — cannot place orders.")
        raise typer.Exit(1)

    # separate risk state per venue so a paper/testnet peak can't false-trip
    # another; dry-runs get their OWN file so a previewed (guessed) equity can't
    # poison the armed venue's drawdown peak.
    venue = "testnet" if testnet else "mainnet"
    suffix = "" if arm else "_dryrun"
    risk_file = f"{cfg['data_cache_dir']}/risk_{venue}{suffix}.json"
    rm = RiskManager(RiskLimits(**cfg.get("risk", {})), state_path=risk_file)
    if spot:
        broker = LiveBroker(exchange_id=exec_exchange, api_key=key, secret=secret,
                            dry_run=not arm, testnet=testnet, risk=rm)
        is_futures = False
    else:
        # futures (long/short) via direct REST — ccxt's futures testnet is deprecated
        from autotrader.live.binance_futures import FuturesBroker, FuturesClient
        broker = FuturesBroker(FuturesClient(key, secret, testnet=testnet),
                               risk=rm, dry_run=not arm, testnet=testnet)
        is_futures = True
    console.rule(f"[bold]LIVE — {broker.mode()} · {'spot/long-only' if spot else 'futures/long-short'} "
                 f"· {exec_exchange}[/]")
    if rm.halted:
        console.print(f"[bold red]kill-switch HALTED ({rm.state.halt_reason}). "
                      "Run `autotrader risk --resume` first.[/]")
        raise typer.Exit(1)

    # 12-coin universe matching the walk-forward validation (do not trim — the
    # confirmed book leans on low-beta names like TRX that the 8-coin set omitted)
    universe = symbol or ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT",
                          "XRP/USDT", "ADA/USDT", "DOGE/USDT", "AVAX/USDT",
                          "LINK/USDT", "LTC/USDT", "DOT/USDT", "TRX/USDT"]
    scfg = cfg.get("strategy", {})
    long_only = bool(scfg.get("long_only", False))
    dcfg = cfg.get("directional", {})
    dweight = float(dcfg.get("weight", 0.0))
    leverage = float(dcfg.get("leverage", 1.0))
    if long_only:
        from autotrader.analysis.cross_sectional import sig_lottery, sig_lowbeta, sig_lowvol
        fac_fns = {"lowbeta": sig_lowbeta, "lowvol": sig_lowvol, "lottery": sig_lottery}
        fname = scfg.get("factor", "lowbeta")
        flb = int(scfg.get("lookback", 720))
        fn = fac_fns[fname]
        strat = FindingsStrategy(
            factors=[(fname, lambda c, fn=fn, lb=flb: fn(c, lb))],
            long_only=True, directional_weight=0.0,
            max_weight=float(scfg.get("max_weight", 0.0)))
        console.print(f"[cyan]LONG-ONLY smart-beta: {fname}({flb}) · leverage {leverage}x "
                      f"· no shorts (walk-forward-confirmed book)[/]")
    else:
        strat = FindingsStrategy(
            factors=list(EXPANDED_FACTORS),
            directional_weight=dweight,
            trend_fast=int(dcfg.get("trend_fast", 24)),
            trend_slow=int(dcfg.get("trend_slow", 168)),
            trend_scale=float(dcfg.get("trend_scale", 2.0)),
        )
        if dweight > 0 or leverage != 1.0:
            console.print(f"[cyan]directional overlay: k={dweight} · leverage={leverage}x "
                          f"· gross cap {rm.limits.max_gross_exposure}[/]")
    rcfg = cfg.get("regime", {})
    regime_on = bool(rcfg.get("enabled", False))
    regime_lb = int(rcfg.get("lookback", 30))
    regime_mode = str(rcfg.get("mode", "flat"))
    if regime_on:
        if long_only:
            off_desc = "FLAT (cash)"
        elif regime_mode == "tsmom_short":
            off_desc = "short-only directional (hybrid)"
        else:
            off_desc = "go market-neutral"
        console.print(f"[cyan]stablecoin risk-gate: ON (lookback {regime_lb}d) — "
                      f"risk-off → {off_desc}[/]")
    poll = interval or int(cfg["paper"]["poll_seconds"])

    # track record: every tick's equity → a live out-of-sample Sharpe we can
    # later judge against the backtest with a pre-committed decision rule.
    from autotrader.live.ledger import Ledger
    ledger = Ledger(f"{cfg['data_cache_dir']}/ledger_{'testnet' if testnet else 'mainnet'}.jsonl",
                    interval_s=float(poll))

    def tick() -> bool:
        from datetime import datetime, timezone
        dfs = {}
        for s in universe:
            try:
                dfs[s] = fetch_ohlcv_extended(s, cfg["timeframe"], history, cfg["exchange"],
                                              cfg["data_cache_dir"], use_cache=False)
            except Exception:
                pass
        closes = xs.build_panels(dfs)["closes"]
        # risk-on/off gate. risk-off → de-lever, and either go market-neutral
        # ("flat") or keep a short-only directional tilt ("tsmom_short" hybrid:
        # profit confirmed downtrends, sit out flat chop). risk-on → full overlay.
        eff_lev, risk_on = leverage, True
        if regime_on:
            try:
                from autotrader.data.onchain import stablecoin_supply_cached, supply_regime
                supply = stablecoin_supply_cached(cfg["data_cache_dir"])
                risk_on = bool(supply_regime(supply, regime_lb).iloc[-1])
            except Exception as exc:
                console.print(f"[yellow]regime feed unavailable ({exc}); staying risk-on[/]")
            if long_only:
                # confirmed strategy: invested (levered) risk-on, flat (cash) risk-off
                eff_lev = leverage if risk_on else 0.0
                desc = (f"RISK-ON — long-only invested {leverage}x" if risk_on
                        else "RISK-OFF — FLAT (cash)")
            elif risk_on:
                strat.directional_weight, strat.exposure_clip, eff_lev = dweight, "", leverage
                desc = "RISK-ON — full overlay"
            elif regime_mode == "tsmom_short":
                strat.directional_weight, strat.exposure_clip, eff_lev = dweight, "short_only", 1.0
                desc = "RISK-OFF — short-only directional, de-levered"
            else:  # "flat"
                strat.directional_weight, strat.exposure_clip, eff_lev = 0.0, "", 1.0
                desc = "RISK-OFF — beta cut, de-levered"
            console.print(f"  [{'green' if risk_on else 'red'}]regime: {desc}[/]")
        target = strat.target_weights(closes, cfg["fees"], cfg.bars_per_year).iloc[-1].to_dict()
        if eff_lev != 1.0:
            target = {k: v * eff_lev for k, v in target.items()}
        eq = equity or broker.equity()
        if eq <= 0:
            if broker.dry_run:
                eq = float(cfg["paper"]["starting_cash"])   # preview only
            else:
                console.print("[yellow]skip tick: couldn't read live equity "
                              "(no guessed-equity fallback on a real venue)[/]")
                return False
        if is_futures:
            intents = broker.rebalance_to(target, eq)
        else:
            prices = broker.prices(universe) or {s: float(closes[s].iloc[-1]) for s in closes.columns}
            intents = broker.rebalance_to(target, eq, prices)
        stamp = datetime.now(timezone.utc).isoformat()[:19]
        sent = [i for i in intents if i.get("status") in ("sent", "simulated")]
        ledger.record(stamp, eq, target, intents, broker.mode())
        # desktop alerts on state transitions (halt / verdict ready / big move)
        if not broker.dry_run:
            from autotrader.live.alerts import check_and_alert
            summ = ledger.summary()
            check_and_alert(f"{cfg['data_cache_dir']}/alert_{venue}.json",
                            equity=eq, halted=rm.halted,
                            halt_reason=rm.state.halt_reason,
                            n_samples=summ["n_rebalances"], verdict=summ["verdict"],
                            risk_on=risk_on if regime_on else None)
        console.print(f"{stamp} | equity ${eq:,.0f} | {len(sent)} orders ({broker.mode().split()[0]})")
        for it in sent:
            console.print(f"  {it['side']} {it.get('qty', it.get('amount',''))} {it['symbol']} "
                          f"(${it['notional']:,.0f}) [{it['status']}]")
        for it in intents:
            if "risk_notes" in it:
                console.print(f"  [yellow]risk: {'; '.join(it['risk_notes'])}[/]")
        if rm.halted:
            console.print(f"  [bold red]HALTED — {rm.state.halt_reason}. Stopping.[/]")
            return True
        return False

    if loop:
        console.print(f"[green]continuous loop every {poll}s — Ctrl-C to stop[/]")
        while True:
            try:
                if tick():
                    break
            except Exception as exc:
                console.print(f"[red]tick error:[/] {exc}")
            _time.sleep(poll)
    else:
        tick()
        if broker.dry_run:
            console.print("[dim]DRY-RUN: nothing sent. Next: get FUTURES testnet keys (see TESTNET.md), "
                          "then `autotrader live --testnet --arm --loop`.[/]")


@app.command()
def risk(
    config: str = typer.Option("config.yaml"),
    resume: bool = typer.Option(False, "--resume", help="clear a tripped kill-switch (after review)"),
):
    """Inspect risk limits + kill-switch state; --resume to re-arm after a halt."""
    from autotrader.risk import RiskLimits, RiskManager

    cfg = _cfg(config)
    rm = RiskManager(RiskLimits(**cfg.get("risk", {})))
    lim, st = rm.limits, rm.state
    console.rule("[bold]Risk limits[/]")
    console.print(f"  max gross exposure : {lim.max_gross_exposure:.0%}")
    console.print(f"  max position/coin  : {lim.max_position_pct:.0%}")
    console.print(f"  drawdown kill-switch: −{lim.max_drawdown_pct:.0%} from peak")
    console.print(f"  daily loss limit   : −{lim.max_daily_loss_pct:.0%}")
    console.rule("[bold]Current state[/]")
    console.print(f"  peak equity : ${st.peak_equity:,.2f}")
    if st.halted:
        console.print(f"  [bold red]HALTED — {st.halt_reason}[/]")
        if resume:
            rm.resume()
            console.print("  [green]kill-switch cleared — trading re-armed.[/]")
        else:
            console.print("  [dim]review why, then `autotrader risk --resume` to re-arm.[/]")
    else:
        console.print("  [green]OK — trading allowed.[/]")


@app.command()
def macro(
    config: str = typer.Option("config.yaml"),
    symbol: str = typer.Option("BTC/USDT", help="beta proxy to time"),
    lookbacks: str = typer.Option("14,30,60,90", help="supply-momentum lookbacks (days)"),
):
    """NEW edge family: stablecoin-supply regime filter (DefiLlama, free, no key).
    Tests timing crypto beta on whether total stablecoin supply is growing."""
    import ccxt
    import numpy as np
    import pandas as pd

    from autotrader.data.onchain import stablecoin_supply, supply_regime

    cfg = _cfg(config)
    console.print("[dim]fetching stablecoin supply (DefiLlama) + daily price...[/]")
    ss = stablecoin_supply()
    ex = getattr(ccxt, cfg["exchange"])({"enableRateLimit": True})
    since = ex.parse8601("2017-08-01T00:00:00Z")
    rows = []
    while True:
        o = ex.fetch_ohlcv(symbol, "1d", since=since, limit=1000)
        if not o:
            break
        rows += o
        since = o[-1][0] + 86_400_000
        if len(o) < 1000:
            break
    px = pd.Series({pd.to_datetime(c[0], unit="ms").normalize(): c[4] for c in rows}).sort_index()
    px = px[~px.index.duplicated()]
    df = pd.concat([ss.rename("ss"), px.rename("px")], axis=1).dropna()
    ret = df["px"].pct_change().fillna(0.0)

    def stats(x):
        eq = (1 + x).cumprod()
        sh = float(x.mean() / x.std() * np.sqrt(365)) if x.std() > 0 else 0.0
        return float(eq.iloc[-1] - 1), sh, float((eq / eq.cummax() - 1).min())

    console.rule(f"[bold]Stablecoin-supply regime filter — {symbol}, "
                 f"{len(df)} days {df.index[0].date()}→{df.index[-1].date()}[/]")
    tbl = Table(header_style="bold")
    for c in ("strategy", "return", "Sharpe", "maxDD", "in mkt"):
        tbl.add_column(c)
    tr, sh, dd = stats(ret)
    tbl.add_row("buy & hold", f"{tr:+.0%}", f"{sh:+.2f}", f"{dd:.0%}", "100%")
    for lb in [int(x) for x in lookbacks.split(",")]:
        sig = supply_regime(df["ss"], lb)
        st = sig.shift(1).fillna(0.0) * ret
        tr, sh, dd = stats(st)
        tbl.add_row(f"long when {lb}d supply↑", f"{tr:+.0%}", f"{sh:+.2f}",
                    f"{dd:.0%}", f"{sig.mean():.0%}")
    console.print(tbl)
    console.print("[dim]edge is drawdown reduction (risk-off when supply contracts), "
                  "not linear return prediction. Validated across 2018 & 2022 bears.[/]")


@app.command()
def ledger(
    config: str = typer.Option("config.yaml"),
    testnet: bool = typer.Option(True, "--testnet/--mainnet", help="which venue's track record"),
    interval: int = typer.Option(3600, help="tick spacing (s) used to annualize the Sharpe"),
):
    """Show the LIVE track record + the pre-committed decision rule's verdict.

    This is the consolidation tool: it turns accumulated rebalances into a live
    out-of-sample Sharpe and judges it against the backtest — skill vs luck."""
    from autotrader.live.ledger import (BACKTEST_SHARPE, LIVE_SHARPE_FLOOR,
                                        MIN_REBALANCES, Ledger)

    cfg = _cfg(config)
    path = f"{cfg['data_cache_dir']}/ledger_{'testnet' if testnet else 'mainnet'}.jsonl"
    s = Ledger(path, interval_s=float(interval)).summary()
    console.rule(f"[bold]Live track record — {'testnet' if testnet else 'MAINNET'}[/]")
    if s["n_rebalances"] < 2:
        console.print(f"  rebalances logged : {s['n_rebalances']} (need ≥2)")
        console.print(f"  [dim]run `autotrader live --testnet --arm --loop` to accumulate samples.[/]")
        return
    console.print(f"  rebalances logged : {s['n_rebalances']} / {MIN_REBALANCES} for a verdict")
    console.print(f"  equity            : ${s['equity_start']:,.2f} → ${s['equity_now']:,.2f}  "
                  f"({s['total_return_pct']:+.2f}%)")
    color = "green" if (s["live_sharpe"] or 0) >= LIVE_SHARPE_FLOOR else "red"
    console.print(f"  live Sharpe       : [{color}]{s['live_sharpe']:.2f}[/]  "
                  f"(backtest {BACKTEST_SHARPE} · floor {LIVE_SHARPE_FLOOR})")
    console.print(f"  hit rate          : {s['hit_rate']:.0%}")
    console.print(f"  avg turnover/tick : {s['avg_turnover_ratio']:.0%} of equity")
    console.rule("[bold]Decision[/]")
    vcolor = "green" if "SURVIVED" in s["verdict"] else (
        "yellow" if ("BUILDING" in s["verdict"] or "MARGINAL" in s["verdict"]) else "red")
    console.print(f"  [{vcolor}]{s['verdict']}[/]")


@app.command()
def preflight(
    config: str = typer.Option("config.yaml"),
    live: bool = typer.Option(False, "--live", help="check live-trading prerequisites"),
):
    """Pre-flight checks before going live: connectivity, config sanity, keys."""
    import os

    cfg = _cfg(config)
    ok = True
    console.rule("[bold]Pre-flight[/]")
    # 1. exchange connectivity (public)
    try:
        import ccxt
        ex = getattr(ccxt, cfg["exchange"])({"enableRateLimit": True})
        px = ex.fetch_ticker("BTC/USDT")["last"]
        console.print(f"  [green]✓[/] exchange reachable ({cfg['exchange']}, BTC ${px:,.0f})")
    except Exception as e:
        ok = False
        console.print(f"  [red]✗[/] exchange unreachable: {e}")
    # 2. risk limits sane
    r = cfg.get("risk", {})
    if 0 < r.get("max_drawdown_pct", 0) <= 0.5 and r.get("max_gross_exposure", 0) > 0:
        console.print(f"  [green]✓[/] risk limits set (kill-switch −{r['max_drawdown_pct']:.0%})")
    else:
        ok = False
        console.print("  [red]✗[/] risk limits missing/insane — refusing to go live")
    # 3. live keys
    has_keys = bool(os.getenv("BINANCE_API_KEY") and os.getenv("BINANCE_SECRET"))
    if live:
        if has_keys:
            console.print("  [green]✓[/] API keys present (live possible)")
        else:
            ok = False
            console.print("  [red]✗[/] no BINANCE_API_KEY/SECRET — live not possible (good: stay on paper/testnet)")
        console.print("  [yellow]reminder: dry_run→testnet→tiny real→scale. Never skip a step.[/]")
    else:
        console.print("  [dim]paper mode — no keys needed. Use --live to check real-trading prereqs.[/]")
    console.print(f"\n[bold {'green' if ok else 'red'}]Pre-flight {'PASSED' if ok else 'FAILED'}[/]")


@app.command()
def mm(
    config: str = typer.Option("config.yaml"),
    symbol: list[str] = typer.Option(None, help="symbols to check (repeatable)"),
    maker_fee_bps: float = typer.Option(2.0, help="maker fee in bps (Binance futures base ~2.0; rebate tiers go negative)"),
):
    """Market-making (spread-capture) feasibility: live spread vs maker fee vs
    adverse selection. Shows whether a maker can even profit before latency."""
    from autotrader.analysis.market_making import measure_spread_bps, mm_economics, realized_vol_bps

    cfg = _cfg(config)
    syms = symbol or ["BTC/USDT", "SOL/USDT", "DOGE/USDT", "LINK/USDT", "ADA/USDT"]
    console.rule("[bold]Market-making economics (live spreads)[/]")
    tbl = Table(show_header=True, header_style="bold")
    for c in ("symbol", "spread bps", "half−fee", "1m vol bps", "breakeven avoidance", "verdict"):
        tbl.add_column(c)
    for s in syms:
        try:
            sp = measure_spread_bps(s, cfg["exchange"])
            vol = realized_vol_bps(s, "1m", 1000, cfg["exchange"])
            e = mm_economics(sp, vol, maker_fee_bps)
        except Exception as ex:
            console.print(f"[yellow]skip {s}: {ex}[/]")
            continue
        gm = e["gross_margin_bps"]
        if gm <= 0:
            verdict = "[red]impossible (fee > half-spread)[/]"
        elif e["breakeven_avoidance"] >= 0.99:
            verdict = "[red]needs ~HFT (>99% dodge)[/]"
        elif e["breakeven_avoidance"] >= 0.9:
            verdict = "[yellow]hard (>90% dodge)[/]"
        else:
            verdict = "[green]plausible regime[/]"
        tbl.add_row(s, f"{sp:.3f}", f"{gm:+.3f}", f"{vol:.1f}",
                    f"{e['breakeven_avoidance']:.1%}", verdict)
    console.print(tbl)
    console.print(
        "\n[dim]half−fee = what you keep from the spread after the maker fee, per fill. "
        "breakeven avoidance = fraction of adverse selection you must dodge (via latency/queue) "
        "just to break even. <90% = a real small-shop game; >99% = pure HFT; negative half−fee = "
        "the fee alone exceeds the spread, so retail makers lose before they start.[/]")
    console.print("[yellow]Honest caveat: a maker strategy CANNOT be trusted from a klines backtest "
                  "(fill assumptions lie) and testnet flow isn't real. This economics check is the "
                  "honest pre-filter; real validation needs recorded L2 + tiny live capital.[/]")


@app.command()
def strategies():
    """List the available strategy library and tunable params."""
    from autotrader.strategies.registry import all_strategies

    table = Table(title="Strategy library", header_style="bold")
    table.add_column("name")
    table.add_column("parameters")
    for name, cls in all_strategies().items():
        params = ", ".join(f"{k}{list(v[1:])}" for k, v in cls.param_space.items())
        table.add_row(name, params)
    console.print(table)


def _coerce(v: str):
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    return v


if __name__ == "__main__":
    app()
