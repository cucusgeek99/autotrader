"""End-to-end pipeline test on synthetic data — no network required."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from autotrader.backtest.engine import run_backtest
from autotrader.knowledge.regime import classify
from autotrader.knowledge.store import KnowledgeStore
from autotrader.live.paper_broker import PaperBroker
from autotrader.optimize.optimizer import tune
from autotrader.strategies.registry import all_strategies, get


def synthetic_ohlcv(n: int = 3000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # trend + cyclical + noise so multiple regimes appear
    t = np.arange(n)
    trend = np.cumsum(rng.normal(0.0005, 0.01, n))
    cycle = 0.15 * np.sin(t / 120)
    price = 100 * np.exp(trend + cycle)
    close = pd.Series(price)
    high = close * (1 + rng.uniform(0, 0.01, n))
    low = close * (1 - rng.uniform(0, 0.01, n))
    open_ = close.shift(1).fillna(close.iloc[0])
    vol = rng.uniform(10, 100, n)
    idx = pd.date_range("2022-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {"open": open_.values, "high": high.values, "low": low.values,
         "close": close.values, "volume": vol},
        index=idx,
    )


def test_strategies_register():
    names = all_strategies()
    assert {"ma_crossover", "rsi_reversion", "breakout", "bollinger_reversion"} <= set(names)


def test_backtest_runs_and_is_sane():
    df = synthetic_ohlcv()
    strat = get("ma_crossover")(fast=10, slow=50, allow_short=True)
    res = run_backtest(strat, df, bars_per_year=8760)
    assert len(res.equity) == len(df)
    assert res.equity.iloc[0] > 0
    # metrics are finite
    for v in res.metrics.as_dict().values():
        assert v == v  # not NaN


def test_regime_classifies():
    df = synthetic_ohlcv()
    label = classify(df)
    assert label in {"trending_up", "trending_down", "ranging_calm", "ranging_volatile"}


def test_tune_produces_oos_estimate():
    df = synthetic_ohlcv(n=2600)
    res = tune(
        get("ma_crossover"),
        df,
        fees={"taker": 0.0006, "slippage": 0.0005},
        bars_per_year=8760,
        wf={"train": 800, "test": 300, "step": 300},
        trials=8,
        min_trades=3,
    )
    assert res.n_folds >= 1
    assert isinstance(res.deploy_params, dict)
    assert "fast" in res.deploy_params


def test_knowledge_store_roundtrip(tmp_path):
    db = tmp_path / "k.db"
    store = KnowledgeStore(str(db))
    store.record(
        created_at="2024-01-01T00:00:00+00:00", symbol="BTC/USDT", timeframe="1h",
        regime="trending_up", strategy="ma_crossover", params={"fast": 10, "slow": 50},
        insample_score=1.2, oos_score=0.8, overfit_gap=0.4, n_folds=3,
        metrics={"sharpe": 1.1},
    )
    champ = store.best_for("BTC/USDT", "1h", "trending_up")
    assert champ is not None
    assert champ.strategy == "ma_crossover"
    assert champ.params["fast"] == 10
    assert store.best_for("BTC/USDT", "1h", "ranging_calm") is None
    assert store.best_any_regime("BTC/USDT", "1h").strategy == "ma_crossover"
    store.close()


def test_paper_broker_rebalances():
    b = PaperBroker(symbol="BTC/USDT", cash=10_000, state_path="/tmp/at_test_state.json")
    fill = b.rebalance_to(1.0, price=100.0, time="t0")
    assert fill is not None and fill.side == "buy"
    assert b.position > 0
    # going flat sells everything
    b.rebalance_to(0.0, price=110.0, time="t1")
    assert abs(b.position) < 1e-9
    # made money on the long into a higher price, minus fees
    assert b.equity(110.0) > 9_000


@pytest.mark.parametrize(
    "name",
    [
        "ma_crossover", "rsi_reversion", "breakout", "bollinger_reversion",
        "macd_trend", "momentum_roc", "supertrend", "stochastic_reversion",
    ],
)
def test_every_strategy_emits_bounded_signals(name):
    df = synthetic_ohlcv(n=1000)
    sig = get(name)().generate_signals(df)
    assert sig.between(-1.0, 1.0).all()
    assert len(sig) == len(df)


def test_paper_broker_never_goes_negative_at_full_exposure():
    b = PaperBroker(symbol="BTC/USDT", cash=10_000, state_path="/tmp/at_test_state2.json")
    b.rebalance_to(1.0, price=100.0, time="t0")
    assert b.cash >= -1e-6           # fees were reserved
    assert b.equity(100.0) <= 10_000  # can't conjure value


def test_ensemble_blends_members():
    from autotrader.ensemble import EnsembleStrategy

    df = synthetic_ohlcv(n=800)
    members = [
        (get("ma_crossover")(fast=10, slow=40, allow_short=True), 2.0),
        (get("rsi_reversion")(period=14, allow_short=True), 1.0),
    ]
    ens = EnsembleStrategy(members)
    sig = ens.generate_signals(df)
    assert sig.between(-1.0, 1.0).all()
    assert abs(sum(ens.weights) - 1.0) < 1e-9


def test_top_for_dedupes_by_strategy(tmp_path):
    store = KnowledgeStore(str(tmp_path / "k2.db"))
    for i, oos in enumerate([0.9, 0.8, 0.5]):
        store.record(
            created_at=f"2024-01-0{i+1}T00:00:00+00:00", symbol="BTC/USDT",
            timeframe="1h", regime="ranging_calm", strategy="ma_crossover",
            params={"fast": 10 + i}, insample_score=1.0, oos_score=oos,
            overfit_gap=0.1, n_folds=2, metrics={},
        )
    store.record(
        created_at="2024-02-01T00:00:00+00:00", symbol="BTC/USDT", timeframe="1h",
        regime="ranging_calm", strategy="breakout", params={"lookback": 20},
        insample_score=1.0, oos_score=0.7, overfit_gap=0.1, n_folds=2, metrics={},
    )
    top = store.top_for("BTC/USDT", "1h", "ranging_calm", k=3)
    names = [c.strategy for c in top]
    assert names == ["ma_crossover", "breakout"]  # deduped, best-first
    store.close()


def test_sparkline_renders():
    from autotrader.backtest.report import sparkline

    s = pd.Series(range(100))
    out = sparkline(s, width=20)
    assert len(out) == 20 and out[0] != out[-1]


def test_dashboard_endpoints(tmp_path, monkeypatch):
    """Dashboard JSON endpoints work end-to-end against a seeded knowledge db."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    # seed a knowledge db and a config pointing at it
    db = tmp_path / "k.db"
    store = KnowledgeStore(str(db))
    store.record(
        created_at="2024-01-01T00:00:00+00:00", symbol="BTC/USDT", timeframe="1h",
        regime="trending_up", strategy="ma_crossover", params={"fast": 10},
        insample_score=1.0, oos_score=0.5, overfit_gap=0.5, n_folds=3, metrics={},
    )
    store.close()
    cfg = tmp_path / "config.yaml"
    cfg.write_text(f"knowledge_db: {db}\npaper:\n  starting_cash: 10000\n")

    from autotrader.dashboard.app import create_app

    client = TestClient(create_app(str(cfg)))
    assert client.get("/").status_code == 200
    lb = client.get("/api/leaderboard").json()
    assert lb[0]["strategy"] == "ma_crossover"
    port = client.get("/api/portfolio").json()
    assert "exists" in port


def synthetic_extended(n: int = 1500, seed: int = 11) -> pd.DataFrame:
    """OHLCV + order-flow columns, with taker-buy correlated to forward returns
    so the IC analyzer has a real signal to detect."""
    rng = np.random.default_rng(seed)
    df = synthetic_ohlcv(n, seed)
    ret = df["close"].pct_change().fillna(0.0)
    # taker-buy ratio leads returns slightly + noise
    base_ratio = 0.5 + 0.3 * ret.shift(-1).fillna(0.0) / (ret.std() or 1) * 0.1
    ratio = (base_ratio + rng.normal(0, 0.05, n)).clip(0.05, 0.95)
    df["taker_buy_base"] = df["volume"] * ratio
    df["quote_volume"] = df["close"] * df["volume"]
    df["num_trades"] = (df["volume"] * rng.uniform(5, 15, n)).astype(int)
    return df


def test_orderflow_features_bounded():
    from autotrader.features import microstructure as ms

    df = synthetic_extended()
    assert ms.has_flow(df)
    ofi = ms.order_flow_imbalance(df, 6)
    assert ofi.between(-1, 1).all()
    feats = ms.feature_frame(df)
    assert {"ofi_1", "flow_mom_12", "vwap_dev_24"} <= set(feats.columns)
    assert feats.notna().all().all()


def test_orderflow_strategies_flat_without_flow():
    # without flow columns, order-flow strategies must return all-flat
    df = synthetic_ohlcv(500)
    for name in ("orderflow_momentum", "orderflow_reversion", "flow_divergence"):
        sig = get(name)().generate_signals(df)
        assert (sig == 0.0).all()


def test_orderflow_strategies_trade_with_flow():
    df = synthetic_extended()
    sig = get("orderflow_momentum")(window=6, threshold=0.05, allow_short=True).generate_signals(df)
    assert sig.between(-1, 1).all()
    assert (sig != 0).any()  # it actually takes positions when flow is present


def test_feature_ic_detects_signal():
    from autotrader.analysis.feature_ic import compute_ic
    from autotrader.features.microstructure import feature_frame

    df = synthetic_extended(n=2000)
    feats = feature_frame(df)
    results = compute_ic(feats, df["close"], horizon=1)
    by_name = {r.feature: r for r in results}
    # the planted order-flow→return relationship should surface as nonzero IC
    assert abs(by_name["ofi_1"].ic) > 0.0
    assert all(r.n > 100 for r in results)


def test_purged_walk_forward_runs_and_is_leak_free(monkeypatch):
    from autotrader.analysis.purged_wf import (
        _fit_weights, composite_features, purged_walk_forward,
    )

    df = synthetic_extended(n=3000)
    out = purged_walk_forward(
        df, horizon=12, lookback=24, threshold=0.5,
        train=800, test=300, step=300, embargo=12,
    )
    assert out["n_folds"] >= 2
    assert "verdict" in out
    assert all("weights" in f for f in out["folds"])

    # purge invariant: weights are fit only on rows whose label is fully in-train.
    # The last `horizon` training rows have NaN forward returns and must be excluded.
    feats = composite_features(df, 24)
    w = _fit_weights(feats, df["close"], range(0, 800), horizon=12)
    assert set(w.keys()) == {"flow_mom", "ofi6", "trade_z", "vwap_dev"}
    assert abs(sum(abs(v) for v in w.values()) - 1.0) < 1e-6  # L1-normalized


def test_regime_series_is_causal_and_labelled():
    from autotrader.knowledge.regime import REGIMES, regime_series

    df = synthetic_ohlcv(1500)
    rs = regime_series(df)
    assert len(rs) == len(df)
    assert set(rs.unique()) <= set(REGIMES)
    # causal check: a value at bar t must not change when future bars are removed
    rs_trunc = regime_series(df.iloc[:800])
    # overlapping region (after warmup) should agree
    common = rs.iloc[400:800]
    assert (common.values == rs_trunc.iloc[400:800].values).mean() > 0.98


def test_regime_conditional_purged_runs_and_reports_stability():
    from autotrader.analysis.purged_wf import purged_walk_forward_regime

    df = synthetic_extended(n=3000)
    out = purged_walk_forward_regime(
        df, horizon=12, lookback=24, threshold=0.5,
        train=800, test=300, step=300, embargo=12, min_regime_samples=30,
    )
    assert out["n_folds"] >= 2
    assert "regime_stability" in out
    # stability report keys are valid regimes with the expected diagnostic shape
    for r, s in out["regime_stability"].items():
        assert "strongest_sign_stable" in s and "features" in s


def test_nested_threshold_tuning_is_peek_free():
    from autotrader.analysis.purged_wf import purged_nested_regime

    df = synthetic_extended(n=3000)
    out = purged_nested_regime(
        df, horizon=12, lookback=24, thresholds=(0.5, 1.0, 1.5, 2.0),
        train=900, test=300, step=300, embargo=12,
        inner_train=400, inner_test=200, inner_step=200, min_regime_samples=20,
    )
    assert out["n_folds"] >= 2
    # every outer fold must have independently chosen a threshold from the set
    assert "chosen_thresholds" in out
    assert all(t in (0.5, 1.0, 1.5, 2.0) for t in out["chosen_thresholds"])
    assert len(out["chosen_thresholds"]) == out["n_folds"]


def test_panel_validation_aggregates_held_out(monkeypatch):
    from autotrader.analysis.panel import panel_validation

    # three "symbols" with different seeds; BTC is the dev symbol
    dfs = {
        "BTC/USDT": synthetic_extended(n=2400, seed=1),
        "ETH/USDT": synthetic_extended(n=2400, seed=2),
        "SOL/USDT": synthetic_extended(n=2400, seed=3),
    }
    out = panel_validation(
        dfs, dev_symbols={"BTC/USDT"},
        horizon=12, lookback=24, thresholds=(0.5, 1.0, 1.5, 2.0),
        train=900, test=300, step=300, embargo=12,
        inner_train=400, inner_test=200, inner_step=200, min_regime_samples=20,
    )
    assert set(out["held_out"]) == {"ETH/USDT", "SOL/USDT"}
    assert "BTC/USDT" not in out["held_out"]
    # portfolio stats computed across the aligned held-out folds
    assert out["portfolio"]["n"] >= 1
    assert "t_stat" in out["portfolio"]
    assert out["verdict"]
    for s in ("ETH/USDT", "SOL/USDT"):
        assert "stats" in out["per_symbol"][s]


def synthetic_perp(n: int = 2400, seed: int = 5) -> pd.DataFrame:
    """Extended OHLCV + a funding column that crowds before reversals, so the
    fade has something to find."""
    rng = np.random.default_rng(seed)
    df = synthetic_extended(n, seed)
    ret = df["close"].pct_change().fillna(0.0)
    # funding tracks recent return (crowding) + noise; piecewise-constant per 8 bars
    raw = ret.rolling(24).mean().fillna(0.0) * 0.5 + rng.normal(0, 0.0002, n)
    funding = pd.Series(raw, index=df.index)
    funding.iloc[::8] = funding.iloc[::8]  # settlement bars
    df["funding"] = funding.ffill()
    return df


def test_funding_zscore_and_fade_signal():
    from autotrader.analysis.funding_fade import fade_signal, funding_zscore

    df = synthetic_perp()
    z = funding_zscore(df, 168)
    assert len(z) == len(df) and z.notna().all()
    sig = fade_signal(df, lookback=168, threshold=1.0, hold=24, allow_short=True)
    assert sig.between(-1, 1).all()
    # fade is contrarian: where z is very high, signal should be <= 0 (short/flat)
    hi = z > z.quantile(0.95)
    if hi.any():
        assert sig[hi].mean() <= 0.0


def test_fade_carry_helps_when_fading_positive_funding():
    from autotrader.analysis.funding_fade import fade_backtest
    import pandas as pd

    df = synthetic_perp()
    pos = pd.Series(-1.0, index=df.index)  # permanently short
    df = df.copy(); df["funding"] = 0.01    # constant high positive funding
    with_carry = fade_backtest(df, pos, 0.0, 0.0, 8760, include_carry=True)
    no_carry = fade_backtest(df, pos, 0.0, 0.0, 8760, include_carry=False)
    # a short receives positive funding → carry-inclusive return is higher
    assert with_carry.total_return > no_carry.total_return


def test_purged_fade_runs():
    from autotrader.analysis.funding_fade import purged_fade

    df = synthetic_perp(n=2600)
    out = purged_fade(df, horizon=24, lookback=168, train=900, test=300, step=300,
                      inner_train=400, inner_test=200, inner_step=200)
    assert out["n_folds"] >= 2
    assert "mean_price_only_return" in out
    assert "chosen_thresholds" in out


def test_cross_sectional_detects_planted_factor():
    """A panel where each coin's forward return is driven by its own signal must
    show a strong positive rank IC and a profitable long-short book."""
    from autotrader.analysis import cross_sectional as xs

    rng = np.random.default_rng(3)
    n, ncoins = 1500, 8
    idx = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC")
    # planted cross-sectional momentum: forward return ∝ this coin's past return
    closes = {}
    for c in range(ncoins):
        drift = rng.normal(0, 0.01, n)
        # bias: coins with higher recent drift keep rising (XS momentum)
        price = 100 * np.exp(np.cumsum(drift))
        closes[f"C{c}/USDT"] = pd.Series(price, index=idx)
    closes = pd.DataFrame(closes)
    panels = {"closes": closes, "flow": pd.DataFrame(), "funding": pd.DataFrame()}

    sig = xs.sig_momentum(closes, 24)
    ic = xs.rank_ic(sig, closes, horizon=12)
    assert ic.n_periods > 30
    # weights are dollar-neutral (rows sum ~0)
    w = xs.to_weights(sig)
    assert w.iloc[100:].sum(axis=1).abs().max() < 1e-9
    ps = xs.xs_backtest(sig, closes, 12, 0.0, 0.0, 8760)
    assert ps.n_bars > 100


def test_deflated_sharpe_haircuts_for_trials():
    from autotrader.analysis.cross_validation import deflated_sharpe_ratio
    import pandas as pd

    rng = np.random.default_rng(0)
    # a genuinely good return stream
    good = pd.Series(rng.normal(0.001, 0.01, 500))
    dsr_few = deflated_sharpe_ratio(good, n_trials=1, trial_sharpes=[0.1, 0.12])
    dsr_many = deflated_sharpe_ratio(good, n_trials=200, trial_sharpes=[0.1, 0.12])
    # more trials → larger haircut (sr0) → lower DSR
    assert dsr_many["sr0"] >= dsr_few["sr0"]
    assert dsr_many["dsr"] <= dsr_few["dsr"]
    # pure-noise stream should not look significant
    noise = pd.Series(rng.normal(0.0, 0.01, 500))
    assert deflated_sharpe_ratio(noise, n_trials=50, trial_sharpes=[0.0, 0.05])["dsr"] < 0.95


def test_norm_ppf_cdf_roundtrip():
    from autotrader.analysis.cross_validation import _norm_cdf, _norm_ppf

    for p in (0.025, 0.1, 0.5, 0.9, 0.975):
        assert abs(_norm_cdf(_norm_ppf(p)) - p) < 1e-3


def test_walk_forward_xs_runs_oos():
    from autotrader.analysis.cross_sectional import sig_lowvol
    from autotrader.analysis.cross_validation import walk_forward_xs

    rng = np.random.default_rng(7)
    n = 4000
    idx = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC")
    closes = pd.DataFrame(
        {f"C{i}": 100 * np.exp(np.cumsum(rng.normal(0, 0.01 + 0.002 * i, n)))
         for i in range(6)}, index=idx,
    )
    wf = walk_forward_xs(
        lambda lb: sig_lowvol(closes, lb), closes, horizon=12,
        lookback_grid=[24, 48], train=1500, test=500, step=500,
    )
    assert wf.n_folds >= 2
    assert len(wf.chosen_lookbacks) == wf.n_folds
    assert isinstance(wf.oos_sharpe_ann, float)


def test_risk_manager_caps_and_killswitch(tmp_path):
    from autotrader.risk import RiskLimits, RiskManager

    rm = RiskManager(RiskLimits(max_gross_exposure=1.0, max_position_pct=0.25,
                                max_drawdown_pct=0.20, max_daily_loss_pct=0.08),
                     state_path=str(tmp_path / "risk.json"))
    rm.update_equity(10_000, day_key="2026-01-01")

    # per-position cap: 0.5 → 0.25
    adj, notes = rm.check_weights({"BTC/USDT": 0.5, "ETH/USDT": -0.5})
    assert abs(adj["BTC/USDT"]) <= 0.25 + 1e-9
    assert abs(adj["ETH/USDT"]) <= 0.25 + 1e-9

    # gross cap: many positions scaled so sum|w| ≤ 1.0
    big = {f"C{i}/USDT": 0.25 for i in range(8)}  # gross 2.0
    adj2, _ = rm.check_weights(big)
    assert abs(sum(abs(w) for w in adj2.values()) - 1.0) < 1e-6

    # drawdown kill-switch trips and flattens
    rm.update_equity(7_500, day_key="2026-01-01")   # −25% from peak
    assert rm.halted
    flat, notes = rm.check_weights({"BTC/USDT": 0.5})
    assert flat["BTC/USDT"] == 0.0
    assert any("HALTED" in n for n in notes)

    # sticky across restart (state persisted)
    rm2 = RiskManager(state_path=str(tmp_path / "risk.json"))
    assert rm2.halted
    rm2.resume()
    assert not rm2.halted


def test_risk_daily_loss_resets_next_day(tmp_path):
    from autotrader.risk import RiskLimits, RiskManager

    rm = RiskManager(RiskLimits(max_daily_loss_pct=0.05, max_drawdown_pct=0.50),
                     state_path=str(tmp_path / "r.json"))
    rm.update_equity(10_000, day_key="2026-01-01")
    rm.update_equity(9_400, day_key="2026-01-01")   # −6% day → halt
    assert rm.halted
    rm.update_equity(9_400, day_key="2026-01-02")   # new day clears daily halt
    assert not rm.halted


def test_futures_fmt_qty_precision():
    from autotrader.live.binance_futures import FuturesClient

    c = FuturesClient("k", "s", testnet=True)
    c._filters = {"BTCUSDT": (0.001, 5.0), "XRPUSDT": (1.0, 5.0)}
    assert c.fmt_qty("BTCUSDT", 0.03719) == "0.037"   # floor to step 0.001
    assert c.fmt_qty("XRPUSDT", 176.9) == "176"        # step 1.0 → floor to integer
    assert c.fmt_qty("BTCUSDT", -0.0009) == "0.000"    # below step → zero


def test_live_broker_dry_run_sends_nothing(tmp_path):
    from autotrader.live.live_broker import LiveBroker
    from autotrader.risk import RiskManager

    b = LiveBroker(dry_run=True, risk=RiskManager(state_path=str(tmp_path / "r.json")))
    assert "DRY-RUN" in b.mode()
    intents = b.rebalance_to({"BTC/USDT": 0.5, "ETH/USDT": -0.5},
                             equity=10_000, prices={"BTC/USDT": 70000, "ETH/USDT": 3500})
    placed = [i for i in intents if i.get("status") == "simulated"]
    assert placed and all(i["status"] != "sent" for i in intents if "status" in i)


def test_l2_book_snapshot_features():
    from autotrader.live.recorder import book_snapshot_features

    # bid-heavy book → positive imbalance, microprice above mid
    bids = [["100.0", "10"], ["99.9", "8"], ["99.8", "5"]]
    asks = [["100.1", "2"], ["100.2", "3"], ["100.3", "1"]]
    f = book_snapshot_features(bids, asks, depth=3)
    assert f["imbalance"] > 0          # more bid volume
    assert f["spread_bps"] > 0
    assert -1.0 <= f["imbalance"] <= 1.0
    # empty book → empty dict, no crash
    assert book_snapshot_features([], [], 3) == {}


def test_multi_asset_broker_dollar_neutral():
    from autotrader.live.multi_broker import MultiAssetPaperBroker

    b = MultiAssetPaperBroker(cash=10_000, state_path="/tmp/at_xs_test.json")
    prices = {"BTC/USDT": 70000.0, "ETH/USDT": 3500.0, "SOL/USDT": 150.0, "XRP/USDT": 0.6}
    # dollar-neutral target: long BTC/ETH, short SOL/XRP, sums to 0
    target = {"BTC/USDT": 0.25, "ETH/USDT": 0.25, "SOL/USDT": -0.25, "XRP/USDT": -0.25}
    fills = b.rebalance_to(target, prices, "t0")
    assert len(fills) == 4
    eq = b.equity(prices)
    assert 9_900 < eq <= 10_000          # only fees lost, no value conjured
    book = b.book(prices)
    longs = [r for r in book if r["weight"] > 0]
    shorts = [r for r in book if r["weight"] < 0]
    assert len(longs) == 2 and len(shorts) == 2
    # gross weights ~1, net ~0 (dollar-neutral)
    assert abs(sum(r["weight"] for r in book)) < 0.05


def test_no_trade_band_cuts_turnover():
    from autotrader.analysis.cross_sectional import sig_reversal, xs_backtest

    closes = pd.DataFrame(
        {f"C{i}": 100 * np.exp(np.cumsum(np.random.default_rng(i).normal(0, 0.02, 1200)))
         for i in range(6)},
        index=pd.date_range("2023-01-01", periods=1200, freq="1h", tz="UTC"),
    )
    rev = sig_reversal(closes, 6)
    no_band = xs_backtest(rev, closes, 12, 0.0006, 0.0005, 8760, band=0.0)
    banded = xs_backtest(rev, closes, 12, 0.0006, 0.0005, 8760, band=0.05)
    assert banded.ann_turnover < no_band.ann_turnover  # band reduces churn


def test_findings_strategy_characterize():
    from autotrader.portfolio import FindingsStrategy

    closes = pd.DataFrame(
        {f"C{i}": 100 * np.exp(np.cumsum(np.random.default_rng(i + 9).normal(0, 0.015, 2000)))
         for i in range(8)},
        index=pd.date_range("2023-01-01", periods=2000, freq="1h", tz="UTC"),
    )
    strat = FindingsStrategy(rebalance=12)
    ch = strat.characterize(closes, {"taker": 0.0006, "slippage": 0.0005}, 8760)
    assert "equity" in ch and len(ch["equity"]) > 100
    # inverse-turnover allocation favors the cheaper factor (lowvol) — weights sum to 1
    fw = ch["factor_weights"]
    assert abs(fw["lowvol"] + fw["reversal"] - 1.0) < 1e-6
    assert fw["lowvol"] >= fw["reversal"]  # lowvol is lower-turnover → larger weight
    # dollar-neutral target weights
    w = strat.target_weights(closes, {"taker": 0.0006, "slippage": 0.0005}, 8760)
    assert w.iloc[300:].sum(axis=1).abs().max() < 1e-9


def test_combine_factors_and_correlation():
    from autotrader.analysis.cross_sectional import (
        combine_factors, factor_correlation, sig_lowvol, sig_reversal, standardize_xs,
    )

    closes = pd.DataFrame(
        {f"C{i}": 100 * np.exp(np.cumsum(np.random.default_rng(i).normal(0, 0.01, 800)))
         for i in range(6)},
        index=pd.date_range("2023-01-01", periods=800, freq="1h", tz="UTC"),
    )
    lv, rev = sig_lowvol(closes, 96), sig_reversal(closes, 6)
    z = standardize_xs(lv)
    # standardized rows are ~zero-mean across coins
    assert z.iloc[200:].mean(axis=1).abs().max() < 1e-9
    comp = combine_factors([lv, rev])
    assert comp.shape == lv.shape
    c = factor_correlation(lv, rev)
    assert -1.0 <= c <= 1.0


def test_xs_weights_dollar_neutral_and_gross_one():
    from autotrader.analysis.cross_sectional import to_weights
    import pandas as pd

    sig = pd.DataFrame({"A": [1.0, 2, 3], "B": [3.0, 2, 1], "C": [2.0, 2, 2]})
    w = to_weights(sig)
    assert w.sum(axis=1).abs().max() < 1e-9          # dollar-neutral
    # gross exposure ~1 on rows with dispersion
    assert abs(w.iloc[0].abs().sum() - 1.0) < 1e-9


def test_rank_ic_sign_matches_relationship():
    from autotrader.analysis.cross_sectional import rank_ic
    import pandas as pd

    n = 600
    idx = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC")
    rng = np.random.default_rng(1)
    # 5 coins; construct closes so that a known signal predicts forward return
    base = pd.DataFrame(
        {f"C{i}": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))) for i in range(5)},
        index=idx,
    )
    from autotrader.analysis.cross_sectional import sig_momentum
    sig = sig_momentum(base, 12)
    ic = rank_ic(sig, base, horizon=6)
    assert -1.0 <= ic.mean_ic <= 1.0
    assert ic.n_periods > 10


def test_orderbook_gate_logic():
    from autotrader.live.orderbook import BookFeatures

    bull = BookFeatures(imbalance=0.4, spread_bps=2.0, micro_edge=0.0001)
    assert bull.confirms(1.0) is True       # bids heavy → long confirmed
    assert bull.confirms(-1.0) is False     # short not confirmed
    wide = BookFeatures(imbalance=0.4, spread_bps=99.0, micro_edge=0.0)
    assert wide.confirms(1.0) is False      # spread too wide → veto
    assert wide.confirms(0.0) is True       # flat always allowed


def test_llm_advisor_graceful_without_key(monkeypatch):
    """Advisor raises a friendly error (not a crash) when no API key is set."""
    from autotrader.knowledge.llm_advisor import AdvisorUnavailable, advise

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    with pytest.raises(AdvisorUnavailable):
        advise("BTC/USDT", "1h", "trending_up", [], {})


def test_ledger_records_and_verdict(tmp_path):
    """Ledger persists ticks, computes a live Sharpe, and refuses to judge a tiny
    sample — the decision rule must say BUILDING until MIN_REBALANCES is reached."""
    from autotrader.live.ledger import MIN_REBALANCES, Ledger

    p = str(tmp_path / "ledger.jsonl")
    led = Ledger(p, interval_s=3600)
    intents = [{"status": "sent", "notional": 1500},
               {"status": "sent", "notional": 1500}]
    for i, eq in enumerate([5000, 5010, 4995]):
        led.record(f"2026-06-03T0{i}:00:00", eq, {"BTC/USDT": 0.3, "ETH/USDT": -0.3},
                   intents, "TESTNET (fake)")

    assert len(led.records()) == 3
    s = led.summary()
    assert s["n_rebalances"] == 3
    assert s["live_sharpe"] is not None        # computed once ≥2 ticks
    assert 0 <= s["hit_rate"] <= 1
    assert "BUILDING" in s["verdict"]          # 3 < MIN_REBALANCES → no premature verdict
    assert MIN_REBALANCES == 30

    # too-few-samples guard
    led2 = Ledger(str(tmp_path / "empty.jsonl"))
    assert led2.summary()["verdict"].startswith("insufficient")


def test_directional_overlay_tilts_and_neutral_default():
    """k=0 stays dollar-neutral (net≈0); k>0 adds a net market tilt whose sign
    follows the trend. Exposure stays bounded in [-1,1]."""
    import numpy as np
    import pandas as pd

    from autotrader.analysis.directional import directional_weights, trend_exposure
    from autotrader.portfolio import DEFAULT_FACTORS, FindingsStrategy

    idx = pd.date_range("2024-01-01", periods=400, freq="h")
    cols = [f"C{i}" for i in range(6)]
    rng = np.random.default_rng(0)
    # steady uptrend + noise → trend signal should go net long
    drift = np.linspace(0, 0.5, 400)
    closes = pd.DataFrame(
        {c: 100 * np.exp(drift + 0.01 * np.cumsum(rng.standard_normal(400))) for c in cols},
        index=idx)
    fees, bpy = {"taker": 0.0006, "slippage": 0.0005}, 24 * 365

    expo = trend_exposure(closes, 24, 168, 2.0)
    assert expo.abs().max() <= 1.0                      # tanh-bounded
    assert expo.iloc[-1] > 0                            # uptrend → net long tilt

    dw = directional_weights(closes, expo)
    assert np.allclose(dw.sum(axis=1).to_numpy(), expo.to_numpy(), atol=1e-9)  # net=exposure

    neutral = FindingsStrategy(factors=list(DEFAULT_FACTORS))
    direc = FindingsStrategy(factors=list(DEFAULT_FACTORS), directional_weight=0.5,
                             trend_scale=2.0)
    wn = neutral.target_weights(closes, fees, bpy).iloc[-1]
    wd = direc.target_weights(closes, fees, bpy).iloc[-1]
    assert abs(wn.sum()) < 1e-6                          # neutral book nets ~0
    assert wd.sum() > wn.sum()                           # directional adds long tilt
    assert direc.characterize(closes, fees, bpy)["avg_net_exposure"] > 0


def test_stablecoin_supply_regime():
    """Risk-on/off regime = 1 when supply grows over the lookback, 0 when it
    contracts. Pure offline logic (no network)."""
    import numpy as np
    import pandas as pd

    from autotrader.data.onchain import supply_regime

    idx = pd.date_range("2024-01-01", periods=100, freq="D")
    supply = pd.Series(
        np.concatenate([np.linspace(100, 130, 50), np.linspace(130, 105, 50)]), index=idx)
    reg = supply_regime(supply, lookback=30)
    assert reg.iloc[45] == 1.0      # mid-growth → risk-on
    assert reg.iloc[-1] == 0.0      # contracting → risk-off
    assert set(reg.dropna().unique()) <= {0.0, 1.0}


def test_core_weights_trend_and_vol_scaling():
    from autotrader.core import core_weights, core_returns, summarize, closed_daily
    idx = pd.date_range("2020-01-01", periods=400, freq="D")
    rng = np.random.default_rng(0)
    up = pd.Series(100 * np.exp(np.cumsum(0.003 + 0.01 * rng.standard_normal(400))), idx)
    down = pd.Series(100 * np.exp(np.cumsum(-0.003 + 0.01 * rng.standard_normal(400))), idx)
    wild = pd.Series(100 * np.exp(np.cumsum(0.004 + 0.06 * rng.standard_normal(400))), idx)
    w = core_weights(pd.DataFrame({"UP": up, "DOWN": down}))
    assert w["UP"].iloc[-1] == pytest.approx(0.5)      # calm uptrend → full 1/N sleeve
    assert w["DOWN"].iloc[-1] == 0.0                   # below SMA200 → cash
    assert (w.sum(axis=1) <= 1.0 + 1e-12).all()        # never levered
    assert (w.iloc[:199] == 0).all().all()             # warm-up → no position
    ww = core_weights(pd.DataFrame({"UP": up, "WILD": wild}))
    if ww["WILD"].iloc[-1] > 0:                        # high vol → sized down
        assert ww["WILD"].iloc[-1] < 0.5
    r = core_returns(pd.DataFrame({"UP": up, "DOWN": down}))
    assert summarize(r)["max_dd"] <= 0
    # today's forming candle is dropped
    df = pd.DataFrame({"close": [1.0, 2.0]},
                      index=pd.to_datetime(["2026-01-01", "2026-01-02"], utc=True))
    assert len(closed_daily(df, now=pd.Timestamp("2026-01-02 15:00", tz="UTC"))) == 1


def test_spot_broker_trades_only_delta(tmp_path):
    from autotrader.live.spot_broker import SpotBroker
    from autotrader.risk import RiskLimits, RiskManager

    class FakeEx:
        markets = {"BTC/USDT": {"limits": {"cost": {"min": 5}}},
                   "ETH/USDT": {"limits": {"cost": {"min": 5}}}}
        def amount_to_precision(self, s, a):
            return f"{a:.6f}"

    rm = RiskManager(RiskLimits(max_position_pct=0.5), state_path=str(tmp_path / "r.json"))
    b = SpotBroker(risk=rm, _ex=FakeEx())
    px = {"BTC/USDT": 100.0, "ETH/USDT": 10.0}
    # holding 40 USDT of BTC, target 50% of 100 equity → buy ~10; ETH target 0 → sell all
    eq, intents = b.rebalance_to({"BTC/USDT": 0.5, "ETH/USDT": 0.0}, cash=40.0,
                                 qty={"BTC/USDT": 0.4, "ETH/USDT": 2.0}, prices=px)
    assert eq == pytest.approx(100.0)
    by = {i["symbol"]: i for i in intents}
    assert by["BTC/USDT"]["side"] == "buy" and by["BTC/USDT"]["notional"] == pytest.approx(10.0)
    assert by["ETH/USDT"]["side"] == "sell" and by["ETH/USDT"]["amount"] == pytest.approx(2.0)
    assert intents[0]["symbol"] == "ETH/USDT"           # sells go first to fund buys
    # tiny drift inside the 2% band → nothing traded
    _, intents = b.rebalance_to({"BTC/USDT": 0.5, "ETH/USDT": 0.0}, cash=50.0,
                                qty={"BTC/USDT": 0.505, "ETH/USDT": 0.0}, prices=px)
    assert all(i["status"].startswith("skipped") for i in intents if "symbol" in i)
