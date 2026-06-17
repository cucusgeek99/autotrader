"""Cross-sectional (market-neutral) factor research.

Instead of timing one asset — where market beta dominates the noise — we rank a
signal *across* a universe of coins each period and go long the top / short the
bottom, dollar-neutral. Beta cancels; what's left is cross-sectional alpha.

Two big advantages over the single-asset work:
  * **Beta removed** — the long-short book is insensitive to whether crypto
    rallied or dumped; it only cares about *relative* ranking.
  * **Statistical power** — every (period × coin) pair is an observation, so the
    rank-IC t-stat is computed over hundreds of points, not 3–21 folds.

The canonical metric is the **rank information coefficient**: at each time, the
Spearman correlation across the universe between the signal and the forward
return. Its mean is skill; its t-stat (on *non-overlapping* periods, to avoid
autocorrelation inflation) is significance; mean/std is the information ratio.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from autotrader.features import microstructure as ms


# ---- panel construction -------------------------------------------------
def build_panels(dfs: dict) -> dict:
    """Align a dict of {symbol: extended OHLCV df} into time × symbol panels
    over their common date range."""
    closes = pd.DataFrame({s: d["close"] for s, d in dfs.items()}).dropna(how="any")
    idx = closes.index
    flow = pd.DataFrame(
        {s: ms.order_flow_imbalance(d, 6).reindex(idx) for s, d in dfs.items()
         if ms.has_flow(d)}
    )
    funding = pd.DataFrame(
        {s: d["funding"].reindex(idx) for s, d in dfs.items() if "funding" in d}
    )
    dollar_vol = pd.DataFrame({
        s: (d["quote_volume"] if "quote_volume" in d else d["close"] * d["volume"]).reindex(idx)
        for s, d in dfs.items()
    })
    return {"closes": closes, "flow": flow, "funding": funding, "dollar_vol": dollar_vol}


def liquidity_mask(dollar_vol: pd.DataFrame, top_n: int, lookback: int = 336) -> pd.DataFrame:
    """Boolean in-universe mask: keep the `top_n` coins by trailing mean dollar
    volume each bar. Ex-ante (uses only past volume) — a liquidity screen, not a
    performance screen, so no look-ahead / snooping."""
    rv = dollar_vol.rolling(lookback, min_periods=lookback // 2).mean()
    rank = rv.rank(axis=1, ascending=False)
    return rank <= top_n


def screen_signal(signal: pd.DataFrame, mask: pd.DataFrame) -> pd.DataFrame:
    """Set out-of-universe coins to NaN so to_weights drops them from the book."""
    return signal.where(mask.reindex_like(signal).fillna(False))


# ---- signals (each returns a time × symbol panel) -----------------------
def sig_momentum(closes: pd.DataFrame, lookback: int) -> pd.DataFrame:
    return closes.pct_change(lookback)


def sig_reversal(closes: pd.DataFrame, lookback: int) -> pd.DataFrame:
    return -closes.pct_change(lookback)


def sig_lowvol(closes: pd.DataFrame, lookback: int) -> pd.DataFrame:
    return -closes.pct_change().rolling(lookback, min_periods=lookback).std()


def sig_flow(flow: pd.DataFrame, lookback: int) -> pd.DataFrame:
    return flow.ewm(span=lookback, adjust=False, min_periods=lookback).mean()


def sig_funding(funding: pd.DataFrame, lookback: int) -> pd.DataFrame:
    # high funding = crowded long → expect relative underperformance → fade
    return -funding.rolling(lookback, min_periods=lookback // 2).mean()


def sig_lottery(closes: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Lottery / MAX effect: coins with extreme recent single-bar spikes attract
    lottery demand and subsequently underperform → short the high-MAX names."""
    ret = closes.pct_change()
    return -ret.rolling(lookback, min_periods=lookback).max()


def sig_longrev(closes: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Long-term reversal (De Bondt-Thaler): very slow — distinct horizon from
    short-term reversal, so it adds breadth rather than duplicating it."""
    return -closes.pct_change(lookback)


def sig_lowbeta(closes: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Betting-against-beta (Frazzini-Pedersen): long low-beta coins, short
    high-beta. Rolling beta to the equal-weight universe return."""
    ret = closes.pct_change()
    mkt = ret.mean(axis=1)
    w = dict(window=lookback, min_periods=lookback)
    exy = ret.mul(mkt, axis=0).rolling(**w).mean()
    ex = ret.rolling(**w).mean()
    my = mkt.rolling(**w).mean()
    cov = exy.sub(ex.mul(my, axis=0))
    var = mkt.rolling(**w).var()
    beta = cov.div(var, axis=0)
    return -beta


# ---- rank IC ------------------------------------------------------------
@dataclass
class ICReport:
    mean_ic: float
    ic_ir: float       # mean / std of the per-period IC (breadth-adjusted skill)
    t_stat: float      # on non-overlapping periods
    n_periods: int


def ic_series(signal: pd.DataFrame, closes: pd.DataFrame, horizon: int) -> pd.Series:
    """Per-period cross-sectional rank IC (Spearman across coins each bar)."""
    fwd = closes.pct_change(horizon).shift(-horizon)
    ics = []
    for t in signal.index:
        s, f = signal.loc[t], fwd.loc[t]
        valid = s.notna() & f.notna()
        if valid.sum() >= 4 and s[valid].std() > 0 and f[valid].std() > 0:
            ics.append(float(s[valid].rank().corr(f[valid].rank())))
        else:
            ics.append(np.nan)
    return pd.Series(ics, index=signal.index)


def rank_ic(signal: pd.DataFrame, closes: pd.DataFrame, horizon: int) -> ICReport:
    ics = ic_series(signal, closes, horizon).dropna()
    if len(ics) < 10:
        return ICReport(0.0, 0.0, 0.0, len(ics))
    mean_ic = float(ics.mean())
    ic_ir = float(mean_ic / ics.std()) if ics.std() else 0.0
    nonover = ics.iloc[::horizon]  # decorrelate overlapping forward returns
    n = len(nonover)
    se = nonover.std(ddof=1) / np.sqrt(n) if n > 1 else float("inf")
    t = float(nonover.mean() / se) if se not in (0.0, float("inf")) else 0.0
    return ICReport(round(mean_ic, 4), round(ic_ir, 3), round(t, 2), n)


# ---- long-short backtest ------------------------------------------------
def to_weights(signal: pd.DataFrame) -> pd.DataFrame:
    """Dollar-neutral factor weights: cross-sectionally demean each row, scale
    gross exposure to 1 (≈0.5 long / 0.5 short)."""
    demeaned = signal.sub(signal.mean(axis=1), axis=0)
    gross = demeaned.abs().sum(axis=1).replace(0.0, np.nan)
    return demeaned.div(gross, axis=0).fillna(0.0)


def standardize_xs(signal: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional z-score: demean and scale each row across coins so factors
    are on a common scale before combining."""
    mu = signal.mean(axis=1)
    sd = signal.std(axis=1).replace(0.0, np.nan)
    return signal.sub(mu, axis=0).div(sd, axis=0).fillna(0.0)


def combine_factors(signals: list, weights: list | None = None) -> pd.DataFrame:
    """Equal- (or custom-) weight combination of standardized factor panels.
    All panels must share index and columns (use closes-derived factors)."""
    zs = [standardize_xs(s) for s in signals]
    weights = weights or [1.0] * len(zs)
    composite = sum(w * z for w, z in zip(weights, zs))
    return composite


def factor_correlation(a: pd.DataFrame, b: pd.DataFrame) -> float:
    """Mean per-period cross-sectional rank correlation between two factors —
    low |corr| means they add breadth rather than duplicate each other."""
    corrs = []
    for t in a.index:
        x, y = a.loc[t], b.loc[t]
        valid = x.notna() & y.notna()
        if valid.sum() >= 4 and x[valid].std() > 0 and y[valid].std() > 0:
            corrs.append(float(x[valid].rank().corr(y[valid].rank())))
    return round(float(np.nanmean(corrs)), 3) if corrs else 0.0


@dataclass
class PortfolioStats:
    total_return: float
    sharpe: float
    max_drawdown: float
    ann_turnover: float
    n_bars: int


def apply_band(weights: pd.DataFrame, band: float) -> pd.DataFrame:
    """No-trade band: only move a coin's weight when its target shifts by more
    than `band`. A standard, powerful turnover-control lever — small target
    wiggles are ignored, so we don't pay fees chasing noise."""
    if band <= 0:
        return weights
    vals = weights.to_numpy()
    out = np.zeros_like(vals)
    cur = None
    for i in range(len(vals)):
        if cur is None:
            cur = vals[i].copy()
        else:
            row = vals[i]
            move = np.abs(row - cur) > band
            cur = cur.copy()
            cur[move] = row[move]
        out[i] = cur
    return pd.DataFrame(out, index=weights.index, columns=weights.columns)


def xs_net(
    signal: pd.DataFrame, closes: pd.DataFrame, rebalance: int,
    taker_fee: float, slippage: float, band: float = 0.0,
):
    """Per-bar net return series + turnover series for a cross-sectional book."""
    weights = to_weights(signal)
    if rebalance > 1:
        keep = pd.Series(np.arange(len(weights)) % rebalance == 0, index=weights.index)
        weights = weights.where(keep, other=np.nan).ffill().fillna(0.0)
    weights = apply_band(weights, band)
    held = weights.shift(1).fillna(0.0)            # enter next bar (no look-ahead)
    rets = closes.pct_change().fillna(0.0)
    port = (held * rets).sum(axis=1)
    turnover = held.diff().abs().sum(axis=1).fillna(0.0)
    net = port - turnover * (taker_fee + slippage)
    return net, turnover


def xs_backtest(
    signal: pd.DataFrame,
    closes: pd.DataFrame,
    rebalance: int,
    taker_fee: float,
    slippage: float,
    bars_per_year: float,
    band: float = 0.0,
) -> PortfolioStats:
    net, turnover = xs_net(signal, closes, rebalance, taker_fee, slippage, band)
    net = net.dropna()
    equity = (1 + net).cumprod()
    total = float(equity.iloc[-1] - 1) if len(equity) else 0.0
    mean, std = net.mean(), net.std()
    sharpe = float(mean / std * np.sqrt(bars_per_year)) if std > 0 else 0.0
    dd = float((equity / equity.cummax() - 1).min()) if len(equity) else 0.0
    ann_turn = float(turnover.mean() * bars_per_year)
    return PortfolioStats(round(total, 4), round(sharpe, 3), round(dd, 4),
                          round(ann_turn, 1), len(net))


# ---- the standard factor library ----------------------------------------
def factor_library(panels: dict, horizon: int) -> dict:
    """Map of factor name -> signal panel, using sensible default lookbacks."""
    closes, flow, funding = panels["closes"], panels["flow"], panels["funding"]
    lib = {
        "momentum": sig_momentum(closes, 48),
        "reversal": sig_reversal(closes, 6),
        "lowvol": sig_lowvol(closes, 48),
    }
    if not flow.empty:
        lib["orderflow"] = sig_flow(flow, 12)
    if not funding.empty:
        lib["funding_xs"] = sig_funding(funding, 168)
    return lib
