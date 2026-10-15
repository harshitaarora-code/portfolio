"""Quantitative engine. Pure functions on pandas/numpy; no network access.

Conventions: simple (arithmetic) daily returns on adjusted Close; 252 trading days/year; VaR/CVaR are reported as
POSITIVE loss fractions at the stated confidence; rf is an annual assumption converted to daily by compounding.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .config import TRADING_DAYS

CASH = "CASH"


def aligned_returns(prices: pd.DataFrame, bench: pd.Series | None = None):
    """Simple returns on dates where ALL columns (and benchmark) have valid prices. No forward fill."""
    df = prices.copy()
    if bench is not None:
        df = df.join(bench.rename("__bench__"), how="inner")
    r = df.pct_change(fill_method=None)
    r = r.dropna(how="any")
    if bench is not None:
        return r.drop(columns="__bench__"), r["__bench__"]
    return r, None


def buy_and_hold_value(prices: pd.DataFrame, quantities: pd.Series) -> pd.Series:
    """Historical value of FIXED quantities (true buy-and-hold) on common dates."""
    p = prices[quantities.index].dropna(how="any")
    return (p * quantities).sum(axis=1)


def constant_weight_returns(returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """Hypothetical portfolio re-balanced to `weights` daily. Used for forward-looking what-if comparisons."""
    w = weights.reindex(returns.columns).fillna(0.0)
    return returns @ w


def annualized_vol(r: pd.Series) -> float:
    return float(r.std(ddof=1) * np.sqrt(TRADING_DAYS))


def portfolio_vol_cov(w: np.ndarray, cov: pd.DataFrame | np.ndarray) -> float:
    c = np.asarray(cov)
    return float(np.sqrt(TRADING_DAYS * w @ c @ w))


def covariance(returns: pd.DataFrame) -> pd.DataFrame:
    return returns.cov()


def correlation(returns: pd.DataFrame) -> pd.DataFrame:
    return returns.corr()


def high_correlation_pairs(corr: pd.DataFrame, threshold=0.8):
    out = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            v = corr.iloc[i, j]
            if np.isfinite(v) and v >= threshold:
                out.append((cols[i], cols[j], float(v)))
    return sorted(out, key=lambda x: -x[2])


def beta(asset: pd.Series, bench: pd.Series) -> float:
    d = pd.concat([asset, bench], axis=1).dropna()
    v = d.iloc[:, 1].var(ddof=1)
    return float(d.iloc[:, 0].cov(d.iloc[:, 1]) / v) if v > 0 else float("nan")


def max_drawdown(index: pd.Series) -> dict:
    peak = index.cummax()
    dd = index / peak - 1
    trough = dd.idxmin()
    peak_date = index.loc[:trough].idxmax()
    after = index.loc[trough:]
    rec = after[after >= index.loc[peak_date]]
    return {"max_drawdown": float(dd.min()), "peak_date": peak_date, "trough_date": trough,
            "recovery_date": rec.index[0] if len(rec) else None, "series": dd}


def daily_rf(rf_annual: float) -> float:
    return (1 + rf_annual) ** (1 / TRADING_DAYS) - 1


def sharpe(r: pd.Series, rf_annual: float):
    ex = r - daily_rf(rf_annual)
    sd = ex.std(ddof=1)
    return None if not np.isfinite(sd) or sd < 1e-12 else float(ex.mean() / sd * np.sqrt(TRADING_DAYS))


def downside_deviation(r: pd.Series, rf_annual=0.0) -> float:
    d = np.minimum(r - daily_rf(rf_annual), 0.0)
    return float(np.sqrt((d ** 2).mean()) * np.sqrt(TRADING_DAYS))


def sortino(r: pd.Series, rf_annual: float):
    dd = downside_deviation(r, rf_annual)
    return None if dd < 1e-12 else float((r - daily_rf(rf_annual)).mean() * TRADING_DAYS / dd)


def var_cvar(r: pd.Series, conf=0.95):
    q = np.quantile(r, 1 - conf)
    tail = r[r <= q]
    return float(-q), float(-tail.mean())


def tracking_error(r: pd.Series, b: pd.Series) -> float:
    return float((r - b).std(ddof=1) * np.sqrt(TRADING_DAYS))


def information_ratio(r: pd.Series, b: pd.Series):
    te = tracking_error(r, b)
    return None if te < 1e-12 else float((r - b).mean() * TRADING_DAYS / te)


def cagr(index: pd.Series) -> float:
    yrs = (index.index[-1] - index.index[0]).days / 365.25
    return float((index.iloc[-1] / index.iloc[0]) ** (1 / yrs) - 1) if yrs > 0 else float("nan")


def monthly_returns(index: pd.Series) -> pd.Series:
    return index.resample("ME").last().pct_change().dropna()


def rolling_vol(r: pd.Series, window=63) -> pd.Series:
    return r.rolling(window).std(ddof=1) * np.sqrt(TRADING_DAYS)


def rolling_beta(r: pd.Series, b: pd.Series, window=126) -> pd.Series:
    return r.rolling(window).cov(b) / b.rolling(window).var()


def weights_from_values(values: pd.Series) -> pd.Series:
    tot = values.sum()
    if tot <= 0:
        raise ValueError("Total portfolio value must be positive")
    return values / tot


def concentration(weights: pd.Series, sectors: dict | None = None) -> dict:
    w = weights.drop(CASH, errors="ignore")
    hhi = float((w ** 2).sum())
    s = w.sort_values(ascending=False)
    out = {"hhi": hhi, "effective_n": 1 / hhi if hhi > 0 else float("nan"), "largest_weight": float(s.iloc[0]),
           "largest_holding": s.index[0], "top3": float(s.iloc[:3].sum()), "top5": float(s.iloc[:5].sum()),
           "avg_weight": float(w.mean()), "n_holdings": int((w > 0).sum())}
    if sectors:
        sw = w.groupby(lambda t: sectors.get(t) or "Unknown").sum().sort_values(ascending=False)
        out.update(sector_weights=sw.to_dict(), largest_sector=sw.index[0], largest_sector_weight=float(sw.iloc[0]),
                   sector_hhi=float((sw ** 2).sum()))
    return out


def risk_contributions(weights: pd.Series, cov: pd.DataFrame) -> pd.Series:
    """Percent contribution to portfolio variance: w_i*(Σw)_i / wᵀΣw. Sums to 1."""
    w = weights.reindex(cov.index).fillna(0.0).values
    tv = w @ cov.values @ w
    if tv <= 0:
        return pd.Series(np.nan, index=cov.index)
    return pd.Series(w * (cov.values @ w) / tv, index=cov.index)


def expected_returns(returns: pd.DataFrame, bench: pd.Series | None, rf_annual: float) -> dict:
    """Two labelled, distinct estimates. Neither is a forecast or guarantee."""
    hist = returns.mean() * TRADING_DAYS
    out = {"historical_mean": hist}
    if bench is not None:
        mkt_prem = bench.mean() * TRADING_DAYS - rf_annual
        out["capm"] = pd.Series({c: rf_annual + beta(returns[c], bench) * mkt_prem for c in returns.columns})
    return out


def portfolio_metrics(weights: pd.Series, returns: pd.DataFrame, bench: pd.Series | None, rf_annual=0.065,
                      sectors: dict | None = None, min_obs=120, conf=0.95) -> dict:
    """All metrics for a weight vector on a common return window. Metrics that cannot be computed are None."""
    warnings = []
    w = weights.reindex(returns.columns).fillna(0.0)
    n = len(returns)
    m = {"n_obs": n, "window_start": returns.index[0] if n else None, "window_end": returns.index[-1] if n else None,
         "weight_sum": float(weights.sum())}
    m.update(concentration(weights, sectors))
    if n < 2:
        warnings.append("Insufficient return history; risk metrics unavailable")
        m["warnings"] = warnings
        return m
    if n < min_obs:
        warnings.append(f"Only {n} aligned observations (<{min_obs}); estimates unstable")
    pr = constant_weight_returns(returns, w)
    idx = (1 + pr).cumprod()
    cov = covariance(returns)
    m["volatility"] = annualized_vol(pr)
    m["volatility_cov"] = portfolio_vol_cov(w.values, cov)
    m["max_drawdown"] = max_drawdown(idx)["max_drawdown"]
    m["sharpe"] = sharpe(pr, rf_annual)
    m["sortino"] = sortino(pr, rf_annual)
    m["var"], m["cvar"] = var_cvar(pr, conf)
    m["var_conf"] = conf  # 1-day historical, positive loss fraction
    m["return_convention"] = "constant-weight (daily rebalanced) hypothetical on current weights"
    m["hist_expected_return"] = float(pr.mean() * TRADING_DAYS)
    if bench is not None:
        b = bench.reindex(pr.index)
        m["beta"] = beta(pr, b)
        m["tracking_error"] = tracking_error(pr, b)
        m["information_ratio"] = information_ratio(pr, b)
        m["capm_expected_return"] = float(rf_annual + m["beta"] * (b.mean() * TRADING_DAYS - rf_annual))
    else:
        m["beta"] = m["capm_expected_return"] = None
        warnings.append("Benchmark unavailable; beta/tracking error/CAPM not computed")
    m["warnings"] = warnings
    return m
