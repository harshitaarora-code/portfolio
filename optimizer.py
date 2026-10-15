"""Constrained mean-variance optimisation (long-only). Reports infeasibility instead of returning invalid weights."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from .config import TRADING_DAYS


def optimize(mu: pd.Series, cov: pd.DataFrame, objective="min_variance", max_weight=1.0, risk_aversion=3.0,
             rf_annual=0.065, sectors: dict | None = None, sector_limits: dict | None = None,
             current: pd.Series | None = None, max_turnover: float | None = None) -> dict:
    """mu: annualised expected returns; cov: DAILY covariance (annualised internally).
    objective: min_variance | mean_variance (max μᵀw − λwᵀΣw) | max_sharpe."""
    names = list(mu.index); n = len(names)
    S = cov.loc[names, names].values * TRADING_DAYS
    m = mu.values.astype(float)
    if n * max_weight < 1 - 1e-9:
        return {"status": "infeasible", "message": f"{n} assets x max weight {max_weight:.0%} < 100%", "weights": None}
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    if sectors and sector_limits:
        for sec, lim in sector_limits.items():
            idx = [i for i, t in enumerate(names) if sectors.get(t) == sec]
            if idx:
                cons.append({"type": "ineq", "fun": lambda w, idx=idx, lim=lim: lim - w[idx].sum()})
    if current is not None and max_turnover is not None:
        c = current.reindex(names).fillna(0).values
        cons.append({"type": "ineq", "fun": lambda w: max_turnover - 0.5 * np.abs(w - c).sum()})
    if objective == "min_variance":
        f = lambda w: w @ S @ w
    elif objective == "mean_variance":
        f = lambda w: -(m @ w - risk_aversion * w @ S @ w)
    elif objective == "max_sharpe":
        f = lambda w: -((m @ w - rf_annual) / np.sqrt(max(w @ S @ w, 1e-12)))
    else:
        raise ValueError("Unknown objective")
    best = None
    rng = np.random.default_rng(0)
    for x0 in [np.ones(n) / n] + [rng.dirichlet(np.ones(n)) for _ in range(4)]:
        res = minimize(f, np.clip(x0, 0, max_weight), method="SLSQP", bounds=[(0, max_weight)] * n,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-10})
        ok = res.success and abs(res.x.sum() - 1) < 1e-6 and (res.x >= -1e-8).all() and (res.x <= max_weight + 1e-8).all()
        if ok and all(c["fun"](res.x) >= -1e-6 for c in cons if c["type"] == "ineq"):
            if best is None or res.fun < best.fun:
                best = res
    if best is None:
        return {"status": "infeasible", "message": "Optimizer could not satisfy the constraints", "weights": None}
    w = pd.Series(np.clip(best.x, 0, None), index=names); w /= w.sum()
    return {"status": "optimal", "weights": w, "expected_return": float(m @ w.values),
            "volatility": float(np.sqrt(w.values @ S @ w.values)),
            "assumptions": "μ and Σ are historical estimates; sensitive to window; not a recommendation."}
