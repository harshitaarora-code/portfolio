"""Scenario engine: add / reduce / replace. Never touches actual holdings; works on value vectors."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from . import analytics as an
from .analytics import CASH


@dataclass
class Context:
    prices: pd.DataFrame            # adjusted Close, wide, includes holdings + candidates
    bench: pd.Series | None = None
    rf_annual: float = 0.065
    sectors: dict = field(default_factory=dict)
    cost_bps: float = 25.0
    min_obs: int = 120

    def returns(self, symbols):
        cols = [s for s in dict.fromkeys(symbols) if s != CASH]
        missing = [s for s in cols if s not in self.prices.columns]
        if missing:
            raise ValueError(f"No price history for: {missing}")
        r, b = an.aligned_returns(self.prices[cols], self.bench)
        r[CASH] = 0.0   # cash: zero return, zero risk (simplification, documented)
        return r, b


def evaluate(ctx: Context, values: pd.Series, universe) -> dict:
    """Metrics for `values` on a return window common to the whole universe (so before/after use identical observations)."""
    r, b = ctx.returns(list(universe))
    w = an.weights_from_values(values[values > 1e-9])
    return an.portfolio_metrics(w, r, b, ctx.rf_annual, ctx.sectors, ctx.min_obs)


@dataclass
class ScenarioResult:
    label: str
    before_values: pd.Series
    after_values: pd.Series
    before: dict
    after: dict
    cost_inr: float
    notes: list


def _result(ctx, label, before_values, after_values, cost, notes, universe):
    return ScenarioResult(label, before_values, after_values, evaluate(ctx, before_values, universe),
                          evaluate(ctx, after_values, universe), cost, notes)


def simulate_addition(ctx: Context, values: pd.Series, candidate: str, amount: float) -> ScenarioResult:
    """Scenario A: new cash. Value rises by (amount - costs); the contribution is a cash flow, not performance."""
    if amount <= 0:
        raise ValueError("Amount must be positive")
    cost = amount * ctx.cost_bps / 1e4
    after = values.copy()
    after[candidate] = after.get(candidate, 0.0) + (amount - cost)
    notes = [f"New cash of INR {amount:,.0f} treated as a contribution, not performance; est. costs INR {cost:,.0f}."]
    return _result(ctx, f"Add {candidate} (INR {amount:,.0f} new cash)", values, after, cost, notes, list(values.index) + [candidate])


def simulate_replacement(ctx: Context, values: pd.Series, sells: dict, buys: dict, cash_fraction=0.0) -> ScenarioResult:
    """Scenario B/others: sell `sells` {symbol: INR}, put `buys` {symbol: fraction of invested proceeds} (fractions sum to 1)
    and optionally retain `cash_fraction` of proceeds as cash. Costs are charged on both legs and deducted from the buys."""
    for s, a in sells.items():
        if s not in values.index:
            raise ValueError(f"{s} not in portfolio")
        if a <= 0 or a > values[s] + 1e-6:
            raise ValueError(f"Invalid sell amount for {s}: must be in (0, {values[s]:,.2f}]")
    if buys and abs(sum(buys.values()) - 1) > 1e-9:
        raise ValueError("Buy fractions must sum to 1")
    if not 0 <= cash_fraction <= 1:
        raise ValueError("cash_fraction must be in [0,1]")
    proceeds = sum(sells.values())
    invest_gross = proceeds * (1 - cash_fraction) if buys else 0.0
    cash_kept = proceeds - invest_gross
    cost = ctx.cost_bps / 1e4 * (proceeds + invest_gross)
    invest_net = max(invest_gross - cost, 0.0)
    after = values.copy()
    for s, a in sells.items():
        after[s] -= a
    for s, f in buys.items():
        after[s] = after.get(s, 0.0) + invest_net * f
    if cash_kept > 0:
        after[CASH] = after.get(CASH, 0.0) + cash_kept
    after = after[after > 1e-9]
    notes = [f"Estimated transaction costs INR {cost:,.0f} deducted from purchases.",
             "Cash is modelled as zero-return/zero-risk.",
             "Tax impact (capital gains) is NOT modelled."]
    return _result(ctx, f"Sell {', '.join(sells)} -> {', '.join(buys) or 'cash'}", values, after, cost, notes,
                   list(values.index) + list(buys))


def simulate_reduction(ctx: Context, values: pd.Series, symbol: str, amount: float) -> ScenarioResult:
    return simulate_replacement(ctx, values, {symbol: amount}, {}, 1.0)


PCT_METRICS = {"largest_weight", "top3", "top5", "largest_sector_weight", "volatility", "max_drawdown", "var", "cvar",
               "hist_expected_return", "capm_expected_return", "tracking_error"}
COMPARE_KEYS = ["hhi", "effective_n", "largest_weight", "top3", "top5", "largest_sector_weight", "sector_hhi", "volatility",
                "beta", "max_drawdown", "sharpe", "sortino", "var", "cvar", "hist_expected_return", "capm_expected_return"]


def compare(before: dict, after: dict) -> pd.DataFrame:
    rows = []
    for k in COMPARE_KEYS:
        b, a = before.get(k), after.get(k)
        if b is None or a is None or (isinstance(b, float) and np.isnan(b)):
            rows.append({"metric": k, "current": b, "proposed": a, "abs_change": None, "pp_change": None, "rel_change_pct": None}); continue
        d = a - b
        rows.append({"metric": k, "current": b, "proposed": a, "abs_change": d,
                     "pp_change": d * 100 if k in PCT_METRICS else None,
                     "rel_change_pct": (d / abs(b) * 100) if b != 0 else None})
    return pd.DataFrame(rows)


def compare_scenarios(results: list[ScenarioResult]) -> pd.DataFrame:
    rows = {"Current": results[0].before} if results else {}
    for r in results:
        rows[r.label] = r.after
    keys = COMPARE_KEYS
    return pd.DataFrame({k: {m: v.get(m) for m in keys} for k, v in rows.items()}).T


DEFAULT_WEIGHTS = {"volatility": 0.4, "beta": 0.1, "hhi": 0.3, "largest_sector_weight": 0.2}


def rank_scenarios(results: list[ScenarioResult], weights: dict | None = None) -> pd.DataFrame:
    """Transparent score = -Σ w_k * relative_change_k over lower-is-better metrics (vol, beta, HHI, top sector weight).
    Positive = risk/concentration reduced vs current. Ignores return, valuation, tax; users may re-weight."""
    weights = weights or DEFAULT_WEIGHTS
    out = []
    for r in results:
        s, used = 0.0, {}
        for k, wk in weights.items():
            b, a = r.before.get(k), r.after.get(k)
            if b not in (None, 0) and a is not None and np.isfinite(b) and np.isfinite(a):
                rc = (a - b) / abs(b); s += -wk * rc; used[k] = rc
        out.append({"scenario": r.label, "score": s, "cost_inr": r.cost_inr, **{f"rel_{k}": v for k, v in used.items()}})
    return pd.DataFrame(out).sort_values("score", ascending=False).reset_index(drop=True)
