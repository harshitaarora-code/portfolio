"""AI analyst. Numbers come ONLY from deterministic tool outputs; the LLM explains them.
Works without an API key (deterministic template summary). LLM path requires ANTHROPIC_API_KEY."""
from __future__ import annotations
import json, os, re
import numpy as np
import pandas as pd
from . import analytics as an, scenarios as sc

SECTIONS = ["executive_summary", "main_findings", "risk_implications", "diversification_implications",
            "current_vs_proposed", "potential_advantages", "potential_disadvantages", "assumptions", "data_limitations"]

SYSTEM = ("You are a portfolio analyst assistant for Indian retail investors. Use ONLY numbers returned by tools; never invent "
          "prices, fundamentals or forecasts. Distinguish facts, estimates and interpretations. State uncertainty and when data is "
          "unavailable. Never promise returns or diversification. This is decision support, not advice. Respond with a JSON object "
          f"with string-list fields: {SECTIONS}.")


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (pd.Series,)):
        return _clean(o.to_dict())
    if isinstance(o, pd.DataFrame):
        return _clean(o.to_dict(orient="index"))
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, pd.Timestamp):
        return o.date().isoformat()
    return o


class AnalystTools:
    """Deterministic tool functions bound to the current session state. Every output is logged for validation."""

    def __init__(self, ctx: sc.Context, values: pd.Series, fundamentals: dict | None = None):
        self.ctx, self.values, self.fund, self.log = ctx, values, fundamentals or {}, []

    def _rec(self, name, out):
        out = _clean(out); self.log.append({"tool": name, "output": out}); return out

    def calculate_portfolio_metrics(self):
        m = sc.evaluate(self.ctx, self.values, list(self.values.index))
        return self._rec("calculate_portfolio_metrics", {k: v for k, v in m.items() if k != "sector_weights"} | {"sector_weights": m.get("sector_weights")})

    def calculate_correlation_matrix(self, threshold: float = 0.8):
        r, _ = self.ctx.returns(list(self.values.index))
        c = an.correlation(r.drop(columns=an.CASH))
        return self._rec("calculate_correlation_matrix", {"high_pairs": an.high_correlation_pairs(c, threshold), "threshold": threshold})

    def calculate_risk_contributions(self):
        r, _ = self.ctx.returns(list(self.values.index))
        w = an.weights_from_values(self.values)
        rc = an.risk_contributions(w, an.covariance(r))
        return self._rec("calculate_risk_contributions", {"weights": w, "risk_contribution_share": rc.drop(an.CASH, errors="ignore")})

    def simulate_addition(self, candidate: str, amount: float):
        res = sc.simulate_addition(self.ctx, self.values, candidate, amount)
        return self._rec("simulate_addition", {"label": res.label, "comparison": sc.compare(res.before, res.after).set_index("metric"), "cost_inr": res.cost_inr, "notes": res.notes})

    def simulate_reduction(self, symbol: str, amount: float):
        res = sc.simulate_reduction(self.ctx, self.values, symbol, amount)
        return self._rec("simulate_reduction", {"label": res.label, "comparison": sc.compare(res.before, res.after).set_index("metric"), "cost_inr": res.cost_inr, "notes": res.notes})

    def simulate_replacement(self, sell_symbol: str, sell_amount: float, buy_symbol: str):
        res = sc.simulate_replacement(self.ctx, self.values, {sell_symbol: sell_amount}, {buy_symbol: 1.0})
        return self._rec("simulate_replacement", {"label": res.label, "comparison": sc.compare(res.before, res.after).set_index("metric"), "cost_inr": res.cost_inr, "notes": res.notes})

    def retrieve_verified_fundamentals(self, symbol: str):
        f = self.fund.get(symbol)
        return self._rec("retrieve_verified_fundamentals", f if f else {"symbol": symbol, "available": False, "note": "No fundamentals retrieved; do not infer."})

    SCHEMAS = [
        {"name": "calculate_portfolio_metrics", "description": "Concentration, volatility, beta, drawdown, Sharpe, VaR for the current portfolio.", "input_schema": {"type": "object", "properties": {}}},
        {"name": "calculate_correlation_matrix", "description": "Highly correlated holding pairs.", "input_schema": {"type": "object", "properties": {"threshold": {"type": "number"}}}},
        {"name": "calculate_risk_contributions", "description": "Each holding's share of portfolio variance.", "input_schema": {"type": "object", "properties": {}}},
        {"name": "simulate_addition", "description": "Add a stock with new cash (INR).", "input_schema": {"type": "object", "properties": {"candidate": {"type": "string"}, "amount": {"type": "number"}}, "required": ["candidate", "amount"]}},
        {"name": "simulate_reduction", "description": "Sell part of a holding and hold cash.", "input_schema": {"type": "object", "properties": {"symbol": {"type": "string"}, "amount": {"type": "number"}}, "required": ["symbol", "amount"]}},
        {"name": "simulate_replacement", "description": "Sell a holding amount and buy another stock.", "input_schema": {"type": "object", "properties": {"sell_symbol": {"type": "string"}, "sell_amount": {"type": "number"}, "buy_symbol": {"type": "string"}}, "required": ["sell_symbol", "sell_amount", "buy_symbol"]}},
        {"name": "retrieve_verified_fundamentals", "description": "Third-party fundamentals if retrieved; may be missing.", "input_schema": {"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]}},
    ]


def flatten_numbers(o, acc=None):
    acc = [] if acc is None else acc
    if isinstance(o, dict):
        [flatten_numbers(v, acc) for v in o.values()]
    elif isinstance(o, list):
        [flatten_numbers(v, acc) for v in o]
    elif isinstance(o, (int, float)) and not isinstance(o, bool):
        acc.append(float(o))
    return acc


def validate_numbers(text: str, facts: list, rel_tol=0.015, abs_tol=0.06):
    """Return numbers in `text` that cannot be matched to any tool-output figure (also as %, or x100). Small ints/years ignored."""
    pool = set()
    for v in facts:
        pool.update({v, v * 100, v / 1e5, v / 1e7})
    unmatched = []
    for tok in re.findall(r"-?\d[\d,]*\.?\d*", text):
        try:
            x = float(tok.replace(",", ""))
        except ValueError:
            continue
        if abs(x) <= 10 and x == int(x) or 1990 <= x <= 2100 and x == int(x):
            continue
        if not any(abs(abs(x) - abs(p)) <= max(abs_tol, rel_tol * abs(p)) for p in pool):
            unmatched.append(tok)
    return unmatched


def template_summary(tools: AnalystTools) -> dict:
    """Deterministic, number-faithful explanation used when no LLM key is configured."""
    m = tools.calculate_portfolio_metrics()
    f = lambda v, p=True: "n/a" if v is None else (f"{v*100:.1f}%" if p else f"{v:.2f}")
    main = [f"{m['n_holdings']} holdings; largest is {m['largest_holding']} at {f(m['largest_weight'])}; top-3 = {f(m['top3'])}.",
            f"HHI {m['hhi']:.3f}, effective number of holdings {m['effective_n']:.1f}."]
    if m.get("largest_sector"):
        main.append(f"Largest sector: {m['largest_sector']} at {f(m['largest_sector_weight'])}.")
    risk = [f"Historical annualised volatility {f(m.get('volatility'))}; max drawdown {f(m.get('max_drawdown'))}; beta {f(m.get('beta'), False)}.",
            f"Historical {round(m.get('var_conf', 0.95) * 100)}% 1-day VaR style loss estimate {f(m.get('var'))} (CVaR {f(m.get('cvar'))})." if m.get("var") is not None else "VaR unavailable."]
    return {"executive_summary": ["Deterministic summary of calculated metrics (no LLM configured)."], "main_findings": main,
            "risk_implications": risk, "diversification_implications": [f"Effective holdings {m['effective_n']:.1f} versus {m['n_holdings']} nominal."],
            "current_vs_proposed": ["Run a scenario to compare."], "potential_advantages": [], "potential_disadvantages": [],
            "assumptions": ["Constant-weight historical estimates on current weights; not a backtest of actual trades."],
            "data_limitations": m.get("warnings", []) or ["Yahoo Finance data may be delayed/incomplete."]}


def ask(question: str, tools: AnalystTools, model: str | None = None, max_steps=6) -> dict:
    """Returns {'report': dict, 'mode': 'llm'|'template', 'unmatched_numbers': [...], 'error': str|None}. Never raises."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return {"report": template_summary(tools), "mode": "template", "unmatched_numbers": [],
                "error": "ANTHROPIC_API_KEY not set — showing deterministic summary. Quantitative features unaffected."}
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=key)
        model = model or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
        msgs = [{"role": "user", "content": question}]
        for _ in range(max_steps):
            resp = client.messages.create(model=model, max_tokens=2000, system=SYSTEM, tools=AnalystTools.SCHEMAS, messages=msgs)
            if resp.stop_reason != "tool_use":
                text = "".join(b.text for b in resp.content if b.type == "text")
                break
            msgs.append({"role": "assistant", "content": resp.content})
            results = []
            for b in resp.content:
                if b.type == "tool_use":
                    try:
                        out = getattr(tools, b.name)(**b.input)
                    except Exception as e:
                        out = {"error": str(e)}
                    results.append({"type": "tool_result", "tool_use_id": b.id, "content": json.dumps(out)})
            msgs.append({"role": "user", "content": results})
        else:
            raise RuntimeError("Tool loop did not terminate")
        m = re.search(r"\{.*\}", text, re.S)
        report = json.loads(m.group(0)) if m else {"executive_summary": [text]}
        facts = flatten_numbers([l["output"] for l in tools.log])
        unmatched = validate_numbers(json.dumps(report), facts)
        return {"report": report, "mode": "llm", "unmatched_numbers": unmatched, "error": None}
    except Exception as e:
        return {"report": template_summary(tools), "mode": "template", "unmatched_numbers": [], "error": f"AI unavailable ({e}); showing deterministic summary."}
