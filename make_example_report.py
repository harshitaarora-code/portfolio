"""Generates docs/example_report_SYNTHETIC.pdf from SYNTHETIC data (no network)."""
import pandas as pd
from pfai import analytics as an, scenarios as sc, report, demo, ai
p, b = demo.synthetic_prices(seed=7, end="2026-10-09")
df = pd.read_csv("data/sample_portfolio.csv"); held = [s + ".NS" for s in df["Symbol"]]
v = pd.Series((df["Quantity"].values * p[held].iloc[-1].values), index=held)
ctx = sc.Context(p, b, 0.065, demo.sectors(), 25)
r, bb = an.aligned_returns(p[held], b)
m = an.portfolio_metrics(an.weights_from_values(v), r, bb, 0.065, demo.sectors())
res = sc.simulate_addition(ctx, v, "SUNPHARMA.NS", v.sum() * 0.15)
cmp_ = sc.compare(res.before, res.after)
t = ai.AnalystTools(ctx, v); out = ai.template_summary(t)
pdf = report.to_pdf("Portfolio analysis report (SYNTHETIC DEMO DATA)", m, cmp_, out,
                    {"Data": demo.SYNTHETIC_LABEL, "Scenario": res.label, "Benchmark": "synthetic index", "Window": f"{m['window_start'].date()} to {m['window_end'].date()} ({m['n_obs']} obs)"},
                    [demo.SYNTHETIC_LABEL] + m["warnings"])
open("docs/example_report_SYNTHETIC.pdf", "wb").write(pdf)
print(cmp_.round(4).to_string(index=False))
