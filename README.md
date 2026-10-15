# Portfolio Intelligence Platform (India) — yfinance prototype

Analytical decision-support for Indian retail investors: upload holdings → risk/concentration analysis → simulate
**adding / reducing / replacing** stocks → grounded AI explanation → PDF/CSV export. No trade execution. Not investment advice.

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # optional: add ANTHROPIC_API_KEY for LLM explanations
streamlit run app.py
```
* **Demo mode (default)** needs no network or keys; it uses **synthetic** prices and is labelled as such everywhere.
* **Live mode** uses Yahoo Finance via `yfinance` (no broker credentials / paid keys). Choose it in the sidebar.
* Without `ANTHROPIC_API_KEY` the AI page shows a deterministic summary; all quantitative features still work.

Run tests: `python -m pytest -q`. Regenerate the example report: `python make_example_report.py`.

## Layout
| Path | Purpose |
|---|---|
| `app.py` | Streamlit UI (9 pages) |
| `pfai/market_data.py` | yfinance adapter, normalisation/quality checks, caching, ticker mapping, fundamentals |
| `pfai/portfolio.py` | CSV/Excel ingestion, column mapping, validation, holdings table |
| `pfai/analytics.py` | Quant engine (returns, vol, cov/corr, beta, drawdown, Sharpe/Sortino, VaR/CVaR, HHI, risk contributions) |
| `pfai/scenarios.py` | Add / reduce / replace simulators, comparison, transparent ranking |
| `pfai/optimizer.py` | Constrained long-only mean-variance (min-variance, mean-variance, max-Sharpe) |
| `pfai/ai.py` | Tool functions + LLM loop + numeric-claim validation + no-key fallback |
| `pfai/db.py` | SQLite schema, user-scoped access, cache table |
| `pfai/report.py` | CSV / PDF export |
| `tests/` | 25 tests (synthetic fixtures, independently computed expectations) |

## Deployment
* **Local/prototype:** SQLite (`PFAI_DB_PATH`). 
* **Hosted:** Streamlit Community Cloud / Docker (`pip install -r requirements.txt; streamlit run app.py --server.port $PORT`). Put `ANTHROPIC_API_KEY` in the platform's secrets manager, never in the repo.
* **PostgreSQL upgrade:** NOT implemented; `db.py` uses plain SQL with `?` placeholders and would need a driver/placeholder swap and real authentication (the current `user_id` is a local placeholder, not authentication).
* **Commercial use:** review Yahoo Finance's terms first (see `docs/DATA_SOURCES.md`).

See `docs/METHODOLOGY.md`, `docs/DATA_SOURCES.md`, `docs/IMPLEMENTATION_REPORT.md`.
