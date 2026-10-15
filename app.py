import time
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pfai import analytics as an, scenarios as sc, optimizer as opt, portfolio as pf, market_data as md, ai, report, demo, db, config

st.set_page_config(page_title="Portfolio Intelligence (India)", page_icon="📊", layout="wide")
st.markdown("""<style>
.block-container{padding-top:1.5rem;max-width:1250px}
div[data-testid=stMetric]{background:var(--secondary-background-color);border-radius:10px;padding:12px 14px}
.banner{border-left:4px solid #d97706;background:rgba(217,119,6,.10);padding:8px 12px;border-radius:6px;margin-bottom:10px;font-size:.9rem}
.synthetic{border-left-color:#dc2626;background:rgba(220,38,38,.10)}
</style>""", unsafe_allow_html=True)

S = st.session_state
S.setdefault("settings", config.Settings())
S.setdefault("mode", "Demo (synthetic data)")
S.setdefault("holdings", None); S.setdefault("prices", None); S.setdefault("bench", None)
S.setdefault("meta", {}); S.setdefault("scn", []); S.setdefault("fund", {}); S.setdefault("ai_report", None)
S.setdefault("user_id", "local-user")
pct = lambda v, d=1: "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v*100:.{d}f}%"
num = lambda v, d=2: "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:,.{d}f}"
inr = lambda v: f"₹{v:,.0f}"


@st.cache_resource
def con():
    return db.connect()


@st.cache_data(ttl=3600, show_spinner=False)
def cached_prices(tickers: tuple, period: str, force: bool):
    return md.get_historical_prices(list(tickers), period, con=con(), force_refresh=force)


def is_demo():
    return S.mode.startswith("Demo")


def banner():
    if is_demo():
        st.markdown(f'<div class="banner synthetic"><b>{demo.SYNTHETIC_LABEL}.</b> Every number below is generated for demonstration.</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="banner">{config.DISCLAIMER}</div>', unsafe_allow_html=True)


def ctx():
    s = S.settings
    return sc.Context(S.prices, S.bench, s.risk_free_annual, S.meta.get("sectors", {}), s.cost_bps, s.min_obs)


def values():
    h = S.holdings
    return pd.Series((h["quantity"] * h["price"]).values, index=h["yahoo_ticker"].values)


def need_portfolio():
    if S.holdings is None:
        st.info("Load a portfolio first (Portfolio upload page)."); st.stop()


def load_demo():
    p, b = demo.synthetic_prices()
    df = pd.read_csv("data/sample_portfolio.csv")
    m = pf.apply_mapping(df, pf.suggest_mapping(df.columns))
    v = pf.validate_holdings(m)
    v["yahoo_ticker"] = v["symbol"] + ".NS"; v["mapping_status"] = "verified (synthetic)"
    v["price"] = v["yahoo_ticker"].map(p.iloc[-1])
    S.holdings, S.prices, S.bench = v, p, b
    S.meta = {"sectors": demo.sectors(), "retrieved_at": time.time(), "warnings": [demo.SYNTHETIC_LABEL],
              "latest_obs": str(p.index[-1].date())}


def live_load(v):
    tickers = tuple(v["yahoo_ticker"].tolist() + [S.settings.benchmark])
    res = cached_prices(tickers, S.settings.period, S.pop('force_refresh', False))
    px_ = res.prices
    if S.settings.benchmark not in px_.columns:
        st.error("Benchmark data unavailable; beta/benchmark comparison will be skipped."); bench = None
    else:
        bench = px_[S.settings.benchmark]
    stock = px_.drop(columns=[S.settings.benchmark], errors="ignore")
    missing = [t for t in v["yahoo_ticker"] if t not in stock.columns]
    if missing:
        st.error(f"No price history for {missing}; they are excluded from analysis."); v = v[~v["yahoo_ticker"].isin(missing)]
    v = v.copy(); v["price"] = v["yahoo_ticker"].map(stock.apply(lambda c: c.dropna().iloc[-1]))
    secs = {t: md.get_fundamentals(t, con()).get("sector") for t in v["yahoo_ticker"]}
    S.holdings, S.prices, S.bench = v, stock, bench
    S.meta = {"sectors": secs, "retrieved_at": res.retrieved_at, "warnings": res.quality.warnings, "latest_obs": res.quality.latest_observation,
              "from_cache": res.from_cache, "stale_cache": res.stale_cache}


# ----------------------------------------------------------------- sidebar
pages = ["1 · Welcome", "2 · Portfolio upload", "3 · Overview", "4 · Risk analysis", "5 · Stock explorer", "6 · Scenario lab",
         "7 · AI analyst", "8 · Reports", "9 · Settings & data status"]
page = st.sidebar.radio("Navigate", pages)
S.mode = st.sidebar.selectbox("Data mode", ["Demo (synthetic data)", "Live (Yahoo Finance via yfinance)"],
                              index=0 if is_demo() else 1)
if S.meta.get("retrieved_at"):
    st.sidebar.caption(f"Last retrieval: {datetime.fromtimestamp(S.meta['retrieved_at']):%Y-%m-%d %H:%M}")
if st.sidebar.button("🔄 Refresh data"):
    cached_prices.clear(); S['force_refresh'] = True; st.rerun()
banner()

# ----------------------------------------------------------------- 1 Welcome
if page == pages[0]:
    st.title("📊 Portfolio Intelligence for Indian Investors")
    st.write("Upload your NSE/BSE holdings, measure concentration and risk, then test what **adding, reducing or replacing** a stock "
             "would do to your portfolio before you act. Numbers are computed by a deterministic engine; the AI only explains them.")
    c = st.columns(3)
    c[0].markdown("**Analyse**\n\nAllocation, sector exposure, volatility, beta, drawdown, VaR, correlation.")
    c[1].markdown("**Simulate**\n\nNew-cash purchases, sale-funded replacements, cash raises — with before/after metrics.")
    c[2].markdown("**Explain**\n\nGrounded explanations plus PDF/CSV export. No trade execution.")
    if st.button("Try the demo portfolio (synthetic data)", type="primary"):
        S.mode = "Demo (synthetic data)"; load_demo(); st.success("Demo portfolio loaded — open 'Overview'.")

# ----------------------------------------------------------------- 2 Upload
elif page == pages[1]:
    st.header("Portfolio upload")
    tab1, tab2 = st.tabs(["CSV / Excel", "Manual entry"])
    raw = None
    with tab1:
        f = st.file_uploader("CSV or Excel", type=["csv", "xlsx", "xls"])
        if f:
            raw = pf.read_file(f, f.name)
        elif st.button("Use sample file"):
            raw = pd.read_csv("data/sample_portfolio.csv")
    with tab2:
        base = pd.DataFrame({"Symbol": [""], "Exchange": ["NSE"], "Quantity": [0], "Average Price": [0.0]})
        ed = st.data_editor(base, num_rows="dynamic", width="stretch", key="manual")
        if st.button("Use manual entries"):
            raw = ed[ed["Symbol"].astype(str).str.strip() != ""].copy()
            if raw.empty:
                st.warning("Add at least one holding."); raw = None
    if raw is not None:
        S["raw_upload"] = raw
    raw = S.get("raw_upload")
    if raw is not None:
        sug = pf.suggest_mapping(raw.columns); opts = [None] + list(raw.columns)
        st.subheader("Column mapping")
        cols = st.columns(len(sug)); mp = {}
        for (field, guess), c in zip(sug.items(), cols):
            mp[field] = c.selectbox(field, opts, index=opts.index(guess) if guess in opts else 0, key=f"map_{field}")
        if not (mp["symbol"] and mp["quantity"]):
            st.error("Map at least Symbol and Quantity.")
        else:
            v = pf.validate_holdings(pf.apply_mapping(raw, mp))
            show = v.assign(issues=v["issues"].map("; ".join))
            st.subheader("Preview"); st.dataframe(show, width="stretch")
            bad = v[v["issues"].map(len) > 0]
            if len(bad):
                st.warning(f"{len(bad)} row(s) have issues. Nothing is silently discarded: fix the file, or merge duplicates below.")
                if st.button("Merge duplicate holdings (quantity-weighted avg price)"):
                    S["raw_upload"] = pf.merge_duplicates(v.drop(columns="issues")); st.rerun()
            ok = v[v["issues"].map(lambda x: not [i for i in x if not i.startswith("Fractional")])]
            if is_demo():
                st.caption("Demo mode: only the synthetic universe is available.")
            if st.button("Validate tickers & load portfolio", type="primary", disabled=ok.empty):
                if is_demo():
                    unk = [s for s in ok["symbol"] if f"{s}.NS" not in demo.UNIVERSE]
                    if unk:
                        st.error(f"Not in synthetic universe: {unk}. Switch to Live mode.")
                    else:
                        load_demo(); st.success("Loaded (synthetic).")
                else:
                    with st.spinner("Validating tickers against Yahoo Finance…"):
                        r = pf.resolve_all(ok, S.get("confirm", {}))
                    S["resolved"] = r
            r = S.get("resolved")
            if r is not None and not is_demo():
                st.dataframe(r[["symbol", "exchange", "yahoo_ticker", "mapping_status", "mapping_note"]], width="stretch")
                amb = r[r["mapping_status"] == "ambiguous"]
                for _, row in amb.iterrows():
                    opts_ = [c for c in md.candidate_tickers(row["symbol"])]
                    ch = st.selectbox(f"{row['symbol']}: which listing do you hold?", ["— choose —"] + opts_, key=f"amb_{row['symbol']}")
                    if ch != "— choose —":
                        S.setdefault("confirm", {})[row["symbol"]] = ch
                if len(amb) and st.button("Re-validate with my choices"):
                    S["resolved"] = pf.resolve_all(ok, S["confirm"]); st.rerun()
                good = r[r["mapping_status"] == "verified"]
                if len(good) and not len(amb):
                    if len(good) < len(r):
                        st.warning(f"{len(r)-len(good)} holding(s) unresolved and will be excluded.")
                    if st.button("Fetch prices & load verified holdings"):
                        try:
                            live_load(good); st.success("Loaded.")
                        except md.DataUnavailable as e:
                            st.error(f"Yahoo Finance unavailable: {e}. You can retry, or use Demo mode.")

# ----------------------------------------------------------------- 3 Overview
elif page == pages[2]:
    st.header("Portfolio overview"); need_portfolio()
    v = values(); w = an.weights_from_values(v); c = an.concentration(w, S.meta["sectors"])
    h = S.holdings.copy(); h["weight"] = h["quantity"] * h["price"] / (h["quantity"] * h["price"]).sum()
    h["market_value"] = h["quantity"] * h["price"]; h["unrealised_pl"] = h["market_value"] - h["quantity"] * h["avg_price"]
    h["sector"] = h["yahoo_ticker"].map(S.meta["sectors"]).fillna("Unknown")
    k = st.columns(5)
    k[0].metric("Portfolio value", inr(v.sum())); k[1].metric("Holdings", c["n_holdings"])
    k[2].metric("Largest holding", pct(c["largest_weight"]), c["largest_holding"], delta_color="off")
    k[3].metric("Top-3 / Top-5", f"{pct(c['top3'],0)} / {pct(c['top5'],0)}"); k[4].metric("Effective holdings", num(c["effective_n"], 1))
    k = st.columns(4)
    k[0].metric("Largest sector", c.get("largest_sector", "n/a"), pct(c.get("largest_sector_weight")), delta_color="off")
    k[1].metric("HHI", num(c["hhi"], 3)); k[2].metric("Avg weight", pct(c["avg_weight"]))
    k[3].metric("Unrealised P/L", inr(h["unrealised_pl"].sum()) if h["unrealised_pl"].notna().all() else "n/a")
    a, b = st.columns(2)
    a.plotly_chart(px.pie(h, names="yahoo_ticker", values="market_value", hole=.45, title="Allocation"), width="stretch")
    b.plotly_chart(px.pie(h, names="sector", values="market_value", hole=.45, title="Sector allocation"), width="stretch")
    st.dataframe(h[["yahoo_ticker", "sector", "quantity", "avg_price", "price", "market_value", "weight", "unrealised_pl"]]
                 .sort_values("market_value", ascending=False).style.format({"weight": "{:.1%}", "market_value": "{:,.0f}", "unrealised_pl": "{:,.0f}", "price": "{:,.2f}"}),
                 width="stretch")
    st.caption("Prices: latest adjusted close in the loaded history (not a real-time quote).")

# ----------------------------------------------------------------- 4 Risk
elif page == pages[3]:
    st.header("Risk analysis"); need_portfolio()
    v = values(); r, b = an.aligned_returns(S.prices[list(v.index)], S.bench)
    w = an.weights_from_values(v); m = an.portfolio_metrics(w, r, b, S.settings.risk_free_annual, S.meta["sectors"], S.settings.min_obs, S.settings.var_conf)
    for wmsg in m["warnings"]:
        st.warning(wmsg)
    st.caption(f"Window {m['window_start'].date()} → {m['window_end'].date()} · {m['n_obs']} aligned daily observations · benchmark {S.settings.benchmark} · "
               f"{m.get('return_convention')} · rf {S.settings.risk_free_annual:.2%} (assumption)")
    k = st.columns(5)
    k[0].metric("Volatility (ann.)", pct(m.get("volatility"))); k[1].metric("Beta", num(m.get("beta")))
    k[2].metric("Max drawdown", pct(m.get("max_drawdown"))); k[3].metric("Sharpe", num(m.get("sharpe")))
    k[4].metric(f"1-day VaR {S.settings.var_conf:.0%} / CVaR", f"{pct(m.get('var'))} / {pct(m.get('cvar'))}")
    pr = an.constant_weight_returns(r, w); idx = (1 + pr).cumprod()
    fig = go.Figure(); fig.add_scatter(x=idx.index, y=idx, name="Portfolio (constant current weights)")
    if b is not None:
        fig.add_scatter(x=b.index, y=(1 + b).cumprod(), name=S.settings.benchmark)
    fig.update_layout(title="Cumulative growth of 1 (hypothetical, not your actual history)", hovermode="x unified"); st.plotly_chart(fig, width="stretch")
    dd = an.max_drawdown(idx)
    a, c2 = st.columns(2)
    c2.plotly_chart(px.area(dd["series"], title="Drawdown").update_layout(showlegend=False), width="stretch")
    corr = an.correlation(r)
    a.plotly_chart(px.imshow(corr, zmin=-1, zmax=1, color_continuous_scale="RdBu_r", text_auto=".2f", title="Correlation (daily returns)"), width="stretch")
    st.caption(f"Max drawdown: peak {dd['peak_date'].date()} → trough {dd['trough_date'].date()} → recovery {dd['recovery_date'].date() if dd['recovery_date'] is not None else 'not recovered'}")
    rc = an.risk_contributions(w, an.covariance(r))
    tbl = pd.DataFrame({"weight": w, "risk share": rc, "volatility": r.std() * np.sqrt(252),
                        "beta": [an.beta(r[t], b) if b is not None else None for t in r.columns]})
    st.subheader("Individual stock risk & contribution to portfolio variance"); st.dataframe(tbl.style.format({"weight": "{:.1%}", "risk share": "{:.1%}", "volatility": "{:.1%}", "beta": "{:.2f}"}), width="stretch")
    hp = an.high_correlation_pairs(corr, 0.7)
    st.write("**High historical correlation (≥0.70):** " + (", ".join(f"{a_}–{b_} ({c_:.2f})" for a_, b_, c_ in hp) if hp else "none"))
    st.caption("Correlation alone does not establish that two businesses are redundant.")
    with st.expander("Sensitivity to lookback"):
        rows = []
        for n in [126, 252, 504, len(r)]:
            n = min(n, len(r)); sub = r.iloc[-n:]
            rows.append({"obs": n, "volatility": an.annualized_vol(an.constant_weight_returns(sub, w)),
                         "beta": an.beta(an.constant_weight_returns(sub, w), b.iloc[-n:]) if b is not None else None})
        st.dataframe(pd.DataFrame(rows).drop_duplicates("obs"), width="stretch")

# ----------------------------------------------------------------- 5 Explorer
elif page == pages[4]:
    st.header("Stock explorer"); need_portfolio()
    held = set(S.holdings["yahoo_ticker"])
    if is_demo():
        cand = st.selectbox("Candidate (synthetic universe)", sorted(set(demo.UNIVERSE) - held))
    else:
        sym = st.text_input("NSE/BSE symbol (e.g. SUNPHARMA)"); ex = st.selectbox("Exchange", ["NSE", "BSE"]); cand = None
        if sym and st.button("Validate & load"):
            mres = md.resolve_symbol(sym, ex)
            if mres.status != "verified":
                st.error(mres.message or "Ticker could not be validated")
            else:
                try:
                    pr_ = md.get_historical_prices(mres.yahoo_ticker, S.settings.period, con=con()).prices
                    S.prices = S.prices.join(pr_, how="outer") if mres.yahoo_ticker not in S.prices else S.prices
                    S["cand"] = mres.yahoo_ticker
                except md.DataUnavailable as e:
                    st.error(str(e))
        cand = S.get("cand")
    if cand and cand in S.prices.columns:
        S["cand"] = cand
        s = S.prices[cand].dropna(); rr = s.pct_change(fill_method=None).dropna()
        st.plotly_chart(px.line(s, title=f"{cand} adjusted close"), width="stretch")
        bb = S.bench.pct_change(fill_method=None).dropna() if S.bench is not None else None
        k = st.columns(5)
        k[0].metric("Ann. volatility", pct(an.annualized_vol(rr))); k[1].metric("Beta", num(an.beta(rr, bb)) if bb is not None else "n/a")
        k[2].metric("Max drawdown", pct(an.max_drawdown(s)["max_drawdown"])); k[3].metric("CAGR (window)", pct(an.cagr(s)))
        k[4].metric("Latest obs.", str(s.index[-1].date()))
        r_all, _ = an.aligned_returns(S.prices[list(held) + [cand]])
        st.write("**Correlation with your holdings**"); st.dataframe(r_all.corr()[cand].drop(cand).rename("corr").to_frame().style.format("{:.2f}"))
        if not is_demo():
            f = md.get_fundamentals(cand, con()); S.fund[cand] = f
            st.write("**Fundamentals (third-party; verify with official filings; missing ≠ zero)**"); st.json({k_: v for k_, v in f.items() if v is not None})
        else:
            st.caption("Fundamentals are not available for synthetic data.")
        st.info("Stock-level attractiveness is separate from portfolio impact — test it in the Scenario lab.")

# ----------------------------------------------------------------- 6 Scenario lab
elif page == pages[5]:
    st.header("Scenario lab"); need_portfolio()
    v = values(); held = list(v.index)
    pool = [c for c in S.prices.columns if c not in held]
    if not pool:
        st.info("No candidate stocks loaded; add one in the Stock explorer.")
    tabs = st.tabs(["➕ Add with new cash", "➖ Reduce / raise cash", "🔁 Replace", "📊 Compare"])
    with tabs[0]:
        cand = st.selectbox("Candidate", pool, index=pool.index(S["cand"]) if S.get("cand") in pool else 0, key="a_c") if pool else None
        amt = st.number_input("New cash (₹)", 1000.0, 1e9, float(round(v.sum() * 0.1, -3)), 1000.0, key="a_amt")
        if cand and st.button("Run addition"):
            try:
                S.scn.append(sc.simulate_addition(ctx(), v, cand, amt))
            except Exception as e:
                st.error(str(e))
        if cand:
            with st.expander("Test several amounts"):
                rows = []
                for f_ in [0.05, 0.1, 0.2, 0.3]:
                    try:
                        rr = sc.simulate_addition(ctx(), v, cand, v.sum() * f_)
                        rows.append({"amount": v.sum() * f_, "vol": rr.after["volatility"], "beta": rr.after["beta"], "HHI": rr.after["hhi"], "largest wt": rr.after["largest_weight"]})
                    except Exception as e:
                        st.error(str(e)); break
                st.dataframe(pd.DataFrame(rows).style.format({"amount": "{:,.0f}", "vol": "{:.2%}", "largest wt": "{:.1%}", "beta": "{:.2f}", "HHI": "{:.3f}"}))
    with tabs[1]:
        sym = st.selectbox("Holding to reduce", held, key="r_s"); mx = float(v[sym])
        amt = st.slider("Sell amount (₹)", 0.0, mx, mx * 0.25, mx / 100 if mx else 1.0, key="r_a")
        if st.button("Run reduction (hold cash)") and amt > 0:
            S.scn.append(sc.simulate_reduction(ctx(), v, sym, amt))
    with tabs[2]:
        sym = st.selectbox("Sell", held, key="p_s"); mx = float(v[sym])
        amt = st.slider("Sell amount (₹)", 0.0, mx, mx * 0.5, mx / 100 if mx else 1.0, key="p_a")
        buys = st.multiselect("Buy (split equally)", pool, key="p_b"); cf = st.slider("Keep as cash (%)", 0, 100, 0) / 100
        if st.button("Run replacement") and amt > 0 and (buys or cf == 1):
            S.scn.append(sc.simulate_replacement(ctx(), v, {sym: amt}, {b_: 1 / len(buys) for b_ in buys} if buys else {}, cf))
    with tabs[3]:
        if not S.scn:
            st.info("Run a scenario first.")
        else:
            if st.button("Clear scenarios"):
                S.scn = []; st.rerun()
            last = S.scn[-1]
            st.subheader(last.label); [st.caption("• " + n) for n in last.notes]
            cmp_ = sc.compare(last.before, last.after)
            st.dataframe(cmp_.style.format({c_: "{:,.4f}" for c_ in cmp_.columns if c_ != "metric"}, na_rep="n/a"), width="stretch")
            for wmsg in last.after.get("warnings", []):
                st.warning(wmsg)
            if len(S.scn) > 0:
                st.subheader("All scenarios (ranked)")
                st.caption("Score = −Σ w·(relative change) over volatility (0.4), HHI (0.3), top-sector weight (0.2), beta (0.1). Positive ⇒ less risk/concentration. Ignores returns, valuation, tax.")
                st.dataframe(sc.rank_scenarios(S.scn), width="stretch")
                st.dataframe(sc.compare_scenarios(S.scn), width="stretch")
    with st.expander("Optional: constrained optimisation"):
        mw = st.slider("Max weight per stock", 0.1, 1.0, 0.35, 0.05); obj = st.selectbox("Objective", ["min_variance", "mean_variance", "max_sharpe"])
        if st.button("Optimise"):
            r, b = an.aligned_returns(S.prices[held], S.bench); mu = r.mean() * 252
            res = opt.optimize(mu, an.covariance(r), obj, mw, rf_annual=S.settings.risk_free_annual)
            if res["status"] != "optimal":
                st.error(res["message"])
            else:
                st.dataframe(pd.DataFrame({"optimised": res["weights"], "current": an.weights_from_values(v)}).style.format("{:.1%}"))
                st.caption(f"Vol {pct(res['volatility'])} · exp. return {pct(res['expected_return'])} (historical mean). {res['assumptions']}")

# ----------------------------------------------------------------- 7 AI analyst
elif page == pages[6]:
    st.header("AI analyst"); need_portfolio()
    import os
    if not os.environ.get("ANTHROPIC_API_KEY"):
        st.info("No ANTHROPIC_API_KEY configured: showing a deterministic summary. All quantitative features still work.")
    q = st.text_input("Ask", "What are the main risks in my portfolio?")
    if st.button("Analyse", type="primary"):
        tools = ai.AnalystTools(ctx(), values(), S.fund)
        with st.spinner("Computing…"):
            out = ai.ask(q, tools)
        S.ai_report = out["report"]
        if out["error"]:
            st.warning(out["error"])
        if out["unmatched_numbers"]:
            st.error(f"Numbers not traceable to calculations were detected and should be disregarded: {out['unmatched_numbers']}")
        elif out["mode"] == "llm":
            st.success("All numbers in the explanation matched calculated tool outputs.")
        for k_, items in out["report"].items():
            st.subheader(k_.replace("_", " ").title()); [st.write("• " + str(i)) for i in (items if isinstance(items, list) else [items])]

# ----------------------------------------------------------------- 8 Reports
elif page == pages[7]:
    st.header("Reports"); need_portfolio()
    v = values(); r, b = an.aligned_returns(S.prices[list(v.index)], S.bench)
    m = an.portfolio_metrics(an.weights_from_values(v), r, b, S.settings.risk_free_annual, S.meta["sectors"], S.settings.min_obs)
    cmp_ = sc.compare(S.scn[-1].before, S.scn[-1].after) if S.scn else None
    info = {"Mode": S.mode, "Last retrieval": datetime.fromtimestamp(S.meta["retrieved_at"]).strftime("%Y-%m-%d %H:%M"),
            "Latest observation": S.meta.get("latest_obs"), "Benchmark": S.settings.benchmark, "Risk-free (assumption)": f"{S.settings.risk_free_annual:.2%}",
            "Window": f"{m['window_start'].date()} to {m['window_end'].date()} ({m['n_obs']} obs)"}
    warns = list(S.meta.get("warnings", [])) + m["warnings"]
    st.download_button("⬇ Metrics (CSV)", report.to_csv(report.metrics_frame(m)), "portfolio_metrics.csv", "text/csv")
    if cmp_ is not None:
        st.download_button("⬇ Scenario comparison (CSV)", report.to_csv(cmp_), "scenario_comparison.csv", "text/csv")
    title = "Portfolio analysis report" + (" (SYNTHETIC DEMO DATA)" if is_demo() else "")
    st.download_button("⬇ Full report (PDF)", report.to_pdf(title, m, cmp_, S.ai_report, info, warns), "portfolio_report.pdf", "application/pdf")
    if st.button("Save analysis run to local database"):
        pid = db.save_portfolio(con(), S.user_id, "Session portfolio", [{"yahoo_ticker": t, "quantity": q_} for t, q_ in S.holdings.set_index("yahoo_ticker")["quantity"].items()])
        rid = db.save_analysis_run(con(), pid, S.settings.period, S.meta["retrieved_at"], m.get("return_convention", ""), warns, m); st.success(f"Saved run {rid}")

# ----------------------------------------------------------------- 9 Settings
else:
    st.header("Settings & data status"); s = S.settings
    s.benchmark = st.selectbox("Benchmark", list(config.BENCHMARKS.values()), index=list(config.BENCHMARKS.values()).index(s.benchmark))
    s.period = st.selectbox("Lookback", ["1y", "2y", "3y", "5y"], index=["1y", "2y", "3y", "5y"].index(s.period))
    s.risk_free_annual = st.number_input("Risk-free rate (annual, assumption)", 0.0, 0.2, s.risk_free_annual, 0.0025, format="%.4f")
    s.cost_bps = st.number_input("Transaction cost (bps of traded notional)", 0.0, 500.0, s.cost_bps, 5.0)
    s.var_conf = st.select_slider("VaR confidence", [0.9, 0.95, 0.99], s.var_conf)
    st.caption("Changing benchmark/lookback requires reloading the portfolio (Live mode).")
    st.subheader("Data status")
    st.write({"mode": S.mode, "last successful retrieval": datetime.fromtimestamp(S.meta["retrieved_at"]).isoformat(timespec="seconds") if S.meta.get("retrieved_at") else "—",
              "latest observation date (separate from retrieval time)": S.meta.get("latest_obs", "—"),
              "served from cache": S.meta.get("from_cache"), "stale cache fallback": S.meta.get("stale_cache"),
              "cache TTLs (app settings, not Yahoo guarantees)": {"historical_h": config.CacheTTL.historical / 3600, "latest_min": config.CacheTTL.latest / 60, "fundamentals_h": config.CacheTTL.fundamentals / 3600}})
    for w_ in S.meta.get("warnings", []):
        st.warning(w_)
    st.caption("Yahoo Finance via yfinance: unofficial, not guaranteed real-time/complete. Review Yahoo's terms before any commercial use.")
