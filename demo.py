"""SYNTHETIC demonstration data (seeded GBM with sector factors). Never live market data; always labelled as such."""
import numpy as np
import pandas as pd

SYNTHETIC_LABEL = "SYNTHETIC DEMO DATA — not real market prices"
UNIVERSE = {  # ticker: (sector, start price)
    "TCS.NS": ("Information Technology", 3800), "INFY.NS": ("Information Technology", 1500),
    "HDFCBANK.NS": ("Financial Services", 1650), "ICICIBANK.NS": ("Financial Services", 1100),
    "RELIANCE.NS": ("Energy", 2900), "ITC.NS": ("Consumer Defensive", 450),
    "SUNPHARMA.NS": ("Healthcare", 1600), "LT.NS": ("Industrials", 3500), "HINDUNILVR.NS": ("Consumer Defensive", 2400),
}


def synthetic_prices(n_days=756, seed=7, end=None):
    rng = np.random.default_rng(seed)
    end = pd.Timestamp(end) if end is not None else pd.Timestamp.now().normalize()
    idx = pd.bdate_range(end=end, periods=n_days)
    mkt = rng.normal(0.0004, 0.009, n_days)
    secs = {s: rng.normal(0, 0.007, n_days) for s in {v[0] for v in UNIVERSE.values()}}
    data = {}
    for t, (sec, p0) in UNIVERSE.items():
        beta = rng.uniform(0.7, 1.2)
        r = beta * mkt + secs[sec] + rng.normal(0.0001, 0.008, n_days)
        data[t] = p0 * np.exp(np.cumsum(r))
    bench = 22000 * np.exp(np.cumsum(mkt))
    df = pd.DataFrame(data, index=idx)
    return df, pd.Series(bench, index=idx, name="^NSEI")


def sectors():
    return {t: v[0] for t, v in UNIVERSE.items()}
