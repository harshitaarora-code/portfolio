"""Yahoo Finance adapter (yfinance), normalisation, quality checks, caching and ticker mapping.

Price series used: with auto_adjust=True yfinance returns 'Close' already adjusted for splits and dividends.
That single adjusted Close series is used everywhere; no further adjustment is applied.
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import StringIO
import numpy as np
import pandas as pd
import yfinance as yf
from . import config
from .db import cache_get, cache_put


class DataUnavailable(Exception):
    pass


def _raw_download(**kw):  # isolated so tests can mock the network
    return yf.download(**kw)


def _ticker(sym):
    return yf.Ticker(sym)


@dataclass
class DataQuality:
    duplicates_removed: int = 0
    invalid_removed: int = 0
    partial_coverage: dict = field(default_factory=dict)  # NaN count vs union of dates (not treated as missing sessions)
    long_gaps: dict = field(default_factory=dict)         # max calendar-day gap between observations
    latest_observation: dict = field(default_factory=dict)
    stale: dict = field(default_factory=dict)
    dropped_tickers: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


@dataclass
class PriceResult:
    prices: pd.DataFrame
    quality: DataQuality
    retrieved_at: float
    from_cache: bool = False
    stale_cache: bool = False


def normalize_download(raw, tickers, stale_days=config.DEFAULT_STALE_DAYS):
    """yf.download frame (single/multi ticker, flat/MultiIndex) -> wide adjusted-Close frame + quality report."""
    q = DataQuality()
    if raw is None or len(raw) == 0:
        raise DataUnavailable("Empty response from Yahoo Finance")
    if isinstance(raw.columns, pd.MultiIndex):
        lvl = next((i for i in range(raw.columns.nlevels) if "Close" in raw.columns.get_level_values(i)), None)
        if lvl is None:
            raise DataUnavailable("No 'Close' column in response")
        close = raw.xs("Close", axis=1, level=lvl)
        if isinstance(close, pd.Series):
            close = close.to_frame(tickers[0])
    else:
        if "Close" not in raw.columns:
            raise DataUnavailable("No 'Close' column in response")
        close = raw[["Close"]].copy()
        close.columns = [tickers[0]]
    close = close.apply(pd.to_numeric, errors="coerce")
    idx = pd.to_datetime(close.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    close.index = idx.normalize()
    close = close.sort_index()
    q.duplicates_removed = int(close.index.duplicated().sum())
    close = close[~close.index.duplicated(keep="last")]
    bad = (close <= 0) | ~np.isfinite(close)
    q.invalid_removed = int((bad & close.notna()).sum().sum())
    close = close.mask(bad)  # invalid -> NaN; never forward-filled
    for t in list(close.columns):
        if close[t].dropna().empty:
            q.dropped_tickers.append(t); q.warnings.append(f"{t}: no valid prices returned")
    close = close.drop(columns=q.dropped_tickers)
    if close.empty:
        raise DataUnavailable("No valid price data for any requested ticker")
    now = pd.Timestamp.now().normalize()
    for t in close.columns:
        s = close[t].dropna()
        q.partial_coverage[t] = int(close[t].isna().sum())
        q.latest_observation[t] = s.index[-1].date().isoformat()
        gaps = s.index.to_series().diff().dt.days.dropna()
        q.long_gaps[t] = int(gaps.max()) if len(gaps) else 0
        age = (now - s.index[-1]).days
        if age > stale_days:
            q.stale[t] = age; q.warnings.append(f"{t}: latest observation is {age} days old")
        if q.long_gaps[t] > 10:
            q.warnings.append(f"{t}: gap of {q.long_gaps[t]} calendar days in history (not filled)")
        if q.partial_coverage[t] > 0:
            q.warnings.append(f"{t}: {q.partial_coverage[t]} dates without a valid price (not filled; may be calendar differences)")
    if q.duplicates_removed:
        q.warnings.append(f"{q.duplicates_removed} duplicate date rows removed")
    if q.invalid_removed:
        q.warnings.append(f"{q.invalid_removed} invalid (<=0 / non-finite) prices set to missing")
    return close, q


def _frame_from_cache(c):
    df = pd.read_json(StringIO(c[0]["frame"]), orient="split"); df.index = pd.to_datetime(df.index)
    return df, DataQuality(**c[0]["quality"])


def get_historical_prices(symbols, period="3y", interval="1d", con=None, ttl=None, force_refresh=False) -> PriceResult:
    """Adjusted-Close history. Fresh cache first; on failure a clearly labelled stale cache; else DataUnavailable."""
    tickers = [symbols] if isinstance(symbols, str) else list(symbols)
    ttl = config.CacheTTL.historical if ttl is None else ttl
    key = f"hist|{','.join(sorted(tickers))}|{period}|{interval}|adj1"
    if con is not None and not force_refresh:
        c = cache_get(con, key, ttl)
        if c and c[2]:
            df, q = _frame_from_cache(c)
            return PriceResult(df, q, c[1], from_cache=True)
    try:
        raw = _raw_download(tickers=tickers if len(tickers) > 1 else tickers[0], period=period, interval=interval,
                            auto_adjust=True, progress=False, group_by="column", threads=True)
        df, q = normalize_download(raw, tickers)
    except Exception as e:
        stale = cache_get(con, key, None) if con is not None else None
        if stale:
            df, qq = _frame_from_cache(stale)
            qq.warnings.append(f"Yahoo Finance unavailable ({e}); showing cached data retrieved "
                               f"{datetime.fromtimestamp(stale[1], timezone.utc):%Y-%m-%d %H:%M} UTC")
            return PriceResult(df, qq, stale[1], from_cache=True, stale_cache=True)
        raise DataUnavailable(f"Yahoo Finance unavailable and no cached data: {e}") from e
    ts = time.time()
    if con is not None:
        cache_put(con, key, {"frame": df.to_json(orient="split", date_format="iso"), "quality": q.__dict__})
    return PriceResult(df, q, ts)


def get_latest_price(symbol):
    """Latest available price (NOT a real-time executable quote). Observation vs retrieval time reported separately."""
    retrieved = time.time()
    price = obs = method = None
    try:
        fi = _ticker(symbol).fast_info
        p = fi["last_price"]
        if p is not None and np.isfinite(p) and p > 0:
            price, method = float(p), "fast_info.last_price"  # provider exposes no observation timestamp here
    except Exception:
        pass
    if price is None:
        try:
            h = _ticker(symbol).history(period="5d", interval="1d", auto_adjust=False)
            h = h[h["Close"] > 0].dropna(subset=["Close"])
            if len(h):
                price, obs, method = float(h["Close"].iloc[-1]), h.index[-1], "history(5d) last close"
        except Exception:
            pass
    if price is None:
        raise DataUnavailable(f"No latest price available for {symbol}")
    return {"symbol": symbol, "price": price, "observation_time": str(obs) if obs is not None else None,
            "retrieved_at": retrieved, "method": method,
            "session": "latest available historical session" if obs is not None else "not established (provider gives no timestamp)",
            "note": "Possibly delayed/cached; not an executable quote."}


FUND_FIELDS = {"name": "longName", "sector": "sector", "industry": "industry", "market_cap": "marketCap",
               "trailing_pe": "trailingPE", "forward_pe": "forwardPE", "price_to_book": "priceToBook",
               "roe": "returnOnEquity", "profit_margin": "profitMargins", "revenue": "totalRevenue",
               "net_income": "netIncomeToCommon", "total_debt": "totalDebt", "total_cash": "totalCash",
               "dividend_yield": "dividendYield", "eps": "trailingEps"}


def normalize_fundamentals(info, symbol):
    """Unavailable values stay None (never 0). Third-party data: verify against official disclosures."""
    info = info or {}
    out = {"symbol": symbol, "retrieved_at": time.time(), "source": "Yahoo Finance via yfinance (third party)",
           "reporting_period": info.get("mostRecentQuarter")}
    for k, src in FUND_FIELDS.items():
        v = info.get(src)
        if isinstance(v, float) and not np.isfinite(v):
            v = None
        out[k] = v
    return out


def get_fundamentals(symbol, con=None, force_refresh=False):
    key = f"fund|{symbol}"
    if con is not None and not force_refresh:
        c = cache_get(con, key, config.CacheTTL.fundamentals)
        if c and c[2]:
            return c[0]
    try:
        f = normalize_fundamentals(_ticker(symbol).info, symbol)
    except Exception:
        f = normalize_fundamentals(None, symbol); f["error"] = "fundamentals unavailable"
    if con is not None and "error" not in f:
        cache_put(con, key, f)
    return f


# ---------------- ticker mapping ----------------
@dataclass
class MappingResult:
    user_symbol: str
    exchange: str | None
    status: str                      # verified | ambiguous | unavailable
    yahoo_ticker: str | None = None
    candidates: list = field(default_factory=list)
    message: str = ""


def candidate_tickers(symbol, exchange=None):
    s = symbol.strip().upper()
    if s.startswith("^") or s.endswith((".NS", ".BO")):
        return [s]
    ex = (exchange or "").strip().upper()
    if ex in ("NSE", "NS"):
        return [f"{s}.NS"]
    if ex in ("BSE", "BO"):
        return [f"{s}.BO"]
    return [f"{s}.NS", f"{s}.BO"]


def validate_ticker(ticker):
    """True only if a small download yields valid dates and positive prices."""
    try:
        raw = _raw_download(tickers=ticker, period="1mo", interval="1d", auto_adjust=True, progress=False)
        df, _ = normalize_download(raw, [ticker], stale_days=10_000)
        return len(df.dropna()) >= 3
    except Exception:
        return False


def resolve_symbol(symbol, exchange=None, confirmed=None) -> MappingResult:
    """Never silently chooses between listings: if >1 validates and no exchange was given, status is 'ambiguous'."""
    if symbol is None or not str(symbol).strip() or str(symbol).lower() == "nan":
        return MappingResult(str(symbol), exchange, "unavailable", message="Missing symbol")
    cands = candidate_tickers(str(symbol), exchange)
    if confirmed:
        if confirmed in cands and validate_ticker(confirmed):
            return MappingResult(symbol, exchange, "verified", confirmed, [confirmed], "User-confirmed")
        return MappingResult(symbol, exchange, "unavailable", message=f"{confirmed} could not be validated")
    ok = [c for c in cands if validate_ticker(c)]
    if len(ok) == 1:
        return MappingResult(symbol, exchange, "verified", ok[0], ok, "Validated by price download (company-name match not verified)")
    if len(ok) > 1:
        return MappingResult(symbol, exchange, "ambiguous", None, ok, "Valid on multiple exchanges; please confirm which listing you hold")
    return MappingResult(symbol, exchange, "unavailable", None, [], f"None of {cands} returned valid data")
