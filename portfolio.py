"""Portfolio ingestion, column mapping and validation. Invalid rows are reported, never silently dropped."""
from __future__ import annotations
import io
import numpy as np
import pandas as pd
from . import market_data as md

REQUIRED = ["symbol", "quantity"]
ALIASES = {
    "symbol": ["symbol", "ticker", "instrument", "stock", "scrip", "name"],
    "exchange": ["exchange", "exch"],
    "quantity": ["quantity", "qty", "shares", "qty.", "units"],
    "avg_price": ["avg_price", "average price", "avg price", "avg. cost", "avg cost", "average purchase price", "buy price", "avg"],
    "purchase_date": ["purchase_date", "purchase date", "date", "buy date"],
    "sector": ["sector"],
    "currency": ["currency", "ccy"],
}


def read_file(file_or_bytes, filename: str) -> pd.DataFrame:
    data = file_or_bytes.read() if hasattr(file_or_bytes, "read") else file_or_bytes
    if filename.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(data))
    return pd.read_csv(io.BytesIO(data))


def suggest_mapping(columns) -> dict:
    low = {c.strip().lower(): c for c in columns}
    out = {}
    for field, names in ALIASES.items():
        out[field] = next((low[n] for n in names if n in low), None)
    return out


def apply_mapping(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    out = pd.DataFrame({f: (df[c] if c in df.columns else None) for f, c in mapping.items() if c})
    for f in ALIASES:
        if f not in out:
            out[f] = None
    return out[list(ALIASES)]


def validate_holdings(df: pd.DataFrame) -> pd.DataFrame:
    """Adds `issues` column (list of strings). Rows with issues are flagged, not removed."""
    d = df.copy().reset_index(drop=True)
    issues = [[] for _ in range(len(d))]
    d["symbol"] = d["symbol"].astype(str).str.strip().str.upper().replace({"NAN": "", "NONE": ""})
    d["quantity"] = pd.to_numeric(d["quantity"], errors="coerce")
    d["avg_price"] = pd.to_numeric(d["avg_price"], errors="coerce")
    d["purchase_date"] = pd.to_datetime(d["purchase_date"], errors="coerce")
    raw_dates = df["purchase_date"] if "purchase_date" in df else pd.Series([None] * len(d))
    for i in range(len(d)):
        if not d.at[i, "symbol"]:
            issues[i].append("Missing symbol")
        q = d.at[i, "quantity"]
        if pd.isna(q) or q <= 0:
            issues[i].append("Invalid quantity")
        elif q != int(q):
            issues[i].append("Fractional quantity (unusual for Indian equities)")
        p = d.at[i, "avg_price"]
        if pd.notna(df["avg_price"].iloc[i] if "avg_price" in df else None) and (pd.isna(p) or p <= 0):
            issues[i].append("Invalid average price")
        if pd.notna(raw_dates.iloc[i]) and pd.isna(d.at[i, "purchase_date"]):
            issues[i].append("Invalid purchase date")
        cur = d.at[i, "currency"]
        if cur is not None and pd.notna(cur) and str(cur).upper() not in ("INR", "RS", "₹"):
            issues[i].append(f"Non-INR currency '{cur}' (MVP is INR only)")
    dup = d.duplicated(subset=["symbol", "exchange"], keep=False) & (d["symbol"] != "")
    for i in np.where(dup)[0]:
        issues[i].append("Duplicate holding (same symbol/exchange)")
    d["issues"] = issues
    return d


def merge_duplicates(d: pd.DataFrame) -> pd.DataFrame:
    """Explicit user action: combine duplicate rows with quantity-weighted average price."""
    def agg(g):
        q = g["quantity"].sum()
        ap = (g["quantity"] * g["avg_price"]).sum() / q if g["avg_price"].notna().all() else np.nan
        r = g.iloc[0].copy(); r["quantity"], r["avg_price"] = q, ap
        return r
    return d.groupby(["symbol", "exchange"], dropna=False, as_index=False, sort=False).apply(agg, include_groups=False).reset_index(drop=True)


def resolve_all(d: pd.DataFrame, confirmations: dict | None = None):
    """Resolve each row to a verified Yahoo ticker. Ambiguous / unavailable rows are returned for user action."""
    confirmations = confirmations or {}
    results = []
    for _, r in d.iterrows():
        ex = r.get("exchange") if pd.notna(r.get("exchange")) else None
        results.append(md.resolve_symbol(r["symbol"], ex, confirmed=confirmations.get(r["symbol"])))
    d = d.copy()
    d["yahoo_ticker"] = [m.yahoo_ticker for m in results]
    d["mapping_status"] = [m.status for m in results]
    d["mapping_note"] = [m.message + (f" Options: {m.candidates}" if m.status == "ambiguous" else "") for m in results]
    return d


def holdings_table(d: pd.DataFrame, last_prices: pd.Series, sectors: dict | None = None) -> pd.DataFrame:
    """Market value, weight and unrealised P/L (vs average cost where available)."""
    t = d.dropna(subset=["yahoo_ticker"]).copy()
    t["price"] = t["yahoo_ticker"].map(last_prices)
    t["market_value"] = t["quantity"] * t["price"]
    tot = t["market_value"].sum()
    t["weight"] = t["market_value"] / tot
    t["cost_value"] = t["quantity"] * t["avg_price"]
    t["unrealised_pl"] = t["market_value"] - t["cost_value"]
    t["unrealised_pl_pct"] = t["unrealised_pl"] / t["cost_value"]
    if sectors is not None:
        t["sector"] = t["yahoo_ticker"].map(sectors).fillna(t["sector"]) if "sector" in t else t["yahoo_ticker"].map(sectors)
    return t
