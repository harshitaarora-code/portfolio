import os
from dataclasses import dataclass

CONFIG_VERSION = "1.0"
TRADING_DAYS = 252
DB_PATH = os.environ.get("PFAI_DB_PATH", "pfai.sqlite")

BENCHMARKS = {"NIFTY 50": "^NSEI", "NIFTY BANK": "^NSEBANK", "SENSEX": "^BSESN", "NIFTY NEXT 50": "^NSMIDCP"}


class CacheTTL:
    """Application cache lifetimes in seconds. NOT guarantees about Yahoo's own update cadence."""
    historical = 6 * 3600
    latest = 5 * 60
    fundamentals = 24 * 3600
    mapping = 7 * 24 * 3600


@dataclass
class Settings:
    benchmark: str = "^NSEI"
    period: str = "3y"
    risk_free_annual: float = 0.065  # user-adjustable assumption (approx. Indian 10y G-sec yield)
    min_obs: int = 120               # below this, estimates are flagged unstable
    stale_days: int = 7
    cost_bps: float = 25.0           # assumed cost per traded notional (brokerage, STT, charges)
    var_conf: float = 0.95


DEFAULT_STALE_DAYS = Settings.stale_days

DISCLAIMER = ("Analytical decision-support tool. Not investment advice and no guarantee of future performance. "
              "Market data comes from Yahoo Finance via yfinance (third party, possibly delayed or incomplete); "
              "verify against official exchange/company disclosures before acting.")
