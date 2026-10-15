"""SQLite persistence. Simulated trades are stored separately from actual holdings. Portfolio access is user-scoped."""
import json, sqlite3, time
from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS users(id TEXT PRIMARY KEY, created_at REAL);
CREATE TABLE IF NOT EXISTS securities(id INTEGER PRIMARY KEY, yahoo_ticker TEXT UNIQUE, name TEXT, exchange TEXT, sector TEXT);
CREATE TABLE IF NOT EXISTS security_mappings(user_symbol TEXT, exchange TEXT, yahoo_ticker TEXT, status TEXT, verified_at REAL, PRIMARY KEY(user_symbol, exchange));
CREATE TABLE IF NOT EXISTS portfolios(id INTEGER PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id), name TEXT, created_at REAL);
CREATE TABLE IF NOT EXISTS holdings(id INTEGER PRIMARY KEY, portfolio_id INTEGER NOT NULL REFERENCES portfolios(id), yahoo_ticker TEXT, quantity REAL, avg_price REAL, purchase_date TEXT, sector TEXT);
CREATE TABLE IF NOT EXISTS price_history(yahoo_ticker TEXT, date TEXT, close REAL, source TEXT, retrieved_at REAL, adjustment TEXT, PRIMARY KEY(yahoo_ticker, date));
CREATE TABLE IF NOT EXISTS benchmarks(symbol TEXT PRIMARY KEY, name TEXT);
CREATE TABLE IF NOT EXISTS scenarios(id INTEGER PRIMARY KEY, portfolio_id INTEGER NOT NULL REFERENCES portfolios(id), label TEXT, kind TEXT, created_at REAL);
CREATE TABLE IF NOT EXISTS scenario_trades(id INTEGER PRIMARY KEY, scenario_id INTEGER NOT NULL REFERENCES scenarios(id), yahoo_ticker TEXT, side TEXT, amount_inr REAL);
CREATE TABLE IF NOT EXISTS analysis_runs(id INTEGER PRIMARY KEY, portfolio_id INTEGER, period TEXT, data_snapshot_ts REAL, methodology TEXT, config_version TEXT, warnings TEXT, created_at REAL);
CREATE TABLE IF NOT EXISTS analysis_metrics(run_id INTEGER, name TEXT, value REAL);
CREATE TABLE IF NOT EXISTS ai_reports(id INTEGER PRIMARY KEY, run_id INTEGER, body TEXT, validated INTEGER, created_at REAL);
CREATE TABLE IF NOT EXISTS cache_blob(key TEXT PRIMARY KEY, payload TEXT, retrieved_at REAL);
"""


def connect(path=None):
    con = sqlite3.connect(path or config.DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con


def ensure_user(con, user_id):
    con.execute("INSERT OR IGNORE INTO users VALUES(?,?)", (user_id, time.time())); con.commit()


def save_portfolio(con, user_id, name, holdings):
    ensure_user(con, user_id)
    pid = con.execute("INSERT INTO portfolios(user_id,name,created_at) VALUES(?,?,?)", (user_id, name, time.time())).lastrowid
    for h in holdings:
        con.execute("INSERT INTO holdings(portfolio_id,yahoo_ticker,quantity,avg_price,purchase_date,sector) VALUES(?,?,?,?,?,?)",
                    (pid, h["yahoo_ticker"], h["quantity"], h.get("avg_price"), h.get("purchase_date"), h.get("sector")))
    con.commit()
    return pid


def list_portfolios(con, user_id):
    return [dict(r) for r in con.execute("SELECT * FROM portfolios WHERE user_id=?", (user_id,))]


def load_holdings(con, user_id, portfolio_id):
    """Only returns holdings if the portfolio belongs to user_id; otherwise PermissionError."""
    if not con.execute("SELECT 1 FROM portfolios WHERE id=? AND user_id=?", (portfolio_id, user_id)).fetchone():
        raise PermissionError("Portfolio not found for this user")
    return [dict(r) for r in con.execute("SELECT * FROM holdings WHERE portfolio_id=?", (portfolio_id,))]


def save_analysis_run(con, portfolio_id, period, snapshot_ts, methodology, warnings, metrics):
    rid = con.execute("INSERT INTO analysis_runs(portfolio_id,period,data_snapshot_ts,methodology,config_version,warnings,created_at) VALUES(?,?,?,?,?,?,?)",
                      (portfolio_id, period, snapshot_ts, methodology, config.CONFIG_VERSION, json.dumps(warnings), time.time())).lastrowid
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            con.execute("INSERT INTO analysis_metrics VALUES(?,?,?)", (rid, k, float(v)))
    con.commit()
    return rid


def cache_get(con, key, ttl=None):
    r = con.execute("SELECT payload, retrieved_at FROM cache_blob WHERE key=?", (key,)).fetchone()
    if not r:
        return None
    return json.loads(r["payload"]), r["retrieved_at"], (ttl is None or (time.time() - r["retrieved_at"]) <= ttl)


def cache_put(con, key, payload):
    ts = time.time()
    con.execute("INSERT OR REPLACE INTO cache_blob VALUES(?,?,?)", (key, json.dumps(payload), ts)); con.commit()
    return ts
