CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    mode TEXT NOT NULL,
    started_at TEXT NOT NULL,
    symbols TEXT NOT NULL,
    tf TEXT NOT NULL,
    params_json TEXT NOT NULL,
    initial_equity REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS equity (
    ts TEXT PRIMARY KEY,
    equity REAL NOT NULL,
    cash REAL NOT NULL,
    n_positions INTEGER NOT NULL,
    regimes_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS fills (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    size_base REAL NOT NULL,
    price REAL NOT NULL,
    fee REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS fills_symbol_ts ON fills(symbol, ts);
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    entry_ts TEXT NOT NULL,
    exit_ts TEXT NOT NULL,
    entry_price REAL NOT NULL,
    exit_price REAL NOT NULL,
    size_base REAL NOT NULL,
    pnl REAL NOT NULL,
    fees REAL NOT NULL,
    exit_reason TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS positions (
    symbol TEXT PRIMARY KEY,
    size_base REAL NOT NULL,
    entry_price REAL NOT NULL,
    stop_price REAL NOT NULL,
    opened_at TEXT NOT NULL,
    initial_risk REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS signals (
    ts TEXT NOT NULL,
    symbol TEXT NOT NULL,
    agent TEXT NOT NULL,
    direction TEXT NOT NULL,
    confidence REAL NOT NULL,
    reason TEXT NOT NULL,
    regime TEXT NOT NULL,
    PRIMARY KEY (ts, symbol, agent)
);
CREATE TABLE IF NOT EXISTS risk_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    kind TEXT NOT NULL,
    detail TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS rejections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    size_quote REAL NOT NULL,
    reason TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS state (
    key TEXT PRIMARY KEY,
    json TEXT NOT NULL
);
