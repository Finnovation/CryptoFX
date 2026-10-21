# CryptoFX Paper Trading and Web Dashboard — Design Spec

Date: 2026-10-21
Status: Approved in conversation ("Run it")
Owner: Kari
Builds on: `2026-10-20-crypto-trading-system-design.md` (Plan 1, branch `feat/backtest-core`)

## 1. Purpose

Give the system a memory and a face. A SQLite trade log and a paper trading
loop that runs continuously on live Binance candles, plus a web dashboard to
watch wallet, equity, open positions, past trades, win/loss stats, agent votes
and a live candlestick chart with the system's overlays and trade markers.

This replaces spec §4.8 "Reporter (static HTML)" with a served web page. The
go-live gate is shown read-only; the live executor stays in Plan 3.

Honesty rule carried over: the dashboard shows PAPER in large letters until a
live executor exists and the gate passes. Nothing on the page is estimated;
every number comes from the trade log.

## 2. Non-goals

- No live order placement. No exchange API keys anywhere in this plan.
- No user accounts, no multi-user. One optional bearer token.
- No websocket streaming; the page polls every 10 seconds.
- No frontend build chain. One static HTML file, vanilla JS, one chart
  library from a CDN.
- No LLMs.

## 3. Architecture

```
                 +------------------+
  Binance ------>|  DataFeed.sync   |
  (public)       +--------+---------+
                          | closed candles
                          v
   +----------+   +---------------+   +----------------+
   | Store    |<--|  PaperTrader  |-->| Engine.step()  |  (Plan 1 engine, unchanged
   | (SQLite) |   |  (loop)       |   | + PaperExecutor|   except signal exposure)
   +----+-----+   +---------------+   +----------------+
        |
        v
   +----------+      +----------------------+
   | FastAPI  |----->|  index.html          |
   | /api/*   |      |  lightweight-charts  |
   +----------+      +----------------------+
```

Two long-running processes, both started by the CLI: `cryptofx paper run`
(the trader) and `cryptofx web` (the dashboard). They share the SQLite file.
SQLite in WAL mode handles one writer and many readers.

## 4. Components

### 4.1 Store (`cryptofx/store/db.py`, `schema.sql`)

`Store(path: Path)` wraps `sqlite3` (stdlib), WAL mode, `check_same_thread=False`,
one connection per Store, schema applied on open (idempotent `CREATE TABLE IF
NOT EXISTS`).

Tables:

| table | columns |
|---|---|
| runs | id, mode, started_at, symbols, tf, params_json, initial_equity |
| equity | ts PK, equity, cash, n_positions, regimes_json |
| fills | id, ts, symbol, side, size_base, price, fee |
| trades | id, symbol, entry_ts, exit_ts, entry_price, exit_price, size_base, pnl, fees, exit_reason |
| positions | symbol PK, size_base, entry_price, stop_price, opened_at, initial_risk |
| signals | ts, symbol, agent, direction, confidence, reason, regime; PK (ts, symbol, agent) |
| risk_events | id, ts, kind, detail |
| state | key PK, json |
| rejections | id, ts, symbol, side, size_quote, reason |

Write API (each takes models from `core.models` or plain dicts):
`start_run(mode, symbols, tf, params, initial_equity) -> run_id`,
`append_equity(ts, equity, cash, n_positions, regimes)`,
`append_fills(list[Fill])`, `append_trades(list[Trade])`,
`replace_positions(list[Position])`, `append_signals(ts, symbol, regime,
dict[agent, Signal])`, `append_risk_events(list[dict])`,
`append_rejections(ts, list[tuple[Order, str]])`,
`save_state(key, obj)`, `load_state(key) -> obj | None`.

Read API: `status() -> dict` (last equity row, position count, last run,
risk state from state table), `equity_curve(limit=5000) -> list[dict]`,
`trades(limit=500) -> list[dict]`, `trade_stats() -> dict` (count, win_rate,
profit_factor, expectancy, avg_win, avg_loss, gross_pnl, fees, by_symbol,
by_exit_reason), `positions() -> list[dict]`, `latest_signals() -> list[dict]`
(most recent ts per symbol), `risk_events(limit=100)`, `fills(symbol,
since_ts) -> list[dict]`, `first_equity_ts()`.

All ts stored as ISO-8601 UTC strings; read back as tz-aware datetimes.

### 4.2 Executor and risk state persistence

`BacktestExecutor.to_state() -> dict` and `BacktestExecutor.from_state(d,
fee_rate, slippage_rate) -> BacktestExecutor` cover cash, positions, entry
fees, pending orders, fills, rejections, trades. Bars are not persisted; the
loop re-feeds the latest window on restart.

`RiskManager.to_state() / from_state()` cover state, peak_equity,
day_start_equity, day_start_ts, paused_until, events.

`PaperExecutor(BacktestExecutor)`: identical fill model, `mode = "paper"`.
Having a distinct class keeps the mode explicit in logs and the UI.

### 4.3 Engine signal exposure

`Engine.step` records `self.last_signals: dict[symbol, dict[agent, Signal]]`
and `self.last_regimes: dict[symbol, Regime | "gap"]` for the bar it just
processed. No other engine change.

### 4.4 PaperTrader (`cryptofx/paper/trader.py`)

`PaperTrader(feed, store, symbols, tf, params, initial_equity, warmup=200,
now_fn=None)`.

- `bootstrap()`: if `state` has `executor` and `risk`, restore them and the
  last processed ts; otherwise create fresh ones, `start_run("paper", ...)`.
  Builds the engine via `runner.build_engine` with the restored executor and
  risk manager.
- `run_once() -> bool`: adopt any risk state an operator changed in the
  store (see `risk reset`); `feed.sync` each symbol from `last_ts - warmup`
  bars; collect every ts shared by all symbols that is newer than `last_ts`
  (a fresh run takes only the newest bar); for each such ts in order, build
  the trailing `warmup` window per symbol, call `engine.step(windows)` and
  persist in one transaction (equity row, new fills, trades, rejections,
  signals, risk events, positions, executor and risk state, `last_ts`).
  Catch-up is what makes the paper record identical to a backtest of the
  same candles after downtime. Return True if any bar was stepped.
- `run_forever(sleep_fn=time.sleep)`: loop: `run_once()`; sleep until the
  next candle close plus `POLL_DELAY_S = 30`, polling every 60s after that
  until a new bar arrives (exchanges publish the closed candle a few seconds
  late).
- Any exception anywhere inside `run_once` (sync, cache, windowing, step,
  persist): write a `risk_event(kind="error")`, set HALTED on the *last
  persisted* risk snapshot, save that (never the mid-step executor), re-raise.
  Exit code non-zero. A supervisor restart resumes HALTED from the last
  consistent state and will not trade until `risk reset`; the interrupted bar
  then replays cleanly.
- Sync failures (`DataFeedError`) and insufficient history are not fatal: one
  `kind="data"` event at the start of an outage, one `kind="data_ok"` when it
  ends, skip the round.
- Resuming with different `symbols`, `tf` or params than the saved run is
  refused with a clear message.

### 4.5 Go-live gate (`cryptofx/risk/gate.py`), read-only in this plan

`GoLiveGate(store).evaluate() -> list[Criterion]` where `Criterion(name,
passed: bool, actual: str, required: str)`. Criteria from spec §9:
`days_paper >= 42`, `closed_trades >= 50`, `profit_factor > 1.3`,
`max_drawdown < 0.10`, `expectancy > 0`, `no HALTED events in 30 days`.
`passes()` is all-true. Nothing consumes `passes()` yet. The halt check is a
SQL query over `kind IN ('halted','error')` since the cutoff, so a flood of
other events cannot hide one.

### 4.6 Web API (`cryptofx/web/app.py`)

`create_app(store: Store, feed: DataFeed, tf: str, symbols: list[str],
token: str | None) -> FastAPI`.

| route | returns |
|---|---|
| GET / | `static/index.html` |
| GET /api/status | mode, risk_state, paused_until, equity, cash, n_positions, peak_equity, drawdown, last_ts, started_at, initial_equity, total_return, gate: list[Criterion] |
| GET /api/equity?limit | [{ts, equity, cash, n_positions}] |
| GET /api/trades?limit | {trades: [...], stats: trade_stats()} |
| GET /api/positions | [{symbol, size_base, entry_price, stop_price, opened_at, mark, unrealised_pnl, unrealised_pct}] (mark = last cached close) |
| GET /api/candles?symbol&limit=300 | {candles: [{time, open, high, low, close, volume}], overlays: {ema20, ema50, bb_upper, bb_lower, donchian_upper, donchian_lower}, markers: [{time, position, side, text}]} |
| GET /api/signals | latest_signals grouped by symbol with regime |
| GET /api/risk-events?limit | [...] |
| GET /api/symbols | configured symbols and tf |

`time` in candle payloads is unix seconds (what lightweight-charts wants).
If `token` is set, every `/api/*` route requires `Authorization: Bearer
<token>`; `/` is always served. Errors return JSON `{"error": ...}` with 4xx.

### 4.7 Page (`cryptofx/web/static/index.html`)

Single file, dark theme, mobile-first, no framework. Chart library:
`lightweight-charts` 4.x from unpkg (pinned version).

Layout top to bottom:

1. Header: PAPER badge (red LIVE badge reserved for Plan 3), equity, today's
   change, drawdown from peak, risk state chip (ACTIVE green / PAUSED amber /
   HALTED red), last bar time, "stale" warning if last bar is older than 2
   bars.
2. Wallet: cash, positions value, total, initial, total return, with the
   7-9% target and 0-5% realistic bands next to the monthly return.
3. Chart: symbol tabs, candlesticks, EMA20/EMA50 lines, Bollinger band,
   Donchian channel, buy/sell/stop markers, open position stop line.
4. Agents: per symbol, regime chip and one row per agent: direction,
   confidence bar, reason text. This is the "what is it thinking" panel.
5. Positions table with unrealised P&L.
6. Stats tiles: trades, win rate, profit factor, expectancy, avg win, avg
   loss, fees paid; then by-symbol and by-exit-reason breakdowns.
7. Trades table, newest first.
8. Risk and gate: risk events list and the gate checklist with pass/fail.

Polls `/api/status`, `/api/positions`, `/api/signals` every 10s; `/api/trades`,
`/api/equity`, `/api/risk-events` every 60s; `/api/candles` on symbol change
and every 60s. Token, if any, is read from `localStorage` after a one-time
prompt.

### 4.8 CLI additions

```
cryptofx paper run [--symbols --tf --equity --warmup --db --data-root --once]
cryptofx web [--host 127.0.0.1 --port 8000 --db --data-root --symbols --tf]
cryptofx risk reset --confirm --reason "..." [--db]
cryptofx gate [--db]
```

`--once` runs `run_once` and exits (for cron users and tests).

## 5. Stack additions

fastapi, uvicorn[standard], httpx (test client). Everything else is Plan 1.

## 6. Error handling

- Trader: see 4.4. Data errors skip a round; anything else halts and exits.
- Web: never crashes on an empty database; every endpoint returns an empty
  but well-formed payload before the first bar.
- Store: all multi-table writes in one transaction.

## 7. Security

- Binds 127.0.0.1 by default. Exposing it is a conscious `--host 0.0.0.0`.
- Optional bearer token from `CRYPTOFX_TOKEN`. Recommended phone access is
  Tailscale or an SSH tunnel; documented in README.
- No secrets in the database. No exchange keys exist in this plan.

## 8. Testing

- Store: round-trip every table; `trade_stats` on a known trade list;
  empty-db reads.
- Executor/risk state: `from_state(to_state())` reproduces behaviour on the
  next bar exactly (same fills, same equity).
- PaperTrader with a fake feed: first run bootstraps; a second `run_once`
  with no new bar returns False; a new bar steps once and persists; restart
  from the store resumes with identical state; a `DataFeedError` writes a
  data event and does not halt; an unexpected exception halts and persists.
- Gate: each criterion failing alone blocks.
- API via httpx `TestClient` against a seeded store: every route, empty db,
  token enforcement, candles payload shape and marker alignment.
- Page: a smoke test that `/` serves HTML containing the chart container and
  the PAPER badge. Visual check by hand.

## 9. Build order

1. Store + schema + state persistence on executor and risk manager.
2. Engine signal exposure + PaperExecutor + PaperTrader + `paper run`.
3. Gate + `risk reset` + `gate` commands.
4. Web API + tests.
5. Page.
