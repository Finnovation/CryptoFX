# CryptoFX Paper Trading and Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A SQLite trade log, a continuously running paper trader on live Binance candles, a FastAPI JSON API over the log, and a single-page dashboard with wallet, equity, positions, trades, win/loss stats, agent votes and a live candlestick chart.

**Architecture:** `PaperTrader` drives the Plan 1 `Engine.step` once per closed candle with a `PaperExecutor` (the backtest fill model), persisting every step to `Store`. `cryptofx web` serves `/api/*` from the same SQLite file plus one static page polling it.

**Tech Stack:** Python 3.12+, Plan 1 stack, fastapi, uvicorn, httpx, lightweight-charts 4.2.x via unpkg.

**Spec:** `docs/superpowers/specs/2026-10-21-paper-trading-and-dashboard-design.md`

## Global Constraints

- All of Plan 1's Global Constraints still hold (no LLMs, limits in `risk/limits.py`, spot only, no co-author trailers).
- Timestamps stored in SQLite as ISO-8601 UTC strings (`datetime.isoformat()`); read back as tz-aware `datetime`.
- SQLite via stdlib `sqlite3`, WAL mode, `check_same_thread=False`, one connection per `Store`.
- Web server binds `127.0.0.1:8000` by default. `/api/*` requires `Authorization: Bearer <token>` only when a token is configured.
- The page is one file `cryptofx/web/static/index.html`, no build step, chart library pinned to `https://unpkg.com/lightweight-charts@4.2.0/dist/lightweight-charts.standalone.production.js`.
- `POLL_DELAY_S = 30` after candle close; retry every 60 s until the closed bar appears.
- Mode strings: `"paper"` now; `"live"` reserved.

## Review Focus

1. Empty database: every `/api/*` route and `Store` read returns a well-formed empty payload, never a 500. Test in Tasks 1 and 6.
2. Trader restart mid-day: `PaperTrader.bootstrap()` from a saved state reproduces the exact executor and risk state, including pending orders and a HALTED flag, and does not double-process the last bar. Test in Tasks 2 and 4.
3. Exchange returns the closed candle late: `run_once` sees no new shared ts and returns False without writing anything. Test in Task 4.
4. A `DataFeedError` during `run_once` writes a `data` risk event and leaves risk state ACTIVE; any other exception halts, persists and re-raises. Test in Task 4.
5. Candle markers align: a fill at ts T appears as a marker at the candle whose `time` equals T, and a candle request for a symbol with no cache returns 404 JSON not a traceback. Test in Task 6.

---

### Task 1: Store and schema

**Files:**
- Create: `cryptofx/store/__init__.py`, `cryptofx/store/schema.sql`, `cryptofx/store/db.py`, `tests/test_store.py`
- Modify: `pyproject.toml` (add `fastapi>=0.111`, `uvicorn[standard]>=0.29`; dev `httpx>=0.27`; `[tool.setuptools.package-data] cryptofx = ["store/*.sql", "web/static/*"]`)

**Interfaces:**
- Produces `class Store:` in `db.py`:
  - `__init__(self, path: Path | str)`; `":memory:"` allowed. Applies `schema.sql`, sets `PRAGMA journal_mode=WAL`.
  - `start_run(self, mode: str, symbols: list[str], tf: str, params: dict, initial_equity: float) -> int`
  - `append_equity(self, ts: datetime, equity: float, cash: float, n_positions: int, regimes: dict[str, str]) -> None` (INSERT OR REPLACE on ts)
  - `append_fills(self, fills: list[Fill]) -> None`; `append_trades(self, trades: list[Trade]) -> None`
  - `replace_positions(self, positions: list[Position]) -> None`
  - `append_signals(self, ts: datetime, symbol: str, regime: str, signals: dict[str, Signal]) -> None`
  - `append_risk_events(self, events: list[dict]) -> None`; `append_rejections(self, ts: datetime, rejections: list[tuple[Order, str]]) -> None`
  - `save_state(self, key: str, obj: Any) -> None` (json.dumps, default=str); `load_state(self, key: str) -> Any | None`
  - `transaction(self)` context manager (BEGIN/COMMIT, rollback on error)
  - Reads: `status() -> dict` with keys `mode, started_at, initial_equity, symbols, tf, params, last_ts, equity, cash, n_positions` (None/0 when empty), `equity_curve(limit=5000) -> list[dict]` oldest first, `trades(limit=500) -> list[dict]` newest first, `trade_stats() -> dict` with keys `count, wins, losses, win_rate, profit_factor, expectancy, avg_win, avg_loss, gross_pnl, fees, by_symbol: dict[str, {count, pnl}], by_exit_reason: dict[str, {count, pnl}]`, `positions() -> list[dict]`, `latest_signals() -> list[dict]` (rows at the max ts per symbol, keys `ts, symbol, agent, direction, confidence, reason, regime`), `risk_events(limit=100) -> list[dict]` newest first, `fills(symbol: str | None = None, since: datetime | None = None) -> list[dict]` oldest first, `first_equity_ts() -> datetime | None`, `max_drawdown() -> float` from the equity table.
  - `profit_factor` uses the same rules as `metrics.compute_metrics` (0.0 no trades, inf no losses).

- [ ] **Step 1: Write failing tests**

```python
def test_schema_applies_twice(tmp_path): Store(p); Store(p)  # no error
def test_run_and_status_roundtrip(): start_run then status()["mode"] == "paper", symbols list restored, initial_equity 500.0
def test_empty_reads_are_well_formed(): status() has every key with None/0, equity_curve()==[], trades()==[], trade_stats()["count"]==0 and profit_factor==0.0, positions()==[], latest_signals()==[], risk_events()==[], first_equity_ts() is None, max_drawdown()==0.0
def test_equity_append_is_idempotent_on_ts(): append twice same ts -> one row, last wins
def test_fills_trades_positions_roundtrip(): append Fill/Trade/Position models, read dicts with tz-aware datetimes and same numbers
def test_trade_stats_hand_values(): pnls 10, 20, -15 -> win_rate 2/3, profit_factor 2.0, expectancy 5.0, by_symbol and by_exit_reason counts
def test_latest_signals_per_symbol(): two ts for BTC, one for ETH -> returns only max ts rows per symbol
def test_state_roundtrip_and_missing(): save_state("k", {"a": [1, 2]}); load_state("k") == {"a": [1, 2]}; load_state("nope") is None
def test_max_drawdown_from_equity(): 100,120,90,130 -> 0.25
def test_transaction_rolls_back(): inside transaction() append then raise -> row absent
```

- [ ] **Step 2: Run, expect ImportError**
- [ ] **Step 3: Write `schema.sql` (tables per spec §4.1), implement `Store`, add deps to `pyproject.toml` and `pip install -e ".[dev]"`**
- [ ] **Step 4: Run `pytest tests/test_store.py -v`, expect pass; `ruff check .` clean**
- [ ] **Step 5: Commit** `feat: sqlite trade log store`

---

### Task 2: Executor and risk state persistence, PaperExecutor

**Files:**
- Modify: `cryptofx/exec/backtest.py`, `cryptofx/risk/manager.py`
- Create: `cryptofx/exec/paper.py`, `tests/test_state_persistence.py`

**Interfaces:**
- `BacktestExecutor.to_state(self) -> dict` (JSON-safe: pydantic `model_dump(mode="json")` for models); `@classmethod from_state(cls, state: dict, fee_rate=FEE_RATE, slippage_rate=SLIPPAGE_RATE) -> Self`. Covers `_cash, _positions, _entry_fees, _pending, _fills, _rejections, _trades`. `_bars` is not persisted.
- `RiskManager.to_state(self) -> dict`; `@classmethod from_state(cls, state: dict) -> RiskManager`. Covers `state, peak_equity, day_start_equity, day_start_ts, paused_until, events`. Datetimes as ISO strings.
- `class PaperExecutor(BacktestExecutor): mode = "paper"`; `BacktestExecutor.mode = "backtest"`.

- [ ] **Step 1: Write failing tests**

```python
def test_executor_state_roundtrip_reproduces_next_bar():
    # open a position + queue a second order on executor A; B = from_state(A.to_state());
    # feed the same next bar to both -> identical fills, trades, cash, equity, positions
def test_executor_state_is_json_serialisable(): json.dumps(ex.to_state())
def test_risk_state_roundtrip_keeps_halt_and_pause(): HALTED with events -> from_state is HALTED, events equal, paused_until preserved
def test_paper_executor_mode(): PaperExecutor(100.0).mode == "paper" and BacktestExecutor(100.0).mode == "backtest"
```

- [ ] **Step 2: Run, expect AttributeError/ImportError**
- [ ] **Step 3: Implement**
- [ ] **Step 4: Run `pytest tests/test_state_persistence.py tests/test_backtest_executor.py tests/test_risk_manager.py -q`, expect pass**
- [ ] **Step 5: Commit** `feat: executor and risk state persistence, paper executor`

---

### Task 3: Engine signal exposure

**Files:**
- Modify: `cryptofx/core/engine.py`, `tests/test_engine.py`

**Interfaces:**
- `Engine.last_signals: dict[str, dict[str, Signal]]` and `Engine.last_regimes: dict[str, str]` set by every `step` (regime value or `"gap"`). Symbols skipped for gaps get `{}` signals.

- [ ] **Step 1: Write failing test** `test_engine_exposes_last_signals_and_regimes`: after `step`, `last_signals[SYM]` has one entry per agent name and `last_regimes[SYM]` in `{"trending","ranging","chaos","gap"}`.
- [ ] **Step 2: Run, expect AttributeError**
- [ ] **Step 3: Implement (set in `_decide`, reset at top of `step`)**
- [ ] **Step 4: Run `pytest tests/test_engine.py -q`, expect pass**
- [ ] **Step 5: Commit** `feat: engine exposes last signals and regimes`

---

### Task 4: PaperTrader and `paper run`

**Files:**
- Create: `cryptofx/paper/__init__.py`, `cryptofx/paper/trader.py`, `tests/test_paper_trader.py`
- Modify: `cryptofx/cli.py`

**Interfaces:**
- `POLL_DELAY_S = 30`, `RETRY_S = 60`, `STATE_EXECUTOR = "executor"`, `STATE_RISK = "risk"`, `STATE_LAST_TS = "last_ts"`.
- `class PaperTrader: __init__(self, feed: DataFeed, store: Store, symbols: list[str], tf: str, params: Params, initial_equity: float, warmup: int = DEFAULT_WARMUP, now_fn: Callable[[], datetime] | None = None, log: Callable[[str], None] = print)`
  - `bootstrap(self) -> None`: restore or create executor (`PaperExecutor`), risk, `last_ts`; build engine via `build_engine(params, executor, symbols, tf, warmup, risk=risk)`; `start_run` only when fresh.
  - `run_once(self) -> bool` per spec §4.4. Uses `feed.sync(sym, tf, since)` with `since = last_ts - warmup*tf` (or `now - 2*warmup*tf` when fresh), then `feed.get_candles(sym, tf)` tail `warmup` rows. New shared ts must be `> last_ts`.
  - `persist(self, ts, step_row)`: one `store.transaction()` writing equity, new fills/trades/rejections since the last persist (track counts), signals for every symbol, new risk events, positions, states, last_ts.
  - `next_wake(self, now: datetime) -> datetime`: next candle close after `now` plus `POLL_DELAY_S`.
  - `run_forever(self, sleep_fn=time.sleep, max_iterations: int | None = None) -> None` per spec §4.4.
- CLI: `cryptofx paper run --symbols --tf --equity --warmup --db cryptofx.db --data-root data --once`. `--once` prints `"stepped <ts>"` or `"no new bar"`.

- [ ] **Step 1: Write failing tests** (fake feed = `DataFeed(root=tmp, fetch_fn=...)` serving a growing synthetic series; `now_fn` controlled)

```python
def test_bootstrap_fresh_starts_run(): store.status()["mode"] == "paper", state keys saved after first run_once
def test_run_once_no_new_bar_returns_false_and_writes_nothing()
def test_run_once_new_bar_steps_and_persists(): equity row, signals rows for each symbol/agent, positions replaced, last_ts advanced
def test_restart_resumes_identical_state(): trader A runs 3 bars; trader B bootstraps from the same store, runs 1 bar; trader A runs the same bar -> same equity row and same fills
def test_data_error_writes_event_and_stays_active()
def test_unexpected_error_halts_and_persists(): patch engine.step to raise -> risk_events has kind "error", load_state("risk")["state"] == "halted", exception re-raised
def test_next_wake_is_close_plus_delay(): 1h tf, now 10:17 -> 11:00:30
def test_cli_paper_once(monkeypatch): CliRunner with patched DataFeed -> exit 0, output contains "stepped" or "no new bar"
```

- [ ] **Step 2: Run, expect ImportError**
- [ ] **Step 3: Implement trader and CLI command**
- [ ] **Step 4: Run `pytest tests/test_paper_trader.py -q`, then full `pytest -q`, expect pass; ruff clean**
- [ ] **Step 5: Commit** `feat: paper trading loop`

---

### Task 5: Go-live gate, `gate` and `risk reset` commands

**Files:**
- Create: `cryptofx/risk/gate.py`, `tests/test_gate.py`
- Modify: `cryptofx/cli.py`

**Interfaces:**
- `@dataclass(frozen=True) class Criterion: name: str; passed: bool; actual: str; required: str`
- `class GoLiveGate: __init__(self, store: Store)`; `evaluate(self) -> list[Criterion]` in this order and with these names: `paper_days` (>= 42, from `first_equity_ts` to `status()["last_ts"]`), `closed_trades` (>= 50), `profit_factor` (> 1.3), `max_drawdown` (< 0.10), `expectancy` (> 0), `no_halt_30d` (no risk_event kind `halted` within 30 days of last_ts); `passes(self) -> bool`.
- CLI `cryptofx gate --db` prints one line per criterion `[PASS]`/[FAIL] and exits 1 when not passing. `cryptofx risk reset --confirm --reason TEXT --db`: loads risk state, requires HALTED or PAUSED, calls `RiskManager.reset`, saves state and appends the reset event; refuses without `--confirm` or with an empty reason.

- [ ] **Step 1: Write failing tests**: `test_each_criterion_blocks_alone` (seed a passing store, then break one criterion at a time), `test_empty_store_fails_all`, `test_cli_gate_exit_codes`, `test_cli_risk_reset_requires_confirm_and_reason`, `test_cli_risk_reset_clears_halt`.
- [ ] **Step 2: Run, expect ImportError**
- [ ] **Step 3: Implement**
- [ ] **Step 4: Run `pytest tests/test_gate.py -q`, expect pass**
- [ ] **Step 5: Commit** `feat: go-live gate evaluation and risk reset command`

---

### Task 6: Web API

**Files:**
- Create: `cryptofx/web/__init__.py`, `cryptofx/web/app.py`, `cryptofx/web/static/.gitkeep`, `tests/test_web_api.py`
- Modify: `cryptofx/cli.py`

**Interfaces:**
- `create_app(store: Store, feed: DataFeed, tf: str, symbols: list[str], token: str | None = None) -> FastAPI` with the routes of spec §4.6. Overlays computed with `indicators.ema(20/50)`, `bollinger(20, 2)`, `donchian(20)` on the last `limit` candles plus 60 extra for warmup, trimmed to `limit`. Markers: BUY -> `{"position": "belowBar", "shape": "arrowUp", "color": "#22c55e", "text": "buy"}`, SELL -> `aboveBar/arrowDown/#ef4444/"sell"`, stop exits (`Trade.exit_reason == "stop"`) -> `aboveBar/circle/#f59e0b/"stop"`. Positions endpoint `mark` = last cached close for the symbol, `unrealised_pnl = (mark - entry) * size_base`.
- `/api/status` also includes `risk_state`, `paused_until`, `peak_equity`, `drawdown` (from the risk state saved in the store, defaults `active`, None, None, 0.0) and `gate: [Criterion as dict]`, `total_return`.
- Token: dependency reading `Authorization` header; 401 JSON `{"error": "unauthorized"}`.
- CLI `cryptofx web --host 127.0.0.1 --port 8000 --db --data-root --symbols --tf`; token from env `CRYPTOFX_TOKEN`; runs `uvicorn.run(app, host, port)`.

- [ ] **Step 1: Write failing tests** with `fastapi.testclient.TestClient`:

```python
def test_status_empty_db_is_well_formed()
def test_status_after_seed_has_equity_and_gate()
def test_equity_trades_positions_signals_risk_events_routes()
def test_candles_payload_shape_and_marker_alignment(): seed cache + a fill at ts T -> markers[0]["time"] == unix(T) and that time exists in candles
def test_candles_unknown_symbol_404_json()
def test_token_enforced_only_when_configured(): no token -> 200; token set and missing header -> 401; correct header -> 200; "/" always 200
def test_index_served(): "/" returns text/html containing 'id="chart"' and "PAPER"
```

- [ ] **Step 2: Run, expect ImportError**
- [ ] **Step 3: Implement `app.py` and the CLI command; create a placeholder `index.html` containing `<div id="chart">` and `PAPER` (Task 7 replaces it)**
- [ ] **Step 4: Run `pytest tests/test_web_api.py -q`, full suite, ruff**
- [ ] **Step 5: Commit** `feat: web api over the trade log`

---

### Task 7: Dashboard page

**Files:**
- Modify: `cryptofx/web/static/index.html`
- Modify: `README.md` (run paper + web, phone access via Tailscale/SSH tunnel, token)

**Interfaces:**
- Consumes every `/api/*` route from Task 6. Layout and polling per spec §4.7. Colors: background `#0b0f14`, panel `#121821`, text `#e6edf3`, muted `#8b98a5`, green `#22c55e`, red `#ef4444`, amber `#f59e0b`, accent `#3b82f6`.
- Chart: `LightweightCharts.createChart(el, {layout dark, autoSize})`, candlestick series + 2 EMA line series + 2 BB lines + 2 Donchian lines, `setMarkers`, a price line for the open position stop.
- Token handling: if any `/api/*` returns 401, `prompt()` once for the token, store in `localStorage.cryptofx_token`, retry.
- Staleness: header shows "STALE" in amber when `now - last_ts > 2 * tf`.

- [ ] **Step 1: Extend `test_index_served` to assert the page references `lightweight-charts@4.2.0`, contains ids `chart, agents, trades, positions, stats, gate, wallet, equity` and the string `PAPER`**
- [ ] **Step 2: Run, expect failure on the ids**
- [ ] **Step 3: Write the page**
- [ ] **Step 4: Run the suite; then start `cryptofx web` against a store seeded by `tests` helpers and screenshot `/` with Playwright (Chromium at `/opt/pw-browsers`), save to `docs/superpowers/results/2026-10-21-dashboard.png`, eyeball it**
- [ ] **Step 5: Commit** `feat: dashboard page`
