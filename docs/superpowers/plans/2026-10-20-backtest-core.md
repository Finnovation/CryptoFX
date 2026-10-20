# CryptoFX Backtest Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build spec steps 1 and 2: data feed, indicators, four strategy agents, allocator, risk manager, backtest executor, engine, metrics and walk-forward, ending with a CLI that prints an honest backtest number against BTC buy-and-hold.

**Architecture:** One `Engine` steps bar by bar: candles -> RegimeAgent -> strategy agents -> Allocator -> RiskManager -> Executor. The backtest executor fills pending orders at the next bar's open with fees and slippage, and checks stops against each bar's low. All money values are `float`; timestamps are tz-aware UTC `pd.Timestamp`.

**Tech Stack:** Python 3.12+, ccxt, pandas, numpy, pyarrow, pydantic v2, typer, pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-10-20-crypto-trading-system-design.md`

## Global Constraints

- Python >= 3.12. Package name `cryptofx`, src layout not used (package at repo root).
- No TA-Lib, no pandas-ta. Indicators are hand-written in `cryptofx/agents/indicators.py`.
- Agents are pure functions of candles: no I/O, no mutable state.
- Risk limits live only in `cryptofx/risk/limits.py` as module constants. No config file, no CLI flag overrides them.
- Fees 0.1% taker (`FEE_RATE = 0.001`), base slippage 0.05% (`SLIPPAGE_RATE = 0.0005`).
- Spot only: `Direction` has LONG and FLAT. No SHORT anywhere.
- Timeframes `1h` and `4h` only. Symbols default `BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT,XRP/USDT`.
- Candle DataFrame contract everywhere: columns `ts, open, high, low, close, volume` (+ `gap: bool` after `fill_gaps`), `ts` tz-aware UTC, sorted ascending, unique.
- Commits: no co-author trailers. Conventional prefixes (`feat:`, `test:`, `chore:`).

## Review Focus

1. Fewer candles than an indicator's lookback (e.g. 10 bars into a 50-EMA agent): every agent returns FLAT with reason `"insufficient data"`, never raises. Test in Tasks 4, 5, 6.
2. Gap down through the stop on the bar after entry: fill at `min(bar.open, stop_price)` minus slippage, not at the stop. Test in Task 9.
3. A bar with zero or NaN volume: slippage impact term must not divide by zero; impact is capped at 1%. Test in Task 9.
4. Equity shrinks so the computed order is below `MIN_ORDER_QUOTE` (10.0): RiskManager rejects with reason `"below min order"`, no dust position. Test in Task 8.
5. Exchange fetch fails on every retry during `sync`: raises `DataFeedError`, the on-disk cache is unchanged. Test in Task 3.

---

### Task 1: Scaffold and domain models

**Files:**
- Create: `pyproject.toml`, `cryptofx/__init__.py`, `cryptofx/core/__init__.py`, `cryptofx/core/models.py`, `tests/__init__.py`, `tests/test_models.py`, `.gitignore` (append `data/`, `reports/`, `*.db`)

**Interfaces:**
- Produces, all in `cryptofx/core/models.py`, pydantic v2 `BaseModel` with `model_config = ConfigDict(frozen=True)` unless noted:
  - `class Direction(str, Enum)`: `LONG = "long"`, `FLAT = "flat"`
  - `class Regime(str, Enum)`: `TRENDING`, `RANGING`, `CHAOS`
  - `class Side(str, Enum)`: `BUY`, `SELL`
  - `Signal(direction: Direction, confidence: float, reason: str)` with `confidence` constrained `ge=0, le=1`
  - `TargetPosition(symbol: str, size_quote: float, stop_price: float)`
  - `Order(symbol: str, side: Side, size_quote: float, stop_price: float | None, reason: str)`
  - `Fill(symbol: str, side: Side, size_base: float, price: float, fee: float, ts: datetime)`
  - `Rejected(reason: str)`
  - `Position(symbol: str, size_base: float, entry_price: float, stop_price: float, opened_at: datetime, initial_risk: float)` (not frozen; `stop_price` is updated by trailing stops)
  - `Trade(symbol: str, entry_ts: datetime, exit_ts: datetime, entry_price: float, exit_price: float, size_base: float, pnl: float, fees: float, exit_reason: str)`
  - `FLAT_SIGNAL = Signal(direction=Direction.FLAT, confidence=0.0, reason="insufficient data")`

- [ ] **Step 1: Write `pyproject.toml`**

`[project]` name `cryptofx`, version `0.1.0`, `requires-python = ">=3.12"`, dependencies `ccxt>=4.3`, `pandas>=2.2`, `numpy>=1.26`, `pyarrow>=15`, `pydantic>=2.6`, `typer>=0.12`. Optional `dev`: `pytest>=8`, `ruff>=0.4`. `[project.scripts] cryptofx = "cryptofx.cli:app"`. `[tool.pytest.ini_options] testpaths = ["tests"]`. `[tool.ruff] line-length = 100`. Build backend `setuptools`, `[tool.setuptools.packages.find] include = ["cryptofx*"]`.

- [ ] **Step 2: Install**

Run: `python -m venv .venv && . .venv/bin/activate && pip install -e ".[dev]"`
Expected: no errors, `python -c "import ccxt, pandas, pydantic, typer"` succeeds.

- [ ] **Step 3: Write failing tests**

```python
def test_signal_confidence_bounds():
    with pytest.raises(ValidationError):
        Signal(direction=Direction.LONG, confidence=1.5, reason="x")

def test_direction_has_no_short():
    assert {d.value for d in Direction} == {"long", "flat"}

def test_position_stop_is_mutable():
    p = Position(symbol="BTC/USDT", size_base=0.01, entry_price=100.0,
                 stop_price=90.0, opened_at=datetime(2024,1,1,tzinfo=UTC), initial_risk=10.0)
    p.stop_price = 95.0
    assert p.stop_price == 95.0
```

- [ ] **Step 4: Run tests, expect ImportError**
- [ ] **Step 5: Implement models per the Interfaces block**
- [ ] **Step 6: Run `pytest tests/test_models.py -v`, expect 3 passed; run `ruff check .`, expect clean**
- [ ] **Step 7: Commit** `chore: scaffold package and domain models`

---

### Task 2: Indicators

**Files:**
- Create: `cryptofx/agents/__init__.py`, `cryptofx/agents/indicators.py`, `tests/test_indicators.py`

**Interfaces:**
- Produces (all take/return `pd.Series`, index preserved, NaN where lookback not met):
  - `ema(close, n: int) -> pd.Series` using `ewm(span=n, adjust=False)`
  - `sma(close, n) -> pd.Series`
  - `rsi(close, n=14) -> pd.Series` Wilder: first avg gain/loss is SMA of first n changes, then `avg = (prev*(n-1) + cur)/n`
  - `true_range(high, low, close) -> pd.Series`
  - `atr(high, low, close, n=14) -> pd.Series` Wilder-smoothed true range
  - `adx(high, low, close, n=14) -> pd.Series` standard Wilder DI+/DI-/DX/ADX
  - `bollinger(close, n=20, k=2.0) -> tuple[pd.Series, pd.Series, pd.Series]` (mid, upper, lower), population std (`ddof=0`)
  - `donchian(high, low, n=20) -> tuple[pd.Series, pd.Series]` (upper = rolling max high, lower = rolling min low), excluding current bar (`shift(1)`)
  - `realised_vol(close, n=20) -> pd.Series` rolling std of log returns, `ddof=0`

- [ ] **Step 1: Write failing tests**

```python
def test_ema_hand_values():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    assert ema(s, 3).round(4).tolist() == [1.0, 1.5, 2.25, 3.125, 4.0625]

def test_rsi_hand_value():
    s = pd.Series([1,2,3,2,3,4,3,4,5,4,5,6,5,6,7], dtype=float)
    assert round(rsi(s, 14).iloc[14], 2) == 71.43
    assert rsi(s, 14).iloc[:14].isna().all()

def test_rsi_extremes():
    assert rsi(pd.Series(range(1, 31), dtype=float), 14).iloc[-1] == 100.0
    assert rsi(pd.Series(range(30, 0, -1), dtype=float), 14).iloc[-1] == 0.0

def test_true_range_hand_values():
    high = pd.Series([10.0, 12.0, 11.0]); low = pd.Series([8.0, 9.0, 7.0]); close = pd.Series([9.0, 11.0, 8.0])
    assert true_range(high, low, close).tolist()[1:] == [3.0, 4.0]   # max(12-9, |12-9|, |9-9|)=3 ; max(11-7, |11-11|, |7-11|)=4

def test_bollinger_hand_values():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    mid, up, lo = bollinger(s, 5, 2.0)
    assert mid.iloc[-1] == 3.0 and round(up.iloc[-1], 4) == 5.8284 and round(lo.iloc[-1], 4) == 0.1716

def test_donchian_excludes_current_bar():
    high = pd.Series([1.0, 2.0, 3.0, 10.0]); low = pd.Series([0.5, 1.0, 2.0, 9.0])
    up, lo = donchian(high, low, 3)
    assert up.iloc[-1] == 3.0 and lo.iloc[-1] == 0.5

def test_adx_strong_trend_vs_flat():
    n = 100
    trend = pd.Series(np.arange(n, dtype=float) * 1.0 + 100)
    flat = pd.Series(np.full(n, 100.0))
    assert adx(trend + 1, trend - 1, trend, 14).iloc[-1] > 25
    assert adx(flat + 0.1, flat - 0.1, flat, 14).iloc[-1] < 5 or np.isnan(adx(flat + 0.1, flat - 0.1, flat, 14).iloc[-1])
```

- [ ] **Step 2: Run, expect ImportError**
- [ ] **Step 3: Implement all functions in `indicators.py`**
- [ ] **Step 4: Run `pytest tests/test_indicators.py -v`, expect all pass**
- [ ] **Step 5: Commit** `feat: technical indicators`

---

### Task 3: Data cache and feed

**Files:**
- Create: `cryptofx/data/__init__.py`, `cryptofx/data/cache.py`, `cryptofx/data/feed.py`, `tests/test_data.py`

**Interfaces:**
- `cache.py`:
  - `CANDLE_COLUMNS = ["ts", "open", "high", "low", "close", "volume"]`
  - `cache_path(root: Path, exchange: str, symbol: str, tf: str) -> Path` -> `root/exchange/BTC-USDT/1h.parquet` (slash replaced by `-`)
  - `load(path: Path) -> pd.DataFrame | None`
  - `save(df: pd.DataFrame, path: Path) -> None` (creates parents, atomic via temp file + rename)
  - `merge(existing: pd.DataFrame | None, new: pd.DataFrame) -> pd.DataFrame` sorted by ts, dedup keep last
  - `fill_gaps(df: pd.DataFrame, tf: str) -> pd.DataFrame` reindexes to a full `pd.date_range` at the tf frequency, forward-fills OHLC from previous close (open=high=low=close=prev close, volume=0), adds `gap: bool`
  - `TF_TO_PANDAS = {"1h": "1h", "4h": "4h"}`
- `feed.py`:
  - `class DataFeedError(Exception)`
  - `FetchFn = Callable[[str, str, int, int], list[list]]` (symbol, tf, since_ms, limit) -> ccxt OHLCV rows
  - `class DataFeed:`
    - `__init__(self, exchange_id: str = "binance", root: Path = Path("data"), fetch_fn: FetchFn | None = None, retries: int = 3, backoff_s: float = 1.0)`; when `fetch_fn` is None, build `ccxt.binance({"enableRateLimit": True})` and use `fetch_ohlcv`
    - `sync(self, symbol: str, tf: str, since: datetime) -> int` returns rows added; loops in pages of 1000 from `max(since, last cached ts + 1 bar)`; retries each page `retries` times with `backoff_s * 2**attempt` sleep; on exhaustion raises `DataFeedError` without writing
    - `get_candles(self, symbol: str, tf: str, start: datetime | None = None, end: datetime | None = None) -> pd.DataFrame` loads cache, slices, applies `fill_gaps`; raises `DataFeedError` if no cache
    - `max_gap_bars(df) -> int` static helper: longest consecutive run of `gap == True`

- [ ] **Step 1: Write failing tests**

```python
def _rows(start_ms, n, step_ms=3_600_000):
    return [[start_ms + i*step_ms, 100+i, 101+i, 99+i, 100.5+i, 10.0] for i in range(n)]

def test_merge_sorts_and_dedups(): ...  # two overlapping frames -> unique sorted ts, last wins

def test_fill_gaps_marks_missing_hour():
    df = to_df(_rows(0, 2) + _rows(3*3_600_000, 1))   # hours 0,1,3
    out = fill_gaps(df, "1h")
    assert len(out) == 4 and out.gap.tolist() == [False, False, True, False]
    assert out.loc[2, "close"] == out.loc[1, "close"] and out.loc[2, "volume"] == 0

def test_sync_pages_and_caches(tmp_path):
    calls = []
    def fetch(symbol, tf, since_ms, limit):
        calls.append(since_ms)
        return _rows(since_ms, 1000) if since_ms < 1000*3_600_000 else []
    feed = DataFeed(root=tmp_path, fetch_fn=fetch, backoff_s=0)
    added = feed.sync("BTC/USDT", "1h", datetime(1970,1,1,tzinfo=UTC))
    assert added == 1000 and cache_path(tmp_path, "binance", "BTC/USDT", "1h").exists()
    assert feed.sync("BTC/USDT", "1h", datetime(1970,1,1,tzinfo=UTC)) == 0   # idempotent

def test_sync_failure_leaves_cache_untouched(tmp_path):
    feed = DataFeed(root=tmp_path, fetch_fn=lambda *a: (_ for _ in ()).throw(RuntimeError("net")), backoff_s=0)
    with pytest.raises(DataFeedError):
        feed.sync("BTC/USDT", "1h", datetime(2024,1,1,tzinfo=UTC))
    assert not cache_path(tmp_path, "binance", "BTC/USDT", "1h").exists()

def test_get_candles_slices_and_fills(tmp_path): ...  # start/end inclusive, gap column present
```

- [ ] **Step 2: Run, expect ImportError**
- [ ] **Step 3: Implement `cache.py` then `feed.py`**
- [ ] **Step 4: Run `pytest tests/test_data.py -v`, expect pass**
- [ ] **Step 5: Commit** `feat: candle cache and data feed`

---

### Task 4: TrendAgent and agent base

**Files:**
- Create: `cryptofx/agents/base.py`, `cryptofx/agents/trend.py`, `tests/synth.py`, `tests/test_trend_agent.py`

**Interfaces:**
- `base.py`: `class Agent(Protocol): name: str; def signal(self, candles: pd.DataFrame, regime: Regime) -> Signal`
- `trend.py`: `class TrendAgent: name = "trend"; __init__(self, fast: int = 20, slow: int = 50, adx_n: int = 14, adx_min: float = 20.0)`. LONG when `ema(fast) > ema(slow)` and `adx > adx_min` on the last bar; `confidence = min(1.0, (adx - adx_min) / 20 + 0.5)`; else FLAT with reason. Returns `FLAT_SIGNAL` if `len(candles) < slow + adx_n`.
- `tests/synth.py` helpers (used by later tasks): `uptrend(n=200, start=100.0, step=1.0, noise=0.0) -> DataFrame`, `downtrend(...)`, `ranging(n=200, centre=100.0, amp=5.0, period=20) -> DataFrame` (sine), `flat(n=200, price=100.0)`. All produce hourly `ts` from 2024-01-01 UTC, `high=close+0.5`, `low=close-0.5`, `open=prev close`, `volume=1000`.

- [ ] **Step 1: Write failing tests**

```python
def test_trend_long_in_uptrend():
    s = TrendAgent().signal(uptrend(), Regime.TRENDING)
    assert s.direction == Direction.LONG and s.confidence > 0.5

def test_trend_flat_in_downtrend():
    assert TrendAgent().signal(downtrend(), Regime.TRENDING).direction == Direction.FLAT

def test_trend_flat_on_short_history():
    s = TrendAgent().signal(uptrend(n=10), Regime.TRENDING)
    assert s == FLAT_SIGNAL
```

- [ ] **Step 2: Run, expect ImportError**
- [ ] **Step 3: Implement `synth.py`, `base.py`, `trend.py`**
- [ ] **Step 4: Run, expect pass**
- [ ] **Step 5: Commit** `feat: trend agent and synthetic candle helpers`

---

### Task 5: RegimeAgent

**Files:**
- Create: `cryptofx/agents/regime.py`, `tests/test_regime_agent.py`

**Interfaces:**
- `class RegimeAgent: name = "regime"; __init__(self, adx_n: int = 14, adx_trend: float = 25.0, vol_n: int = 20, vol_lookback: int = 200, chaos_pct: float = 0.95)`
- `classify(self, candles: pd.DataFrame) -> Regime`: CHAOS if last `realised_vol` is above the `chaos_pct` quantile of the trailing `vol_lookback` values; else TRENDING if last `adx > adx_trend`; else RANGING. Returns RANGING if `len(candles) < max(adx_n*2, vol_n)`.

- [ ] **Step 1: Write failing tests**

```python
def test_regime_trending(): assert RegimeAgent().classify(uptrend()) == Regime.TRENDING
def test_regime_ranging(): assert RegimeAgent().classify(ranging()) == Regime.RANGING
def test_regime_chaos_on_vol_spike():
    df = ranging(n=300)
    df.loc[df.index[-5:], "close"] *= [1.0, 1.3, 0.75, 1.3, 0.8]  # violent last bars
    assert RegimeAgent().classify(df) == Regime.CHAOS
def test_regime_short_history_is_ranging(): assert RegimeAgent().classify(uptrend(n=10)) == Regime.RANGING
```

- [ ] **Step 2: Run, expect ImportError**
- [ ] **Step 3: Implement**
- [ ] **Step 4: Run, expect pass**
- [ ] **Step 5: Commit** `feat: regime agent`

---

### Task 6: MeanRevAgent and BreakoutAgent

**Files:**
- Create: `cryptofx/agents/meanrev.py`, `cryptofx/agents/breakout.py`, `tests/test_meanrev_agent.py`, `tests/test_breakout_agent.py`

**Interfaces:**
- `class MeanRevAgent: name = "meanrev"; __init__(self, rsi_n=14, rsi_entry=30.0, rsi_exit=50.0, bb_n=20, bb_k=2.0)`. LONG when last `rsi < rsi_entry` and `close < bollinger lower`; confidence `= min(1.0, (rsi_entry - rsi) / rsi_entry + 0.5)`. Because agents are stateless, "exit when RSI > 50" is expressed as: LONG while `rsi < rsi_exit` AND the entry condition held within the last `bb_n` bars; otherwise FLAT. FLAT_SIGNAL if `len < max(rsi_n, bb_n) + 1`.
- `class BreakoutAgent: name = "breakout"; __init__(self, entry_n=20, exit_n=10)`. LONG when last `close > donchian(entry_n).upper`, or when a breakout occurred within the last `entry_n` bars and close is still above `donchian(exit_n).lower`; confidence `= min(1.0, 0.5 + (close - upper)/upper * 10)`. FLAT_SIGNAL if `len < entry_n + 1`.

- [ ] **Step 1: Write failing tests**

```python
def test_meanrev_long_after_oversold_dip():
    df = ranging(n=200); df.loc[df.index[-3:], "close"] -= [6, 9, 12]   # drive below lower band
    assert MeanRevAgent().signal(df, Regime.RANGING).direction == Direction.LONG
def test_meanrev_flat_in_uptrend(): assert MeanRevAgent().signal(uptrend(), Regime.RANGING).direction == Direction.FLAT
def test_meanrev_short_history(): assert MeanRevAgent().signal(ranging(n=5), Regime.RANGING) == FLAT_SIGNAL

def test_breakout_long_on_new_high():
    df = flat(n=100); df.loc[df.index[-1], ["close", "high"]] = [110.0, 110.5]
    assert BreakoutAgent().signal(df, Regime.TRENDING).direction == Direction.LONG
def test_breakout_flat_in_range(): assert BreakoutAgent().signal(flat(), Regime.RANGING).direction == Direction.FLAT
def test_breakout_short_history(): assert BreakoutAgent().signal(flat(n=5), Regime.TRENDING) == FLAT_SIGNAL
```

- [ ] **Step 2: Run, expect ImportError**
- [ ] **Step 3: Implement both agents**
- [ ] **Step 4: Run, expect pass**
- [ ] **Step 5: Commit** `feat: mean reversion and breakout agents`

---

### Task 7: Allocator

**Files:**
- Create: `cryptofx/core/allocator.py`, `tests/test_allocator.py`

**Interfaces:**
- `REGIME_WEIGHTS: dict[Regime, dict[str, float]]` exactly the spec table keyed by agent name (`trend`, `meanrev`, `breakout`).
- `class Allocator: __init__(self, threshold: float = 0.6, atr_n: int = 14, stop_atr_mult: float = 2.0)`
  - `score(self, signals: dict[str, Signal], regime: Regime) -> float` = sum over agents of `weight * confidence * (1 if LONG else 0)`, divided by sum of weights for that regime (so score is in [0,1]; CHAOS weights sum to 0 -> score 0).
  - `allocate(self, symbol: str, signals: dict[str, Signal], regime: Regime, candles: pd.DataFrame, equity: float, risk_fraction: float) -> TargetPosition | None`: None if `score < threshold`; else `stop_distance = stop_atr_mult * atr(...).iloc[-1]`, `stop_price = close - stop_distance`, `size_quote = risk_fraction * equity * close / stop_distance`. Returns None if ATR is NaN or 0.

- [ ] **Step 1: Write failing tests**

```python
def test_score_trending_all_long():
    sig = {k: Signal(direction=Direction.LONG, confidence=1.0, reason="") for k in ("trend","meanrev","breakout")}
    assert Allocator().score(sig, Regime.TRENDING) == pytest.approx(1.0)
def test_score_chaos_is_zero(): ...  # same signals, Regime.CHAOS -> 0.0
def test_score_trending_only_meanrev():
    # weights 1.0/0.3/1.0 ; only meanrev long at 1.0 -> 0.3/2.3
    assert Allocator().score({..}, Regime.TRENDING) == pytest.approx(0.3/2.3)
def test_allocate_sizes_by_atr_stop():
    df = uptrend(n=100)   # high-low = 1.0 every bar -> ATR ~ 1.0 (plus gap term)
    tp = Allocator(threshold=0.0).allocate("BTC/USDT", all_long, Regime.TRENDING, df, equity=1000.0, risk_fraction=0.01)
    close = df.close.iloc[-1]; a = atr(df.high, df.low, df.close, 14).iloc[-1]
    assert tp.stop_price == pytest.approx(close - 2*a) and tp.size_quote == pytest.approx(0.01*1000*close/(2*a))
def test_allocate_none_below_threshold(): ...
```

- [ ] **Step 2: Run, expect ImportError**
- [ ] **Step 3: Implement**
- [ ] **Step 4: Run, expect pass**
- [ ] **Step 5: Commit** `feat: allocator with regime weighting`

---

### Task 8: RiskManager

**Files:**
- Create: `cryptofx/risk/__init__.py`, `cryptofx/risk/limits.py`, `cryptofx/risk/manager.py`, `tests/test_risk_manager.py`

**Interfaces:**
- `limits.py` constants: `RISK_PER_TRADE = 0.01`, `MAX_OPEN_POSITIONS = 3`, `MAX_SYMBOL_FRACTION = 0.25`, `DAILY_LOSS_LIMIT = 0.03`, `DAILY_PAUSE_HOURS = 24`, `MAX_DRAWDOWN = 0.10`, `TRAIL_ATR_MULT = 1.5`, `TRAIL_ACTIVATE_R = 1.0`, `MIN_ORDER_QUOTE = 10.0`, `FEE_RATE = 0.001`, `SLIPPAGE_RATE = 0.0005`, `MAX_IMPACT = 0.01`, `MAX_GAP_BARS = 3`.
- `manager.py`:
  - `class RiskState(str, Enum)`: `ACTIVE`, `PAUSED`, `HALTED`
  - `Approved(order: Order)`, `Resized(order: Order, reason: str)`, `Rejected(reason: str)` (pydantic models; `Rejected` reuses `core.models.Rejected`)
  - `PortfolioView(equity: float, positions: list[Position])`
  - `class RiskManager:`
    - `__init__(self)`; attributes `state: RiskState`, `peak_equity: float | None`, `day_start_equity: float | None`, `day_start_ts`, `paused_until: datetime | None`, `events: list[dict]` (each `{"ts", "kind", "detail"}`)
    - `update_equity(self, ts: datetime, equity: float) -> None`: sets peak; rolls `day_start_equity` at UTC midnight; if `equity <= day_start_equity * (1 - DAILY_LOSS_LIMIT)` and ACTIVE -> PAUSED with `paused_until = ts + 24h`; if PAUSED and `ts >= paused_until` -> ACTIVE; if `equity <= peak_equity * (1 - MAX_DRAWDOWN)` -> HALTED (sticky). Appends an event on every transition.
    - `check(self, order: Order, view: PortfolioView) -> Approved | Resized | Rejected`: SELL orders are always Approved (never block an exit). BUY: Rejected if state != ACTIVE; Rejected `"max positions"` if `len(view.positions) >= MAX_OPEN_POSITIONS` and symbol not already held; Resized to `MAX_SYMBOL_FRACTION * equity` if larger; Rejected `"below min order"` if final `size_quote < MIN_ORDER_QUOTE`; Rejected `"missing stop"` if `order.stop_price is None`. Per-trade risk sizing is the Allocator's job (it reads `risk_fraction()` from here); the manager only enforces the limits above.
    - `trail_stop(self, pos: Position, close: float, atr_value: float) -> float | None`: if `close - pos.entry_price >= TRAIL_ACTIVATE_R * pos.initial_risk`, new stop `= max(pos.stop_price, close - TRAIL_ATR_MULT * atr_value)`; returns new stop if changed else None.
    - `reset(self, reason: str, ts: datetime) -> None`: HALTED -> ACTIVE, resets `peak_equity` to None, logs event with reason. Raises `ValueError` if reason is empty.
    - `risk_fraction(self) -> float` returns `RISK_PER_TRADE`.

- [ ] **Step 1: Write failing tests** (one per limit; names below, assertions per the Interfaces block)

`test_sell_always_approved`, `test_rejects_when_max_positions`, `test_allows_add_to_existing_symbol_at_max`, `test_resizes_over_symbol_fraction`, `test_rejects_below_min_order`, `test_rejects_missing_stop`, `test_daily_loss_pauses_24h_then_resumes`, `test_drawdown_halts_and_is_sticky`, `test_reset_requires_reason_and_clears_halt`, `test_trail_stop_activates_after_1R`, `test_trail_stop_never_lowers`.

- [ ] **Step 2: Run, expect ImportError**
- [ ] **Step 3: Implement `limits.py`, then `manager.py`**
- [ ] **Step 4: Run, expect pass**
- [ ] **Step 5: Commit** `feat: risk manager with hard limits`

---

### Task 9: BacktestExecutor

**Files:**
- Create: `cryptofx/exec/__init__.py`, `cryptofx/exec/base.py`, `cryptofx/exec/backtest.py`, `tests/test_backtest_executor.py`

**Interfaces:**
- `base.py`: `class Executor(Protocol)`: `submit(self, order: Order) -> Fill | Rejected`; `positions(self) -> list[Position]`; `equity(self) -> float`; `cash(self) -> float`; `trades(self) -> list[Trade]`; `on_bar(self, symbol: str, bar: pd.Series) -> list[Trade]` (sets the execution reference for `symbol` to this bar and processes stops; returns trades closed by stops); `set_stop(self, symbol: str, stop_price: float) -> None`.
- `backtest.py`: `class BacktestExecutor(initial_equity: float, fee_rate: float = FEE_RATE, slippage_rate: float = SLIPPAGE_RATE)`.
  - Fill price for BUY `= ref_open * (1 + slippage_rate + impact)`, SELL `= ref * (1 - slippage_rate - impact)`, where `impact = min(MAX_IMPACT, 0.1 * size_quote / (bar.volume * bar.open))` and `impact = MAX_IMPACT` if volume is 0 or NaN.
  - `submit` BUY: `size_base = size_quote / price`, `fee = size_quote * fee_rate`, cash decreases by `size_quote + fee`; Rejected `"insufficient cash"` if cash short. Opens or adds to Position (`initial_risk = entry_price - stop_price`). SELL: closes the whole position at fill price, records a `Trade` with `pnl = (exit - entry) * size_base - fees`, `exit_reason = order.reason`.
  - `on_bar`: for a held symbol, if `bar.low <= pos.stop_price`: exit at `min(bar.open, pos.stop_price) * (1 - slippage_rate)`, `exit_reason="stop"`.
  - `equity()` = cash + sum(size_base * last close seen for that symbol).

- [ ] **Step 1: Write failing tests**

```python
def test_buy_fill_includes_fee_and_slippage():
    ex = BacktestExecutor(1000.0); ex.on_bar("BTC/USDT", bar(open=100, high=101, low=99, close=100, volume=1e6))
    f = ex.submit(Order(symbol="BTC/USDT", side=Side.BUY, size_quote=100.0, stop_price=90.0, reason="t"))
    assert f.price == pytest.approx(100 * 1.0005, rel=1e-6) and f.fee == pytest.approx(0.1)
    assert ex.cash() == pytest.approx(1000 - 100 - 0.1)
def test_stop_gap_fills_at_open_not_stop():
    # entry 100 stop 90; next bar opens at 80 -> exit 80*(1-0.0005)
def test_zero_volume_uses_max_impact(): ...  # price == open*(1+0.0005+0.01)
def test_sell_records_trade_pnl(): ...
def test_insufficient_cash_rejected(): ...
def test_set_stop_raises_stop(): ...
```

- [ ] **Step 2: Run, expect ImportError**
- [ ] **Step 3: Implement**
- [ ] **Step 4: Run, expect pass**
- [ ] **Step 5: Commit** `feat: backtest executor with fees, slippage and stops`

---

### Task 10: Engine

**Files:**
- Create: `cryptofx/core/engine.py`, `tests/test_engine.py`

**Interfaces:**
- `class Engine: __init__(self, agents: list[Agent], regime_agent: RegimeAgent, allocator: Allocator, risk: RiskManager, executor: Executor, symbols: list[str], tf: str, warmup: int = 200)`
- `run(self, candles: dict[str, pd.DataFrame]) -> pd.DataFrame`: all frames must share the same `ts` index (engine aligns by inner join on ts and raises `ValueError` if fewer than `warmup + 1` bars remain). For each bar index `i >= warmup`, in order:
  1. For each symbol: `executor.on_bar(symbol, row_i)` (fills pending orders from step 4 of the previous bar at this bar's open, processes stops).
  2. Submit pending orders (created on bar `i-1`) through executor; collect fills.
  3. `risk.update_equity(ts_i, executor.equity())`.
  4. For each symbol, using candles `[:i+1]`: skip with FLAT if `DataFeed.max_gap_bars` over the last `warmup` bars `> MAX_GAP_BARS`; `regime = regime_agent.classify`; `signals = {a.name: a.signal(...)}`; if held: apply `risk.trail_stop` -> `executor.set_stop`; `target = allocator.allocate(...)`. If target and not held: `Order(BUY)` -> `risk.check` -> pending if Approved/Resized. If held and `allocator.score < threshold`: `Order(SELL, reason="signal exit")` -> pending.
  5. Append `{"ts", "equity", "cash", "n_positions", "regime_<symbol>"...}` to the equity curve.
- Returns equity curve DataFrame indexed by `ts`. `self.trades` exposes `executor.trades()`; `self.risk_events` exposes `risk.events`.

- [ ] **Step 1: Write failing tests**

```python
def test_engine_uptrend_single_symbol_profits():
    eng = Engine([TrendAgent()], RegimeAgent(), Allocator(), RiskManager(), BacktestExecutor(1000.0), ["BTC/USDT"], "1h", warmup=100)
    curve = eng.run({"BTC/USDT": uptrend(n=400)})
    assert curve.equity.iloc[-1] > 1000.0 and len(eng.trades) >= 0
def test_engine_orders_fill_next_bar_open(): ...  # with a stub agent that goes LONG at bar k, the fill ts == ts[k+1] and price ~ open[k+1]
def test_engine_skips_symbol_with_long_gap(): ...  # inject 5 consecutive gap bars -> no orders for that symbol in that window
def test_engine_deterministic(): ...  # two runs produce identical equity curves
def test_engine_raises_on_short_data(): ...
```

- [ ] **Step 2: Run, expect ImportError**
- [ ] **Step 3: Implement**
- [ ] **Step 4: Run, expect pass**
- [ ] **Step 5: Commit** `feat: bar-by-bar engine`

---

### Task 11: Metrics and benchmark

**Files:**
- Create: `cryptofx/backtest/__init__.py`, `cryptofx/backtest/metrics.py`, `tests/test_metrics.py`

**Interfaces:**
- `@dataclass(frozen=True) class Metrics: total_return: float; cagr: float; sharpe: float; sortino: float; max_drawdown: float; profit_factor: float; win_rate: float; avg_win: float; avg_loss: float; expectancy: float; trade_count: int; exposure: float; monthly_returns: pd.Series`
- `compute_metrics(equity: pd.Series, trades: list[Trade], n_positions: pd.Series | None = None) -> Metrics`: equity indexed by tz-aware ts. Daily resample (`last`) for Sharpe/Sortino, annualised with `sqrt(365)`, rf = 0; `profit_factor = inf` when no losses and wins exist, `0.0` when no trades; `exposure` = fraction of bars with `n_positions > 0` (0.0 if None); `monthly_returns` = month-end resample pct change.
- `buy_and_hold(close: pd.Series, initial_equity: float) -> pd.Series`
- `to_table(m: Metrics, name: str) -> str` one-column text table for CLI.

- [ ] **Step 1: Write failing tests**

```python
def test_max_drawdown_hand_value():
    eq = pd.Series([100, 120, 90, 130], index=daily_index(4))
    assert compute_metrics(eq, []).max_drawdown == pytest.approx(0.25)
def test_profit_factor_and_win_rate():
    trades = [trade(pnl=10), trade(pnl=20), trade(pnl=-15)]
    m = compute_metrics(flat_equity(), trades)
    assert m.profit_factor == pytest.approx(2.0) and m.win_rate == pytest.approx(2/3) and m.expectancy == pytest.approx(5.0)
def test_no_trades_profit_factor_zero(): ...
def test_sharpe_of_constant_growth_is_positive(): ...
def test_buy_and_hold_scales_by_close(): ...
```

- [ ] **Step 2: Run, expect ImportError**
- [ ] **Step 3: Implement**
- [ ] **Step 4: Run, expect pass**
- [ ] **Step 5: Commit** `feat: performance metrics and benchmark`

---

### Task 12: Runner and walk-forward

**Files:**
- Create: `cryptofx/backtest/runner.py`, `cryptofx/backtest/walkforward.py`, `tests/test_walkforward.py`

**Interfaces:**
- `runner.py`:
  - `@dataclass class Params: threshold: float = 0.6; trend_fast: int = 20; trend_slow: int = 50`
  - `build_engine(params: Params, executor: Executor, symbols: list[str], tf: str) -> Engine`
  - `@dataclass class BacktestResult: equity: pd.DataFrame; trades: list[Trade]; metrics: Metrics; benchmark: Metrics; params: Params; risk_events: list[dict]`
  - `run_backtest(candles: dict[str, pd.DataFrame], params: Params, initial_equity: float = 500.0, tf: str = "1h") -> BacktestResult` (benchmark is buy-and-hold of the first symbol, BTC by default)
- `walkforward.py`:
  - `windows(start: datetime, end: datetime, train_days: int = 180, test_days: int = 60) -> list[tuple[datetime, datetime, datetime]]` (train_start, test_start, test_end), rolling by `test_days`, last window truncated to `end`
  - `PARAM_GRID = [Params(threshold=t) for t in (0.5, 0.6, 0.7)]`
  - `@dataclass class WalkForwardResult: windows: list[dict]` (each: `train_start, test_start, test_end, chosen: Params, train_sharpe, test_metrics: Metrics`) `; stitched_equity: pd.Series; metrics: Metrics; benchmark: Metrics`
  - `walk_forward(candles, start, end, grid=PARAM_GRID, initial_equity=500.0, tf="1h", train_days=180, test_days=60) -> WalkForwardResult`: per window pick the grid entry with the best train Sharpe, run it on the test window starting from the equity the previous test window ended at, stitch test equity curves.

- [ ] **Step 1: Write failing tests**

```python
def test_windows_roll_by_test_days():
    w = windows(datetime(2024,1,1,tzinfo=UTC), datetime(2024,12,31,tzinfo=UTC))
    assert w[0] == (d(2024,1,1), d(2024,6,29), d(2024,8,28)) and w[1][1] == d(2024,8,28)
    assert w[-1][2] == d(2024,12,31)
def test_walk_forward_only_reports_test_windows(): ...  # stitched_equity ts all >= first test_start
def test_walk_forward_stitches_equity_continuously(): ...  # window k+1 starts at window k's final equity
def test_run_backtest_returns_benchmark(): ...
```

- [ ] **Step 2: Run, expect ImportError**
- [ ] **Step 3: Implement**
- [ ] **Step 4: Run, expect pass**
- [ ] **Step 5: Commit** `feat: backtest runner and walk-forward validation`

---

### Task 13: CLI and first real backtest

**Files:**
- Create: `cryptofx/cli.py`, `tests/test_cli.py`
- Modify: `README.md`

**Interfaces:**
- typer app `app` with sub-app `data` and commands:
  - `cryptofx data sync --symbols TEXT --tf TEXT --since TEXT` (defaults: spec symbols, `1h`, `2023-01-01`) prints rows added per symbol
  - `cryptofx backtest --from TEXT --to TEXT --symbols TEXT --tf TEXT --equity FLOAT --walk-forward/--no-walk-forward` prints `to_table` for system and benchmark side by side, plus monthly returns with a `target 7-9% / realistic 0-5%` reference line, and warns in red (typer `fg=RED`) when system Sharpe < benchmark Sharpe
- `tests/test_cli.py` uses `typer.testing.CliRunner` with a `DataFeed` pointed at a tmp cache seeded from `tests/synth.py` (monkeypatch `cryptofx.cli.DataFeed`).

- [ ] **Step 1: Write failing tests** `test_backtest_command_prints_metrics`, `test_backtest_warns_when_below_benchmark`, `test_data_sync_reports_rows`
- [ ] **Step 2: Run, expect ImportError**
- [ ] **Step 3: Implement `cli.py`; write README usage section (install, sync, backtest, the honesty paragraph from spec section 1)**
- [ ] **Step 4: Run `pytest -q`, expect all pass; `ruff check .` clean**
- [ ] **Step 5: Real run**

Run: `cryptofx data sync --since 2023-01-01` then `cryptofx backtest --from 2023-01-01 --to 2026-10-01 --walk-forward`
Expected: completes without error; record the printed system and benchmark tables in `docs/superpowers/results/2026-10-20-first-backtest.md` verbatim, whatever they say.

- [ ] **Step 6: Commit** `feat: cli with data sync and backtest; record first backtest`
