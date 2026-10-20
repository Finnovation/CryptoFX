# CryptoFX Trading System — Design Spec

Date: 2026-10-20
Status: Draft for review
Owner: Kari

## 1. Purpose and honest framing

Build a crypto trading system that runs a swarm of rule-based strategy agents
behind a hard risk manager, proves or disproves an edge in backtest and paper
trading, and only then is allowed to trade a small live account.

Constraints that shape everything:

- Capital available for live trading: ~A$500. This is a go-live test fund,
  not income. Expected returns at this size are tens of dollars a month at best.
- User is in Australia. Binance public market data is fine for research and
  paper trading (no account needed). Live trading, when unlocked, uses Kraken
  via ccxt. OKX is the fallback.
- The system must make it hard to lose the $500 quickly. Risk limits are code,
  not config.
- Live trading is locked by default and only unlocks when the go-live gate
  (section 9) passes. No manual override flag.

Success for v1 is NOT "makes money". Success is: a working pipeline where
backtest, paper and live share one execution path, with honest metrics that
say whether an edge exists.

### Return target

Kari's stated target is 7-9% per month. For reference, that compounds to
125-180% per year; the best hedge fund in history averaged ~66%/year. A
realistic band for a spot-only, unleveraged retail system is 0-5% in good
months with flat or negative months in between.

The reporter shows monthly return against both the target band (7-9%) and the
realistic band (0-5%). Risk limits in section 4.4 are NOT loosened to chase
the target. If the system reaches the target inside those limits, the gate
will reflect it. If it does not, the reports say so plainly.

## 2. Non-goals (v1)

- No LLM in the signal path. No CrewAI/AutoGen swarm. Agents are deterministic
  Python functions.
- No short selling, no leverage, no futures, no options.
- No sub-hourly timeframes. Fees kill small accounts on scalping.
- No forex. Forex is a separate project after crypto proves out.
- No web app with auth. Reporting is a static HTML file.
- No hyperparameter optimisation beyond coarse walk-forward grids. Overfitting
  is the number one way this fails.

## 3. Architecture

```
            +-----------+     +------------------+
            | DataFeed  | --> | Strategy Agents  |  (trend, meanrev, breakout)
            | (ccxt)    |     | + Regime Agent   |
            +-----------+     +------------------+
                                       |
                                       v  signals
                              +----------------+
                              |   Allocator    |  votes -> target positions
                              +----------------+
                                       |
                                       v  target positions
                              +----------------+
                              |  RiskManager   |  veto / resize / halt
                              +----------------+
                                       |
                                       v  orders
                   +-------------------+-------------------+
                   |                   |                   |
             +-----------+     +-------------+     +-------------+
             | Backtest  |     |   Paper     |     |    Live     |
             | Executor  |     |  Executor   |     |  Executor   |
             +-----------+     +-------------+     +-------------+
                   \                   |                   /
                    \------------------+------------------/
                                       |
                              +----------------+
                              | TradeLog       |  SQLite
                              | Reporter       |  HTML + CLI
                              +----------------+
```

One `Engine` drives one bar at a time through this pipeline. The backtest
executor replays cached candles; paper and live use the same loop on a timer
with fresh candles. The only thing that differs between modes is the
`Executor` implementation.

## 4. Components

### 4.1 DataFeed

- Fetches OHLCV via ccxt from Binance public endpoints. Symbols: BTC/USDT,
  ETH/USDT, SOL/USDT, BNB/USDT, XRP/USDT. Timeframes: 1h, 4h.
- Caches to Parquet under `data/<exchange>/<symbol>/<tf>.parquet`. Incremental
  updates only fetch what is missing.
- Exposes `get_candles(symbol, tf, start, end) -> DataFrame` with columns
  `ts, open, high, low, close, volume`. Guarantees sorted, deduplicated, no
  gaps (gaps are forward-filled and flagged).
- Dependencies: ccxt, pandas, pyarrow.

### 4.2 Strategy Agents

Interface:

```python
class Agent(Protocol):
    name: str
    def signal(self, candles: pd.DataFrame, regime: Regime) -> Signal
```

`Signal` is `direction in {LONG, FLAT}`, `confidence in [0, 1]`, `reason: str`.
Agents are pure: same candles in, same signal out. No I/O, no state beyond
what they derive from candles.

v1 agents:

- **TrendAgent**: EMA(20) over EMA(50) with ADX(14) > 20 -> LONG.
- **MeanRevAgent**: RSI(14) < 30 and close below lower Bollinger(20, 2) -> LONG;
  exits when RSI > 50.
- **BreakoutAgent**: close above Donchian(20) high -> LONG; exits on close
  below Donchian(10) low.
- **RegimeAgent**: classifies TRENDING / RANGING / CHAOS from ADX and realised
  volatility percentile. Not a trader; its output weights the others.

Regime weights (v1 defaults, tested in walk-forward):

| Regime   | Trend | MeanRev | Breakout |
|----------|-------|---------|----------|
| TRENDING | 1.0   | 0.3     | 1.0      |
| RANGING  | 0.3   | 1.0     | 0.3      |
| CHAOS    | 0.0   | 0.0     | 0.0      |

### 4.3 Allocator

- Input: per-symbol list of weighted signals.
- Score = sum(weight * confidence * direction). LONG if score >= 0.6, else FLAT.
- Target position size = `risk_fraction * equity / stop_distance`, where stop
  distance is 2 * ATR(14). Risk fraction comes from RiskManager (default 1%).
- Output: `TargetPosition(symbol, size_quote, stop_price)` per symbol.

### 4.4 RiskManager

Hard-coded constants in `risk/limits.py`. Changing them requires a code change
and a test update.

- Max 1% of equity at risk per trade.
- Max 3 concurrent open positions.
- Max 25% of equity in any one symbol.
- Daily loss limit 3% of starting-day equity: stop opening new positions for
  24h.
- Max drawdown 10% from peak equity: halt all trading, set `HALTED` state,
  require `cryptofx risk reset --confirm` and a written reason logged.
- Every order passes `RiskManager.check(order, portfolio) -> Approved |
  Resized | Rejected(reason)`.
- Stop-loss on every position, placed at entry. Trailing stop at 1.5 * ATR
  once position is 1R in profit.

### 4.5 Executor

```python
class Executor(Protocol):
    def submit(self, order: Order) -> Fill | Queued | Rejected
    def on_bar(self, symbol: str, bar: pd.Series) -> list[Trade]
    def positions(self) -> list[Position]
    def pending(self) -> list[Order]
    def fills(self) -> list[Fill]
    def equity(self) -> float
```

Fill latency belongs to the executor, not the engine: a backtest or paper
executor queues on `submit` and fills on the next `on_bar`; a live executor
may fill immediately. The engine only reads `fills()` and `pending()`.

- **BacktestExecutor**: queues on submit, fills at the next bar's open, then
  stop-checks that same bar (a crash wick on the entry bar fires the stop).
  Forward-filled gap bars never fill orders or trigger stops. Fee 0.1% taker.
  Slippage 0.05% plus volume-impact term capped at 1%. Deterministic.
- **PaperExecutor**: same fill model as backtest but driven by live candles on
  a schedule. Writes to the same TradeLog with `mode=paper`.
- **LiveExecutor**: ccxt Kraken. Market orders only in v1, with a sanity check
  that the quoted price is within 1% of last close. Hard kill switch: a file
  `KILL` in the working directory stops all order submission within one loop.
  Constructor refuses to initialise unless `GoLiveGate.passes()` is true.

### 4.6 Engine

- `Engine(agents, regime_agent, allocator, risk, executor, symbols, tf, warmup)`.
- `step(windows)`: one closed bar per symbol (each window is the trailing
  `warmup` bars) -> executor.on_bar -> risk.update_equity -> regime ->
  signals -> allocate -> risk check -> executor.submit -> equity row.
- `run(candles)` drives `step` from cached candles; the paper/live loop drives
  it once per closed candle.
- Backtest: iterate bars. Paper/live: APScheduler or a plain loop with sleep
  aligned to candle close plus 30s.
- All state persists to SQLite so a restart resumes cleanly.

### 4.7 Backtester and metrics

- Walk-forward: train window 180 days, test window 60 days, roll by 60 days.
  Parameters are chosen on train, evaluated on test, only test results are
  reported. One executor and one RiskManager persist across test windows so
  open positions, drawdown peaks and halts carry over like a continuous run.
- Metrics per run and per agent: total return, CAGR, Sharpe (daily, rf=0),
  Sortino, max drawdown, profit factor, win rate, avg win/loss, trade count,
  exposure.
- Benchmark: buy-and-hold BTC over the same window. If the system cannot beat
  buy-and-hold on risk-adjusted terms, that is reported in red.

### 4.8 TradeLog and Reporter

- SQLite tables: `candles_meta`, `signals`, `orders`, `fills`, `positions`,
  `equity`, `risk_events`.
- `cryptofx report` renders `reports/latest.html`: equity curve vs benchmark,
  drawdown, open positions, per-agent contribution, risk state, go-live gate
  status. Plain HTML + inline SVG, no server.
- `cryptofx status` prints the same in the terminal.

## 5. CLI

```
cryptofx data sync --symbols BTC/USDT,ETH/USDT --tf 1h --since 2023-01-01
cryptofx backtest --from 2023-01-01 --to 2026-10-01 --walk-forward
cryptofx paper run            # long-running loop
cryptofx report
cryptofx status
cryptofx gate                 # prints go-live gate checklist
cryptofx risk reset --confirm --reason "..."
cryptofx live run             # refuses unless gate passes
```

## 6. Project layout

```
cryptofx/
  data/        feed.py, cache.py
  agents/      base.py, trend.py, meanrev.py, breakout.py, regime.py
  core/        engine.py, allocator.py, models.py
  risk/        manager.py, limits.py, gate.py
  exec/        base.py, backtest.py, paper.py, live.py
  backtest/    runner.py, metrics.py, walkforward.py
  store/       db.py, schema.sql
  report/      html.py, cli.py
  cli.py
tests/
docs/
```

## 7. Stack

Python 3.12+, ccxt, pandas, numpy, pyarrow, pydantic v2, typer (CLI), SQLite
via stdlib, pytest, ruff. No pandas-ta or TA-Lib; indicators are small
functions in `agents/indicators.py` so they are testable and dependency-free.

## 8. Error handling

- Exchange errors: retry 3x with backoff, then skip the bar and log a
  `risk_event`. Never guess a price.
- Data gaps: forward-fill, flag, and if a gap exceeds 3 bars, agents return
  FLAT for that symbol until data is clean.
- Executor failure in live: if an order's state is unknown after submit,
  query the exchange before any further action. If still unknown, halt.
- Any uncaught exception in the loop: log, set state HALTED, exit non-zero.
  Supervisor (systemd) restarts; engine resumes from SQLite in HALTED state
  and will not trade until reset.

## 9. Go-live gate

`GoLiveGate.passes()` returns true only when all hold, computed from the paper
TradeLog:

- At least 42 calendar days of continuous paper trading.
- At least 50 closed paper trades.
- Profit factor > 1.3.
- Max drawdown < 10%.
- Positive expectancy per trade after fees.
- No HALTED events in the last 30 days.

The gate is evaluated by code, printed by `cryptofx gate`, and checked again
by `LiveExecutor.__init__`. There is no bypass flag.

When live unlocks, initial live capital is capped in code at A$500 equivalent.

## 10. Testing

- Unit tests for every indicator against hand-computed values.
- Agent tests on synthetic candles (clear uptrend, clear range, flat) asserting
  expected signals.
- RiskManager tests for every limit, including the halt and reset path.
- Executor tests: backtest fill math, fee and slippage, paper equals backtest
  on identical candles.
- Engine integration test: 90 days of cached candles end to end, deterministic
  equity curve snapshot.
- Gate tests: each criterion failing individually blocks live.

## 11. Build order

1. Data feed + cache + indicators + TrendAgent + RiskManager + BacktestExecutor
   + metrics. Output: a first honest backtest number.
2. MeanRev, Breakout, Regime agents + Allocator + walk-forward.
3. SQLite store + PaperExecutor + paper loop.
4. Reporter (HTML + CLI) + gate command.
5. LiveExecutor (Kraken) behind gate + kill switch + systemd unit.

Each step lands with tests and a working CLI command.
