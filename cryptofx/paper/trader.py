"""Paper trading loop: one Engine.step per closed candle, everything persisted.

The trader is the only writer of the SQLite store. Restarting it resumes from
the saved executor and risk state, so a halt survives a reboot. Missed bars
(downtime, a late exchange) are caught up in order so the paper record is the
same one a backtest of the same candles would produce."""

import json
import time
import traceback
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime, timedelta

import pandas as pd

from cryptofx.backtest.runner import DEFAULT_WARMUP, Params, build_engine
from cryptofx.data.feed import TF_MS, DataFeed, DataFeedError
from cryptofx.exec.paper import PaperExecutor
from cryptofx.risk.manager import RiskManager, RiskState
from cryptofx.store.db import Store

POLL_DELAY_S = 30
RETRY_S = 60
STATE_EXECUTOR = "executor"
STATE_RISK = "risk"
STATE_LAST_TS = "last_ts"


def _canon(state: dict | None) -> dict | None:
    return json.loads(json.dumps(state)) if state is not None else None


class PaperTrader:
    def __init__(
        self,
        feed: DataFeed,
        store: Store,
        symbols: list[str],
        tf: str,
        params: Params,
        initial_equity: float,
        warmup: int = DEFAULT_WARMUP,
        now_fn: Callable[[], datetime] | None = None,
        log: Callable[[str], None] = print,
    ) -> None:
        self.feed = feed
        self.store = store
        self.symbols = symbols
        self.tf = tf
        self.params = params
        self.initial_equity = initial_equity
        self.warmup = warmup
        self.now_fn = now_fn or (lambda: datetime.now(UTC))
        self.log = log
        self.tf_delta = timedelta(milliseconds=TF_MS[tf])
        self.last_ts: datetime | None = None
        self._risk_saved: dict | None = None
        self._outage = False

    # -- lifecycle ------------------------------------------------------
    def bootstrap(self) -> None:
        ex_state = self.store.load_state(STATE_EXECUTOR)
        risk_state = self.store.load_state(STATE_RISK)
        last = self.store.load_state(STATE_LAST_TS)
        if ex_state is not None and risk_state is not None:
            self._check_resume_config()
            executor = PaperExecutor.from_state(ex_state)
            risk = RiskManager.from_state(risk_state)
            self.last_ts = datetime.fromisoformat(last) if last else None
            self.log(f"resumed paper run: last bar {self.last_ts}, risk {risk.state.value}")
        else:
            executor = PaperExecutor(self.initial_equity)
            risk = RiskManager()
            self.store.start_run("paper", self.symbols, self.tf, asdict(self.params),
                                 self.initial_equity)
            self.log(f"started fresh paper run with {self.initial_equity:.2f}")
        self._risk_saved = _canon(risk_state)
        self.engine = build_engine(self.params, executor, self.symbols, self.tf, self.warmup,
                                   risk=risk)
        self._mark_persisted()

    def _check_resume_config(self) -> None:
        st = self.store.status()
        problems = []
        if st["symbols"] != self.symbols:
            problems.append(f"symbols: run has {st['symbols']}, you passed {self.symbols}")
        if st["tf"] != self.tf:
            problems.append(f"tf: run has {st['tf']!r}, you passed {self.tf!r}")
        if st["params"] != asdict(self.params):
            problems.append(f"params: run has {st['params']}, you passed {asdict(self.params)}")
        if problems:
            raise ValueError(
                "refusing to resume with a different configuration ("
                + "; ".join(problems) + "). Use a new --db to start a new run."
            )

    def _mark_persisted(self) -> None:
        ex = self.engine.executor
        self._n_fills = len(ex.fills())
        self._n_trades = len(ex.trades())
        self._n_rejections = len(self.engine.rejections)
        self._n_events = len(self.engine.risk.events)

    def _adopt_external_risk_state(self) -> None:
        """An operator may change the risk state in the store (risk reset)
        while we run. Adopt it rather than overwrite it on the next persist."""
        current = _canon(self.store.load_state(STATE_RISK))
        if current is None or current == self._risk_saved:
            return
        self.engine.risk = RiskManager.from_state(current)
        self._risk_saved = current
        self._n_events = len(self.engine.risk.events)
        self.log(f"adopted risk state from store: {self.engine.risk.state.value}")

    # -- one round ------------------------------------------------------
    def _since(self) -> datetime:
        if self.last_ts is not None:
            return self.last_ts - self.warmup * self.tf_delta
        return self.now_fn() - 2 * self.warmup * self.tf_delta

    def _data_problem(self, detail: str) -> None:
        self.log(f"data problem: {detail}")
        if not self._outage:
            self.store.append_risk_events([{"ts": self.now_fn(), "kind": "data", "detail": detail}])
            self._outage = True

    def _data_recovered(self) -> None:
        if self._outage:
            self.store.append_risk_events(
                [{"ts": self.now_fn(), "kind": "data_ok", "detail": "data feed recovered"}]
            )
            self._outage = False

    def run_once(self) -> bool:
        """Process every closed bar not yet processed, in order. Returns True
        if at least one bar was stepped. Any unexpected error halts, persists
        the halt and re-raises."""
        try:
            return self._run_once()
        except DataFeedError as e:
            self._data_problem(str(e))
            return False
        except Exception as e:
            self._halt_on_error(self.now_fn(), e)
            raise

    def _run_once(self) -> bool:
        self._adopt_external_risk_state()
        since = self._since()
        frames: dict[str, pd.DataFrame] = {}
        for sym in self.symbols:
            self.feed.sync(sym, self.tf, since)
        for sym in self.symbols:
            frames[sym] = self.feed.get_candles(sym, self.tf, start=since)
        shared = None
        for df in frames.values():
            idx = pd.Index(df.ts)
            shared = idx if shared is None else shared.intersection(idx)
        assert shared is not None
        if self.last_ts is not None:
            shared = shared[shared > pd.Timestamp(self.last_ts)]
        pending_ts = sorted(shared)
        if self.last_ts is None:
            pending_ts = pending_ts[-1:]  # a fresh run starts at the newest bar; nothing to catch up
        if not pending_ts:
            return False
        if len(pending_ts) > 1:
            self.log(f"catching up {len(pending_ts)} bars")
        stepped = False
        for ts in pending_ts:
            windows = {}
            for sym in self.symbols:
                df = frames[sym]
                windows[sym] = df[df.ts <= ts].tail(self.warmup).reset_index(drop=True)
            short = [s for s, df in windows.items() if len(df) < self.warmup]
            if short:
                self._data_problem(f"insufficient history for {short}: need {self.warmup} bars")
                return stepped
            self._data_recovered()
            row = self.engine.step(windows)
            self.last_ts = ts.to_pydatetime()
            self.persist(self.last_ts, row)
            self.log(f"stepped {self.last_ts.isoformat()} equity={row['equity']:.2f}")
            stepped = True
        return stepped

    def _halt_on_error(self, ts: datetime, err: Exception) -> None:
        """Halt from the last *persisted* risk snapshot. The executor is not
        saved: the in-memory one may be mid-step, and the last persisted state
        replays the bar cleanly after a reset."""
        base = self.store.load_state(STATE_RISK)
        risk = RiskManager.from_state(base) if base else RiskManager()
        risk.state = RiskState.HALTED
        event = {"ts": ts, "kind": "error", "detail": f"{err!r}\n{traceback.format_exc()}"}
        risk.events.append(event)
        try:
            with self.store.transaction():
                self.store.append_risk_events([event])
                self.store.save_state(STATE_RISK, risk.to_state())
        except Exception as store_err:  # noqa: BLE001 - best effort, the original error wins
            self.log(f"could not persist halt: {store_err!r}")
        self.engine.risk = risk
        self._risk_saved = _canon(risk.to_state())
        self._n_events = len(risk.events)
        self.log(f"HALTED after error: {err!r}")

    def persist(self, ts: datetime, row: dict) -> None:
        ex = self.engine.executor
        risk = self.engine.risk
        with self.store.transaction():
            self.store.append_equity(ts, row["equity"], row["cash"], row["n_positions"],
                                     dict(self.engine.last_regimes))
            self.store.append_fills(ex.fills()[self._n_fills :])
            self.store.append_trades(ex.trades()[self._n_trades :])
            new_rej = self.engine.rejections[self._n_rejections :]
            if new_rej:
                self.store.append_rejections(ts, [(o, r) for _, o, r in new_rej])
            for sym in self.symbols:
                self.store.append_signals(ts, sym, self.engine.last_regimes.get(sym, "gap"),
                                          self.engine.last_signals.get(sym, {}))
            self.store.append_risk_events(risk.events[self._n_events :])
            self.store.replace_positions(ex.positions())
            self.store.save_state(STATE_EXECUTOR, ex.to_state())
            self.store.save_state(STATE_RISK, risk.to_state())
            self.store.save_state(STATE_LAST_TS, ts.isoformat())
        self._risk_saved = _canon(risk.to_state())
        self._mark_persisted()

    # -- scheduling -----------------------------------------------------
    def next_wake(self, now: datetime) -> datetime:
        tf_s = self.tf_delta.total_seconds()
        epoch = now.timestamp()
        next_close = (epoch // tf_s + 1) * tf_s
        return datetime.fromtimestamp(next_close, tz=UTC) + timedelta(seconds=POLL_DELAY_S)

    def _last_closed_ts(self, now: datetime) -> datetime:
        tf_s = self.tf_delta.total_seconds()
        return datetime.fromtimestamp((now.timestamp() // tf_s) * tf_s, tz=UTC) - self.tf_delta

    def run_forever(
        self, sleep_fn: Callable[[float], None] = time.sleep, max_iterations: int | None = None
    ) -> None:
        i = 0
        while max_iterations is None or i < max_iterations:
            i += 1
            self.run_once()
            now = self.now_fn()
            behind = self.last_ts is None or self.last_ts < self._last_closed_ts(now)
            wake = now + timedelta(seconds=RETRY_S) if behind else self.next_wake(now)
            sleep_fn(max(1.0, (wake - now).total_seconds()))
