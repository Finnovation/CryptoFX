"""Paper trading loop: one Engine.step per closed candle, everything persisted.

The trader is the only writer of the SQLite store. Restarting it resumes from
the saved executor and risk state, so a halt survives a reboot."""

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

    # -- lifecycle ------------------------------------------------------
    def bootstrap(self) -> None:
        ex_state = self.store.load_state(STATE_EXECUTOR)
        risk_state = self.store.load_state(STATE_RISK)
        last = self.store.load_state(STATE_LAST_TS)
        if ex_state is not None and risk_state is not None:
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
        self.engine = build_engine(self.params, executor, self.symbols, self.tf, self.warmup,
                                   risk=risk)
        self._mark_persisted()

    def _mark_persisted(self) -> None:
        ex = self.engine.executor
        self._n_fills = len(ex.fills())
        self._n_trades = len(ex.trades())
        self._n_rejections = len(self.engine.rejections)
        self._n_events = len(self.engine.risk.events)

    # -- one bar --------------------------------------------------------
    def _since(self) -> datetime:
        if self.last_ts is not None:
            return self.last_ts - self.warmup * self.tf_delta
        return self.now_fn() - 2 * self.warmup * self.tf_delta

    def _data_event(self, detail: str) -> None:
        self.log(f"data problem: {detail}")
        self.store.append_risk_events([{"ts": self.now_fn(), "kind": "data", "detail": detail}])

    def run_once(self) -> bool:
        since = self._since()
        windows: dict[str, pd.DataFrame] = {}
        try:
            for sym in self.symbols:
                self.feed.sync(sym, self.tf, since)
            for sym in self.symbols:
                windows[sym] = self.feed.get_candles(sym, self.tf, start=since)
        except DataFeedError as e:
            self._data_event(str(e))
            return False
        newest = min(df.ts.iloc[-1] for df in windows.values())
        if self.last_ts is not None and newest <= pd.Timestamp(self.last_ts):
            return False
        for sym in self.symbols:
            df = windows[sym]
            windows[sym] = df[df.ts <= newest].tail(self.warmup).reset_index(drop=True)
        short = [s for s, df in windows.items() if len(df) < self.warmup]
        if short:
            self._data_event(f"insufficient history for {short}: need {self.warmup} bars")
            return False
        try:
            row = self.engine.step(windows)
        except Exception as e:
            self._halt_on_error(newest.to_pydatetime(), e)
            raise
        self.last_ts = newest.to_pydatetime()
        self.persist(self.last_ts, row)
        self.log(f"stepped {self.last_ts.isoformat()} equity={row['equity']:.2f}")
        return True

    def _halt_on_error(self, ts: datetime, err: Exception) -> None:
        risk = self.engine.risk
        risk.state = RiskState.HALTED
        risk.events.append({"ts": ts, "kind": "error", "detail": f"{err!r}\n{traceback.format_exc()}"})
        with self.store.transaction():
            self.store.append_risk_events(risk.events[self._n_events :])
            self.store.save_state(STATE_RISK, risk.to_state())
            self.store.save_state(STATE_EXECUTOR, self.engine.executor.to_state())
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
