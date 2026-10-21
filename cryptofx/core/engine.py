"""Bar-by-bar engine: candles -> regime -> agents -> allocator -> risk -> executor.

``step`` is the unit of work for every mode. Backtest drives it from cached
candles via ``run``; a paper or live loop drives it once per closed candle."""

import pandas as pd

from cryptofx.agents.base import Agent
from cryptofx.agents.indicators import atr
from cryptofx.agents.regime import RegimeAgent
from cryptofx.core.allocator import Allocator
from cryptofx.core.models import Fill, Order, Side, Signal, Trade
from cryptofx.data.feed import DataFeed
from cryptofx.exec.base import Executor
from cryptofx.risk.limits import MAX_GAP_BARS
from cryptofx.risk.manager import PortfolioView, RiskManager
from cryptofx.risk.manager import Rejected as RiskRejected


class Engine:
    def __init__(
        self,
        agents: list[Agent],
        regime_agent: RegimeAgent,
        allocator: Allocator,
        risk: RiskManager,
        executor: Executor,
        symbols: list[str],
        tf: str,
        warmup: int = 200,
    ) -> None:
        self.agents = agents
        self.regime_agent = regime_agent
        self.allocator = allocator
        self.risk = risk
        self.executor = executor
        self.symbols = symbols
        self.tf = tf
        self.warmup = warmup
        self.rejections: list[tuple[pd.Timestamp, Order, str]] = []
        self._seen_exec_rejections = len(executor.rejections())
        self.last_signals: dict[str, dict[str, Signal]] = {}
        self.last_regimes: dict[str, str] = {}

    @property
    def fills(self) -> list[Fill]:
        return self.executor.fills()

    @property
    def trades(self) -> list[Trade]:
        return self.executor.trades()

    @property
    def risk_events(self) -> list[dict]:
        return self.risk.events

    # -- alignment ------------------------------------------------------
    def _align(self, candles: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        shared = None
        for sym in self.symbols:
            idx = pd.Index(candles[sym].ts)
            shared = idx if shared is None else shared.intersection(idx)
        if shared is None or len(shared) < self.warmup + 1:
            raise ValueError(
                f"need at least {self.warmup + 1} shared bars, got {0 if shared is None else len(shared)}"
            )
        shared = shared.sort_values()
        out = {}
        for sym in self.symbols:
            df = candles[sym]
            df = df[df.ts.isin(shared)].sort_values("ts").reset_index(drop=True)
            if "gap" not in df.columns:
                df = df.assign(gap=False)
            out[sym] = df
        return out

    # -- main loop ------------------------------------------------------
    def run(self, candles: dict[str, pd.DataFrame]) -> pd.DataFrame:
        data = self._align(candles)
        n = len(next(iter(data.values())))
        rows = []
        for i in range(self.warmup, n):
            windows = {
                sym: data[sym].iloc[max(0, i + 1 - self.warmup) : i + 1] for sym in self.symbols
            }
            rows.append(self.step(windows))
        return pd.DataFrame(rows).set_index("ts")

    def step(self, windows: dict[str, pd.DataFrame]) -> dict:
        """One bar for every symbol. Each window ends at the bar just closed."""
        ts = windows[self.symbols[0]].ts.iloc[-1]
        # 1. advance executor: queued orders fill at this bar's open, stops are checked
        for sym in self.symbols:
            self.executor.on_bar(sym, windows[sym].iloc[-1])
        new_rejections = self.executor.rejections()[self._seen_exec_rejections :]
        self.rejections.extend((ts, o, r) for o, r in new_rejections)
        self._seen_exec_rejections += len(new_rejections)
        # 2. risk state on marked equity
        self.risk.update_equity(ts.to_pydatetime(), self.executor.equity())
        # 3. decisions for the next bar
        held = {p.symbol: p for p in self.executor.positions()}
        self.last_signals = {}
        regimes = {sym: self._decide(sym, windows[sym], held, ts) for sym in self.symbols}
        self.last_regimes = regimes
        row = {
            "ts": ts,
            "equity": self.executor.equity(),
            "cash": self.executor.cash(),
            "n_positions": len(self.executor.positions()),
        }
        for sym in self.symbols:
            row[f"regime_{sym}"] = regimes[sym]
        return row

    def _decide(self, sym: str, window: pd.DataFrame, held: dict, ts: pd.Timestamp) -> str:
        if DataFeed.max_gap_bars(window) > MAX_GAP_BARS:
            self.last_signals[sym] = {}
            return "gap"
        regime = self.regime_agent.classify(window)
        signals = {a.name: a.signal(window, regime) for a in self.agents}
        self.last_signals[sym] = signals
        pos = held.get(sym)
        pending_for_sym = [o for o in self.executor.pending() if o.symbol == sym]
        if pos is not None:
            a = atr(window.high, window.low, window.close, self.allocator.atr_n).iloc[-1]
            if not pd.isna(a):
                new_stop = self.risk.trail_stop(pos, float(window.close.iloc[-1]), float(a))
                if new_stop is not None:
                    self.executor.set_stop(sym, new_stop)
            exit_pending = any(o.side == Side.SELL for o in pending_for_sym)
            if not exit_pending and self.allocator.score(signals, regime) < self.allocator.threshold:
                self.executor.submit(
                    Order(symbol=sym, side=Side.SELL, size_quote=0.0, stop_price=None,
                          reason="signal exit")
                )
            return regime.value
        if pending_for_sym:
            return regime.value  # an entry is already queued for this symbol
        target = self.allocator.allocate(
            sym, signals, regime, window, self.executor.equity(), self.risk.risk_fraction()
        )
        if target is not None:
            order = Order(
                symbol=sym, side=Side.BUY, size_quote=target.size_quote,
                stop_price=target.stop_price,
                reason=";".join(f"{k}:{v.reason}" for k, v in signals.items()),
            )
            view = PortfolioView(
                equity=self.executor.equity(),
                positions=list(held.values()),
                pending_buys=[o for o in self.executor.pending() if o.side == Side.BUY],
                cash=self.executor.cash(),
            )
            verdict = self.risk.check(order, view)
            if isinstance(verdict, RiskRejected):
                self.rejections.append((ts, order, verdict.reason))
            else:
                self.executor.submit(verdict.order)
        return regime.value
