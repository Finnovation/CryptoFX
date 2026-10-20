"""Bar-by-bar engine: candles -> regime -> agents -> allocator -> risk -> executor."""

import pandas as pd

from cryptofx.agents.base import Agent
from cryptofx.agents.indicators import atr
from cryptofx.agents.regime import RegimeAgent
from cryptofx.core.allocator import Allocator
from cryptofx.core.models import Fill, Order, Rejected, Side, Trade
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
        self.fills: list[Fill] = []
        self.rejections: list[tuple[pd.Timestamp, Order, str]] = []
        self._pending: list[Order] = []

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
            ts = data[self.symbols[0]].ts.iloc[i]
            # 1. reference bar + stops
            for sym in self.symbols:
                self.executor.on_bar(sym, data[sym].iloc[i])
            # 2. pending orders from previous bar
            for order in self._pending:
                result = self.executor.submit(order)
                if isinstance(result, Rejected):
                    self.rejections.append((ts, order, result.reason))
                else:
                    self.fills.append(result)
            self._pending = []
            # 3. risk state
            self.risk.update_equity(ts.to_pydatetime(), self.executor.equity())
            # 4. decisions for next bar
            held = {p.symbol: p for p in self.executor.positions()}
            regimes = {}
            for sym in self.symbols:
                window = data[sym].iloc[max(0, i + 1 - self.warmup) : i + 1]
                regimes[sym] = self._decide(sym, window, held, ts)
            # 5. equity curve
            row = {
                "ts": ts,
                "equity": self.executor.equity(),
                "cash": self.executor.cash(),
                "n_positions": len(self.executor.positions()),
            }
            for sym in self.symbols:
                row[f"regime_{sym}"] = regimes[sym]
            rows.append(row)
        return pd.DataFrame(rows).set_index("ts")

    def _decide(self, sym: str, window: pd.DataFrame, held: dict, ts: pd.Timestamp) -> str:
        if DataFeed.max_gap_bars(window) > MAX_GAP_BARS:
            return "gap"
        regime = self.regime_agent.classify(window)
        signals = {a.name: a.signal(window, regime) for a in self.agents}
        pos = held.get(sym)
        if pos is not None:
            a = atr(window.high, window.low, window.close, self.allocator.atr_n).iloc[-1]
            if not pd.isna(a):
                new_stop = self.risk.trail_stop(pos, float(window.close.iloc[-1]), float(a))
                if new_stop is not None:
                    self.executor.set_stop(sym, new_stop)
            if self.allocator.score(signals, regime) < self.allocator.threshold:
                self._pending.append(
                    Order(symbol=sym, side=Side.SELL, size_quote=0.0, stop_price=None,
                          reason="signal exit")
                )
            return regime.value
        target = self.allocator.allocate(
            sym, signals, regime, window, self.executor.equity(), self.risk.risk_fraction()
        )
        if target is not None:
            order = Order(
                symbol=sym, side=Side.BUY, size_quote=target.size_quote,
                stop_price=target.stop_price,
                reason=";".join(f"{k}:{v.reason}" for k, v in signals.items()),
            )
            view = PortfolioView(equity=self.executor.equity(), positions=list(held.values()))
            verdict = self.risk.check(order, view)
            if isinstance(verdict, RiskRejected):
                self.rejections.append((ts, order, verdict.reason))
            else:
                self._pending.append(verdict.order)
        return regime.value
