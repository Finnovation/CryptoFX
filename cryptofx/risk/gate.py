"""Go-live gate: evaluated from the paper trade log, never bypassable."""

from dataclasses import dataclass
from datetime import timedelta

from cryptofx.store.db import Store

MIN_PAPER_DAYS = 42
MIN_TRADES = 50
MIN_PROFIT_FACTOR = 1.3
MAX_DRAWDOWN = 0.10
HALT_FREE_DAYS = 30


@dataclass(frozen=True)
class Criterion:
    name: str
    passed: bool
    actual: str
    required: str


class GoLiveGate:
    def __init__(self, store: Store) -> None:
        self.store = store

    def evaluate(self) -> list[Criterion]:
        st = self.store.status()
        stats = self.store.trade_stats()
        first = self.store.first_equity_ts()
        last = st["last_ts"]
        days = (last - first).total_seconds() / 86400.0 if first and last else 0.0
        dd = self.store.max_drawdown()
        recent_halt = False
        if last is not None:
            cutoff = last - timedelta(days=HALT_FREE_DAYS)
            recent_halt = any(
                e["kind"] == "halted" and e["ts"] >= cutoff for e in self.store.risk_events(limit=10_000)
            )
        pf = stats["profit_factor"]
        return [
            Criterion("paper_days", days >= MIN_PAPER_DAYS, f"{days:.1f}", f">= {MIN_PAPER_DAYS}"),
            Criterion("closed_trades", stats["count"] >= MIN_TRADES, str(stats["count"]),
                      f">= {MIN_TRADES}"),
            Criterion("profit_factor", stats["count"] > 0 and pf > MIN_PROFIT_FACTOR,
                      f"{pf:.2f}", f"> {MIN_PROFIT_FACTOR}"),
            Criterion("max_drawdown", last is not None and dd < MAX_DRAWDOWN, f"{dd:.1%}",
                      f"< {MAX_DRAWDOWN:.0%}"),
            Criterion("expectancy", stats["count"] > 0 and stats["expectancy"] > 0,
                      f"{stats['expectancy']:.4f}", "> 0"),
            Criterion("no_halt_30d", last is not None and not recent_halt,
                      "halt in window" if recent_halt else ("none" if last else "no data"),
                      f"none in {HALT_FREE_DAYS}d"),
        ]

    def passes(self) -> bool:
        return all(c.passed for c in self.evaluate())
