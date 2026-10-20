"""Risk manager: every order passes through here, and it can pause or halt trading."""

from datetime import datetime, timedelta
from enum import Enum

from pydantic import BaseModel, ConfigDict

from cryptofx.core.models import Order, Position, Rejected, Side
from cryptofx.risk import limits


class RiskState(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    HALTED = "halted"


class Approved(BaseModel):
    model_config = ConfigDict(frozen=True)
    order: Order


class Resized(BaseModel):
    model_config = ConfigDict(frozen=True)
    order: Order
    reason: str


class PortfolioView(BaseModel):
    equity: float
    positions: list[Position]


class RiskManager:
    def __init__(self) -> None:
        self.state = RiskState.ACTIVE
        self.peak_equity: float | None = None
        self.day_start_equity: float | None = None
        self.day_start_ts: datetime | None = None
        self.paused_until: datetime | None = None
        self.events: list[dict] = []

    def _event(self, ts: datetime, kind: str, detail: str) -> None:
        self.events.append({"ts": ts, "kind": kind, "detail": detail})

    def update_equity(self, ts: datetime, equity: float) -> None:
        if self.peak_equity is None or equity > self.peak_equity:
            self.peak_equity = equity
        if self.day_start_ts is None or ts.date() != self.day_start_ts.date():
            self.day_start_ts = ts
            self.day_start_equity = equity

        if self.state == RiskState.HALTED:
            return
        if equity <= self.peak_equity * (1 - limits.MAX_DRAWDOWN):
            self.state = RiskState.HALTED
            self._event(ts, "halted", f"drawdown: equity {equity:.2f} vs peak {self.peak_equity:.2f}")
            return
        if self.state == RiskState.PAUSED:
            if self.paused_until is not None and ts >= self.paused_until:
                self.state = RiskState.ACTIVE
                self.paused_until = None
                self._event(ts, "resumed", "daily pause elapsed")
            return
        if equity <= self.day_start_equity * (1 - limits.DAILY_LOSS_LIMIT):
            self.state = RiskState.PAUSED
            self.paused_until = ts + timedelta(hours=limits.DAILY_PAUSE_HOURS)
            self._event(
                ts, "paused", f"daily loss: equity {equity:.2f} vs day start {self.day_start_equity:.2f}"
            )

    def check(self, order: Order, view: PortfolioView) -> Approved | Resized | Rejected:
        if order.side == Side.SELL:
            return Approved(order=order)
        if self.state != RiskState.ACTIVE:
            return Rejected(reason=f"risk state {self.state.value}")
        if order.stop_price is None:
            return Rejected(reason="missing stop")
        held = {p.symbol for p in view.positions}
        if len(view.positions) >= limits.MAX_OPEN_POSITIONS and order.symbol not in held:
            return Rejected(reason="max positions")
        cap = limits.MAX_SYMBOL_FRACTION * view.equity
        size = min(order.size_quote, cap)
        if size < limits.MIN_ORDER_QUOTE:
            return Rejected(reason="below min order")
        if size < order.size_quote:
            resized = order.model_copy(update={"size_quote": size})
            return Resized(order=resized, reason=f"capped to {limits.MAX_SYMBOL_FRACTION:.0%} of equity")
        return Approved(order=order)

    def trail_stop(self, pos: Position, close: float, atr_value: float) -> float | None:
        if close - pos.entry_price < limits.TRAIL_ACTIVATE_R * pos.initial_risk:
            return None
        new_stop = max(pos.stop_price, close - limits.TRAIL_ATR_MULT * atr_value)
        return new_stop if new_stop > pos.stop_price else None

    def reset(self, reason: str, ts: datetime) -> None:
        if not reason.strip():
            raise ValueError("reset requires a written reason")
        self.state = RiskState.ACTIVE
        self.peak_equity = None
        self.paused_until = None
        self._event(ts, "reset", reason)

    def risk_fraction(self) -> float:
        return limits.RISK_PER_TRADE
