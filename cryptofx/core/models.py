"""Domain models shared by every layer of CryptoFX."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class Direction(str, Enum):
    LONG = "long"
    FLAT = "flat"


class Regime(str, Enum):
    TRENDING = "trending"
    RANGING = "ranging"
    CHAOS = "chaos"


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class Signal(_Frozen):
    direction: Direction
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


class TargetPosition(_Frozen):
    symbol: str
    size_quote: float
    stop_price: float


class Order(_Frozen):
    symbol: str
    side: Side
    size_quote: float
    stop_price: float | None
    reason: str


class Fill(_Frozen):
    symbol: str
    side: Side
    size_base: float
    price: float
    fee: float
    ts: datetime


class Rejected(_Frozen):
    reason: str


class Queued(_Frozen):
    """Order accepted by the executor and waiting for the next bar to fill."""

    order: Order


class Position(BaseModel):
    """Open position. Mutable because trailing stops move ``stop_price``."""

    symbol: str
    size_base: float
    entry_price: float
    stop_price: float
    opened_at: datetime
    initial_risk: float


class Trade(_Frozen):
    symbol: str
    entry_ts: datetime
    exit_ts: datetime
    entry_price: float
    exit_price: float
    size_base: float
    pnl: float
    fees: float
    exit_reason: str


FLAT_SIGNAL = Signal(direction=Direction.FLAT, confidence=0.0, reason="insufficient data")
