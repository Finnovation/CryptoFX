"""Executor protocol shared by backtest, paper and live backends.

``submit`` may fill immediately (live market order) or queue the order for the
next bar (backtest, paper). Either way every fill lands in ``fills()`` and every
failure in ``rejections()``, so the Engine treats all backends the same."""

from typing import Protocol

import pandas as pd

from cryptofx.core.models import Fill, Order, Position, Queued, Rejected, Trade


class Executor(Protocol):
    def submit(self, order: Order) -> Fill | Queued | Rejected: ...

    def positions(self) -> list[Position]: ...

    def pending(self) -> list[Order]: ...

    def fills(self) -> list[Fill]: ...

    def rejections(self) -> list[tuple[Order, str]]: ...

    def equity(self) -> float: ...

    def cash(self) -> float: ...

    def trades(self) -> list[Trade]: ...

    def on_bar(self, symbol: str, bar: pd.Series) -> list[Trade]:
        """Advance ``symbol`` to this bar: fill queued orders at its open, then
        process stops. Returns trades closed on this bar. A bar flagged ``gap``
        (forward-filled, never traded) updates the mark only."""
        ...

    def set_stop(self, symbol: str, stop_price: float) -> None: ...
