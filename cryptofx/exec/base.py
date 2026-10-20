"""Executor protocol shared by backtest, paper and live backends."""

from typing import Protocol

import pandas as pd

from cryptofx.core.models import Fill, Order, Position, Rejected, Trade


class Executor(Protocol):
    def submit(self, order: Order) -> Fill | Rejected: ...

    def positions(self) -> list[Position]: ...

    def equity(self) -> float: ...

    def cash(self) -> float: ...

    def trades(self) -> list[Trade]: ...

    def on_bar(self, symbol: str, bar: pd.Series) -> list[Trade]:
        """Set the execution reference bar for ``symbol`` and process stops."""
        ...

    def set_stop(self, symbol: str, stop_price: float) -> None: ...
