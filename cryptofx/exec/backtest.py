"""Deterministic simulated executor with fees, slippage and stop handling.

Orders queue on ``submit`` and fill at the next bar's open, so the Engine does
not have to know about fill latency: a paper or live executor can plug in and
fill however its venue does."""

import math

import pandas as pd

from cryptofx.core.models import Fill, Order, Position, Queued, Rejected, Side, Trade
from cryptofx.risk.limits import FEE_RATE, MAX_IMPACT, SLIPPAGE_RATE


class BacktestExecutor:
    def __init__(
        self,
        initial_equity: float,
        fee_rate: float = FEE_RATE,
        slippage_rate: float = SLIPPAGE_RATE,
    ) -> None:
        self._cash = float(initial_equity)
        self.fee_rate = fee_rate
        self.slippage_rate = slippage_rate
        self._bars: dict[str, pd.Series] = {}
        self._positions: dict[str, Position] = {}
        self._entry_fees: dict[str, float] = {}
        self._pending: list[Order] = []
        self._fills: list[Fill] = []
        self._rejections: list[tuple[Order, str]] = []
        self._trades: list[Trade] = []

    # -- bar advance ----------------------------------------------------
    def on_bar(self, symbol: str, bar: pd.Series) -> list[Trade]:
        self._bars[symbol] = bar
        if bool(bar.get("gap", False)):
            return []  # forward-filled bar: no price ever traded here
        mine = [o for o in self._pending if o.symbol == symbol]
        self._pending = [o for o in self._pending if o.symbol != symbol]
        for order in mine:
            result = self._buy(order, bar) if order.side == Side.BUY else self._sell(order, bar)
            if isinstance(result, Rejected):
                self._rejections.append((order, result.reason))
            else:
                self._fills.append(result)
        pos = self._positions.get(symbol)
        if pos is None or float(bar["low"]) > pos.stop_price:
            return []
        exit_price = min(float(bar["open"]), pos.stop_price) * (1 - self.slippage_rate)
        return [self._close(symbol, exit_price, bar, "stop")]

    def set_stop(self, symbol: str, stop_price: float) -> None:
        pos = self._positions.get(symbol)
        if pos is not None:
            pos.stop_price = stop_price

    # -- orders ---------------------------------------------------------
    def submit(self, order: Order) -> Fill | Queued | Rejected:
        if order.side == Side.BUY and order.stop_price is None:
            return Rejected(reason="missing stop")
        self._pending.append(order)
        return Queued(order=order)

    def _impact(self, bar: pd.Series, size_quote: float) -> float:
        volume = float(bar["volume"])
        open_ = float(bar["open"])
        if math.isnan(volume) or volume <= 0 or open_ <= 0:
            return MAX_IMPACT
        return min(MAX_IMPACT, 0.1 * size_quote / (volume * open_))

    def _buy(self, order: Order, bar: pd.Series) -> Fill | Rejected:
        assert order.stop_price is not None
        price = float(bar["open"]) * (1 + self.slippage_rate + self._impact(bar, order.size_quote))
        fee = order.size_quote * self.fee_rate
        if self._cash < order.size_quote + fee:
            return Rejected(reason="insufficient cash")
        self._cash -= order.size_quote + fee
        size_base = order.size_quote / price
        pos = self._positions.get(order.symbol)
        if pos is None:
            self._positions[order.symbol] = Position(
                symbol=order.symbol,
                size_base=size_base,
                entry_price=price,
                stop_price=order.stop_price,
                opened_at=bar["ts"].to_pydatetime(),
                initial_risk=price - order.stop_price,
            )
            self._entry_fees[order.symbol] = fee
        else:
            total = pos.size_base + size_base
            pos.entry_price = (pos.entry_price * pos.size_base + price * size_base) / total
            pos.size_base = total
            pos.stop_price = order.stop_price
            pos.initial_risk = pos.entry_price - order.stop_price
            self._entry_fees[order.symbol] += fee
        return Fill(
            symbol=order.symbol, side=Side.BUY, size_base=size_base, price=price, fee=fee,
            ts=bar["ts"].to_pydatetime(),
        )

    def _sell(self, order: Order, bar: pd.Series) -> Fill | Rejected:
        pos = self._positions.get(order.symbol)
        if pos is None:
            return Rejected(reason="no position")
        notional = pos.size_base * float(bar["open"])
        price = float(bar["open"]) * (1 - self.slippage_rate - self._impact(bar, notional))
        exit_fee = pos.size_base * price * self.fee_rate
        trade = self._close(order.symbol, price, bar, order.reason)
        return Fill(
            symbol=order.symbol, side=Side.SELL, size_base=trade.size_base, price=price,
            fee=exit_fee, ts=bar["ts"].to_pydatetime(),
        )

    def _close(self, symbol: str, exit_price: float, bar: pd.Series, reason: str) -> Trade:
        pos = self._positions.pop(symbol)
        entry_fee = self._entry_fees.pop(symbol, 0.0)
        proceeds = pos.size_base * exit_price
        exit_fee = proceeds * self.fee_rate
        self._cash += proceeds - exit_fee
        fees = entry_fee + exit_fee
        trade = Trade(
            symbol=symbol,
            entry_ts=pos.opened_at,
            exit_ts=bar["ts"].to_pydatetime(),
            entry_price=pos.entry_price,
            exit_price=exit_price,
            size_base=pos.size_base,
            pnl=(exit_price - pos.entry_price) * pos.size_base - fees,
            fees=fees,
            exit_reason=reason,
        )
        self._trades.append(trade)
        return trade

    # -- state ----------------------------------------------------------
    def positions(self) -> list[Position]:
        return list(self._positions.values())

    def pending(self) -> list[Order]:
        return list(self._pending)

    def fills(self) -> list[Fill]:
        return list(self._fills)

    def rejections(self) -> list[tuple[Order, str]]:
        return list(self._rejections)

    def cash(self) -> float:
        return self._cash

    def trades(self) -> list[Trade]:
        return list(self._trades)

    def equity(self) -> float:
        value = self._cash
        for symbol, pos in self._positions.items():
            bar = self._bars.get(symbol)
            price = float(bar["close"]) if bar is not None else pos.entry_price
            value += pos.size_base * price
        return value
