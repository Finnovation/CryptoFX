from datetime import UTC, datetime, timedelta

import pytest

from cryptofx.core.models import Order, Position, Rejected, Side
from cryptofx.risk.limits import MAX_OPEN_POSITIONS, MAX_SYMBOL_FRACTION, MIN_ORDER_QUOTE
from cryptofx.risk.manager import Approved, PortfolioView, Resized, RiskManager, RiskState

T0 = datetime(2024, 1, 1, 12, tzinfo=UTC)


def pos(symbol="BTC/USDT", entry=100.0, stop=90.0):
    return Position(
        symbol=symbol, size_base=1.0, entry_price=entry, stop_price=stop, opened_at=T0,
        initial_risk=entry - stop,
    )


def buy(symbol="SOL/USDT", size=50.0, stop=90.0):
    return Order(symbol=symbol, side=Side.BUY, size_quote=size, stop_price=stop, reason="t")


def view(equity=1000.0, positions=None):
    return PortfolioView(equity=equity, positions=positions or [])


def test_sell_always_approved():
    rm = RiskManager()
    rm.state = RiskState.HALTED
    sell = Order(symbol="BTC/USDT", side=Side.SELL, size_quote=5.0, stop_price=None, reason="exit")
    assert isinstance(rm.check(sell, view()), Approved)


def test_rejects_when_max_positions():
    held = [pos(f"X{i}/USDT") for i in range(MAX_OPEN_POSITIONS)]
    res = RiskManager().check(buy(), view(positions=held))
    assert isinstance(res, Rejected) and res.reason == "max positions"


def test_allows_add_to_existing_symbol_at_max():
    held = [pos(f"X{i}/USDT") for i in range(MAX_OPEN_POSITIONS)]
    assert isinstance(RiskManager().check(buy(symbol="X0/USDT"), view(positions=held)), Approved)


def test_resizes_over_symbol_fraction():
    res = RiskManager().check(buy(size=500.0), view(equity=1000.0))
    assert isinstance(res, Resized)
    assert res.order.size_quote == pytest.approx(MAX_SYMBOL_FRACTION * 1000.0)


def test_rejects_below_min_order():
    res = RiskManager().check(buy(size=MIN_ORDER_QUOTE - 0.01), view())
    assert isinstance(res, Rejected) and res.reason == "below min order"


def test_rejects_missing_stop():
    res = RiskManager().check(buy(stop=None), view())
    assert isinstance(res, Rejected) and res.reason == "missing stop"


def test_rejects_buy_when_not_active():
    rm = RiskManager()
    rm.state = RiskState.PAUSED
    assert isinstance(rm.check(buy(), view()), Rejected)


def test_daily_loss_pauses_24h_then_resumes():
    rm = RiskManager()
    rm.update_equity(T0, 1000.0)
    rm.update_equity(T0 + timedelta(hours=1), 969.0)  # -3.1%
    assert rm.state == RiskState.PAUSED
    assert rm.paused_until == T0 + timedelta(hours=25)
    rm.update_equity(T0 + timedelta(hours=20), 969.0)
    assert rm.state == RiskState.PAUSED
    rm.update_equity(T0 + timedelta(hours=25), 969.0)
    assert rm.state == RiskState.ACTIVE
    assert [e["kind"] for e in rm.events] == ["paused", "resumed"]


def test_daily_loss_resets_at_utc_midnight():
    rm = RiskManager()
    rm.update_equity(T0, 1000.0)
    rm.update_equity(T0 + timedelta(hours=10), 980.0)  # -2%, fine
    next_day = datetime(2024, 1, 2, 0, tzinfo=UTC)
    rm.update_equity(next_day, 980.0)  # new day starts at 980
    rm.update_equity(next_day + timedelta(hours=1), 960.0)  # -2.04% from 980, fine
    assert rm.state == RiskState.ACTIVE


def test_drawdown_halts_and_is_sticky():
    rm = RiskManager()
    rm.update_equity(T0, 1000.0)
    rm.update_equity(T0 + timedelta(days=5), 899.0)
    assert rm.state == RiskState.HALTED
    rm.update_equity(T0 + timedelta(days=10), 1200.0)
    assert rm.state == RiskState.HALTED
    assert rm.events[-1]["kind"] == "halted"


def test_reset_requires_reason_and_clears_halt():
    rm = RiskManager()
    rm.update_equity(T0, 1000.0)
    rm.update_equity(T0 + timedelta(days=5), 850.0)
    with pytest.raises(ValueError):
        rm.reset("", T0 + timedelta(days=6))
    rm.reset("reviewed drawdown, resuming", T0 + timedelta(days=6))
    assert rm.state == RiskState.ACTIVE
    assert rm.peak_equity is None
    assert rm.events[-1]["kind"] == "reset"


def test_trail_stop_activates_after_1R():
    rm = RiskManager()
    p = pos(entry=100.0, stop=90.0)
    assert rm.trail_stop(p, close=105.0, atr_value=2.0) is None
    assert rm.trail_stop(p, close=110.0, atr_value=2.0) == pytest.approx(107.0)


def test_trail_stop_never_lowers():
    rm = RiskManager()
    p = pos(entry=100.0, stop=108.0)
    assert rm.trail_stop(p, close=110.0, atr_value=2.0) is None


def test_risk_fraction():
    assert RiskManager().risk_fraction() == 0.01


def test_counts_pending_buys_toward_max_positions():
    held = [pos("X0/USDT"), pos("X1/USDT")]
    pending = [buy(symbol="X2/USDT")]
    res = RiskManager().check(buy(symbol="X3/USDT"), PortfolioView(equity=1000.0, positions=held, pending_buys=pending))
    assert isinstance(res, Rejected) and res.reason == "max positions"
    # a pending buy for an already-held symbol does not consume a slot
    res2 = RiskManager().check(
        buy(symbol="X3/USDT"),
        PortfolioView(equity=1000.0, positions=held, pending_buys=[buy(symbol="X0/USDT")]),
    )
    assert isinstance(res2, Approved)


def test_rejects_when_committed_exceeds_cash():
    pending = [buy(symbol="X1/USDT", size=200.0), buy(symbol="X2/USDT", size=200.0)]
    res = RiskManager().check(
        buy(symbol="X3/USDT", size=200.0),
        PortfolioView(equity=1000.0, positions=[], pending_buys=pending, cash=500.0),
    )
    assert isinstance(res, Rejected) and res.reason == "insufficient cash"
