from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from cryptofx.core.models import Fill, Order, Rejected, Side
from cryptofx.exec.backtest import BacktestExecutor

T0 = datetime(2024, 1, 1, tzinfo=UTC)
SYM = "BTC/USDT"


def bar(open, high, low, close, volume=1e6, ts=T0):
    return pd.Series(
        {"ts": pd.Timestamp(ts), "open": open, "high": high, "low": low, "close": close, "volume": volume}
    )


def buy(size=100.0, stop=90.0):
    return Order(symbol=SYM, side=Side.BUY, size_quote=size, stop_price=stop, reason="t")


def sell(reason="signal exit"):
    return Order(symbol=SYM, side=Side.SELL, size_quote=0.0, stop_price=None, reason=reason)


def test_buy_fill_includes_fee_and_slippage():
    ex = BacktestExecutor(1000.0)
    ex.on_bar(SYM, bar(100, 101, 99, 100))
    f = ex.submit(buy())
    assert isinstance(f, Fill)
    assert f.price == pytest.approx(100 * 1.0005, rel=1e-6)
    assert f.fee == pytest.approx(0.1)
    assert f.size_base == pytest.approx(100.0 / f.price)
    assert ex.cash() == pytest.approx(1000 - 100 - 0.1)
    assert len(ex.positions()) == 1
    assert ex.positions()[0].initial_risk == pytest.approx(f.price - 90.0)


def test_equity_marks_to_last_close():
    ex = BacktestExecutor(1000.0)
    ex.on_bar(SYM, bar(100, 101, 99, 100))
    f = ex.submit(buy())
    ex.on_bar(SYM, bar(100, 121, 99, 120))
    assert ex.equity() == pytest.approx(ex.cash() + f.size_base * 120.0)


def test_stop_gap_fills_at_open_not_stop():
    ex = BacktestExecutor(1000.0)
    ex.on_bar(SYM, bar(100, 101, 99, 100))
    f = ex.submit(buy(stop=90.0))
    closed = ex.on_bar(SYM, bar(80, 85, 78, 82, ts=datetime(2024, 1, 1, 1, tzinfo=UTC)))
    assert len(closed) == 1 and ex.positions() == []
    t = closed[0]
    assert t.exit_price == pytest.approx(80 * (1 - 0.0005))
    assert t.exit_reason == "stop"
    assert t.size_base == pytest.approx(f.size_base)


def test_stop_touched_within_bar_fills_at_stop():
    ex = BacktestExecutor(1000.0)
    ex.on_bar(SYM, bar(100, 101, 99, 100))
    ex.submit(buy(stop=90.0))
    closed = ex.on_bar(SYM, bar(95, 96, 89, 92))
    assert closed[0].exit_price == pytest.approx(90 * (1 - 0.0005))


def test_zero_volume_uses_max_impact():
    ex = BacktestExecutor(1000.0)
    ex.on_bar(SYM, bar(100, 101, 99, 100, volume=0.0))
    f = ex.submit(buy())
    assert f.price == pytest.approx(100 * (1 + 0.0005 + 0.01))
    ex2 = BacktestExecutor(1000.0)
    ex2.on_bar(SYM, bar(100, 101, 99, 100, volume=np.nan))
    assert ex2.submit(buy()).price == pytest.approx(100 * (1 + 0.0005 + 0.01))


def test_impact_scales_with_size_and_is_capped():
    ex = BacktestExecutor(100_000.0)
    ex.on_bar(SYM, bar(100, 101, 99, 100, volume=100.0))  # bar notional 10_000
    f = ex.submit(buy(size=1000.0))  # impact = 0.1*1000/10000 = 0.01 -> capped at 0.01
    assert f.price == pytest.approx(100 * (1 + 0.0005 + 0.01))


def test_sell_records_trade_pnl():
    ex = BacktestExecutor(1000.0)
    ex.on_bar(SYM, bar(100, 101, 99, 100))
    f = ex.submit(buy())
    ex.on_bar(SYM, bar(110, 111, 109, 110))
    s = ex.submit(sell())
    assert isinstance(s, Fill) and s.side == Side.SELL
    t = ex.trades()[0]
    assert t.entry_price == pytest.approx(f.price)
    assert t.exit_price == pytest.approx(110 * (1 - 0.0005), rel=1e-6)
    assert t.fees == pytest.approx(f.fee + s.fee)
    assert t.pnl == pytest.approx((t.exit_price - t.entry_price) * t.size_base - t.fees)
    assert t.exit_reason == "signal exit"
    assert ex.positions() == []
    assert ex.cash() == pytest.approx(1000 - 100 - f.fee + t.size_base * t.exit_price - s.fee)


def test_insufficient_cash_rejected():
    ex = BacktestExecutor(50.0)
    ex.on_bar(SYM, bar(100, 101, 99, 100))
    r = ex.submit(buy(size=100.0))
    assert isinstance(r, Rejected) and r.reason == "insufficient cash"
    assert ex.cash() == 50.0


def test_sell_without_position_rejected():
    ex = BacktestExecutor(1000.0)
    ex.on_bar(SYM, bar(100, 101, 99, 100))
    assert isinstance(ex.submit(sell()), Rejected)


def test_submit_without_bar_rejected():
    assert isinstance(BacktestExecutor(1000.0).submit(buy()), Rejected)


def test_set_stop_raises_stop():
    ex = BacktestExecutor(1000.0)
    ex.on_bar(SYM, bar(100, 101, 99, 100))
    ex.submit(buy(stop=90.0))
    ex.set_stop(SYM, 98.0)
    assert ex.positions()[0].stop_price == 98.0
    closed = ex.on_bar(SYM, bar(99, 100, 97, 99))
    assert len(closed) == 1 and closed[0].exit_reason == "stop"


def test_add_to_position_averages_entry():
    ex = BacktestExecutor(1000.0)
    ex.on_bar(SYM, bar(100, 101, 99, 100))
    f1 = ex.submit(buy(size=100.0, stop=90.0))
    ex.on_bar(SYM, bar(200, 201, 199, 200))
    f2 = ex.submit(buy(size=100.0, stop=180.0))
    p = ex.positions()[0]
    assert p.size_base == pytest.approx(f1.size_base + f2.size_base)
    expected_entry = (f1.size_base * f1.price + f2.size_base * f2.price) / p.size_base
    assert p.entry_price == pytest.approx(expected_entry)
    assert p.stop_price == 180.0
