from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from cryptofx.core.models import Fill, Order, Queued, Rejected, Side
from cryptofx.exec.backtest import BacktestExecutor

T0 = datetime(2024, 1, 1, tzinfo=UTC)
SYM = "BTC/USDT"


def bar(open, high, low, close, volume=1e6, ts=T0, gap=False):
    return pd.Series(
        {"ts": pd.Timestamp(ts), "open": open, "high": high, "low": low, "close": close,
         "volume": volume, "gap": gap}
    )


def nxt(i):
    return T0 + timedelta(hours=i)


def buy(size=100.0, stop=90.0):
    return Order(symbol=SYM, side=Side.BUY, size_quote=size, stop_price=stop, reason="t")


def sell(reason="signal exit"):
    return Order(symbol=SYM, side=Side.SELL, size_quote=0.0, stop_price=None, reason="exit" if False else reason)


def opened(ex, size=100.0, stop=90.0, open_=100.0):
    """Queue a buy on bar 0 and fill it at bar 1 open."""
    ex.on_bar(SYM, bar(open_, open_ + 1, open_ - 1, open_))
    assert isinstance(ex.submit(buy(size, stop)), Queued)
    ex.on_bar(SYM, bar(open_, open_ + 1, open_ - 1, open_, ts=nxt(1)))
    return ex.fills()[-1]


def test_submit_queues_and_fills_on_next_bar_open():
    ex = BacktestExecutor(1000.0)
    ex.on_bar(SYM, bar(100, 101, 99, 100))
    q = ex.submit(buy())
    assert isinstance(q, Queued) and ex.positions() == [] and ex.fills() == []
    assert len(ex.pending()) == 1
    ex.on_bar(SYM, bar(120, 121, 119, 120, ts=nxt(1)))
    f = ex.fills()[0]
    assert isinstance(f, Fill) and f.price == pytest.approx(120 * 1.0005, rel=1e-6)
    assert pd.Timestamp(f.ts) == pd.Timestamp(nxt(1))
    assert ex.pending() == [] and len(ex.positions()) == 1


def test_buy_fill_includes_fee_and_slippage():
    ex = BacktestExecutor(1000.0)
    f = opened(ex)
    assert f.price == pytest.approx(100 * 1.0005, rel=1e-6)
    assert f.fee == pytest.approx(0.1)
    assert f.size_base == pytest.approx(100.0 / f.price)
    assert ex.cash() == pytest.approx(1000 - 100 - 0.1)
    assert ex.positions()[0].initial_risk == pytest.approx(f.price - 90.0)


def test_equity_marks_to_last_close():
    ex = BacktestExecutor(1000.0)
    f = opened(ex)
    ex.on_bar(SYM, bar(100, 121, 99, 120, ts=nxt(2)))
    assert ex.equity() == pytest.approx(ex.cash() + f.size_base * 120.0)


def test_stop_gap_fills_at_open_not_stop():
    ex = BacktestExecutor(1000.0)
    f = opened(ex, stop=90.0)
    closed = ex.on_bar(SYM, bar(80, 85, 78, 82, ts=nxt(2)))
    assert len(closed) == 1 and ex.positions() == []
    t = closed[0]
    assert t.exit_price == pytest.approx(80 * (1 - 0.0005))
    assert t.exit_reason == "stop"
    assert t.size_base == pytest.approx(f.size_base)


def test_stop_touched_within_bar_fills_at_stop():
    ex = BacktestExecutor(1000.0)
    opened(ex, stop=90.0)
    closed = ex.on_bar(SYM, bar(95, 96, 89, 92, ts=nxt(2)))
    assert closed[0].exit_price == pytest.approx(90 * (1 - 0.0005))


def test_entry_bar_is_stop_checked():
    """A crash wick on the fill bar must trigger the stop on that same bar."""
    ex = BacktestExecutor(1000.0)
    ex.on_bar(SYM, bar(100, 101, 99, 100))
    ex.submit(buy(stop=90.0))
    closed = ex.on_bar(SYM, bar(100, 101, 50, 95, ts=nxt(1)))
    assert len(ex.fills()) == 1  # the buy filled at 100
    assert len(closed) == 1 and closed[0].exit_reason == "stop"
    assert closed[0].exit_price == pytest.approx(90 * (1 - 0.0005))
    assert ex.positions() == []


def test_gap_bar_defers_fill_until_real_bar():
    ex = BacktestExecutor(1000.0)
    ex.on_bar(SYM, bar(100, 101, 99, 100))
    ex.submit(buy())
    assert ex.on_bar(SYM, bar(100, 100, 100, 100, volume=0.0, ts=nxt(1), gap=True)) == []
    assert ex.fills() == [] and len(ex.pending()) == 1
    ex.on_bar(SYM, bar(110, 111, 109, 110, ts=nxt(2)))
    assert ex.fills()[0].price == pytest.approx(110 * 1.0005, rel=1e-6)


def test_gap_bar_skips_stop_check():
    ex = BacktestExecutor(1000.0)
    opened(ex, stop=90.0)
    closed = ex.on_bar(SYM, bar(50, 50, 50, 50, volume=0.0, ts=nxt(2), gap=True))
    assert closed == [] and len(ex.positions()) == 1


def test_zero_volume_uses_max_impact():
    ex = BacktestExecutor(1000.0)
    ex.on_bar(SYM, bar(100, 101, 99, 100))
    ex.submit(buy())
    ex.on_bar(SYM, bar(100, 101, 99, 100, volume=0.0, ts=nxt(1)))
    assert ex.fills()[0].price == pytest.approx(100 * (1 + 0.0005 + 0.01))
    ex2 = BacktestExecutor(1000.0)
    ex2.on_bar(SYM, bar(100, 101, 99, 100))
    ex2.submit(buy())
    ex2.on_bar(SYM, bar(100, 101, 99, 100, volume=np.nan, ts=nxt(1)))
    assert ex2.fills()[0].price == pytest.approx(100 * (1 + 0.0005 + 0.01))


def test_impact_scales_with_size_and_is_capped():
    ex = BacktestExecutor(100_000.0)
    ex.on_bar(SYM, bar(100, 101, 99, 100))
    ex.submit(buy(size=1000.0))
    ex.on_bar(SYM, bar(100, 101, 99, 100, volume=100.0, ts=nxt(1)))  # notional 10_000
    assert ex.fills()[0].price == pytest.approx(100 * (1 + 0.0005 + 0.01))


def test_sell_records_trade_pnl():
    ex = BacktestExecutor(1000.0)
    f = opened(ex)
    assert isinstance(ex.submit(sell()), Queued)
    ex.on_bar(SYM, bar(110, 111, 109, 110, ts=nxt(2)))
    s = ex.fills()[-1]
    assert s.side == Side.SELL
    t = ex.trades()[0]
    assert t.entry_price == pytest.approx(f.price)
    assert t.exit_price == pytest.approx(110 * (1 - 0.0005), rel=1e-6)
    assert t.fees == pytest.approx(f.fee + s.fee)
    assert t.pnl == pytest.approx((t.exit_price - t.entry_price) * t.size_base - t.fees)
    assert t.exit_reason == "signal exit"
    assert ex.positions() == []
    assert ex.cash() == pytest.approx(1000 - 100 - f.fee + t.size_base * t.exit_price - s.fee)


def test_insufficient_cash_recorded_as_rejection():
    ex = BacktestExecutor(50.0)
    ex.on_bar(SYM, bar(100, 101, 99, 100))
    ex.submit(buy(size=100.0))
    ex.on_bar(SYM, bar(100, 101, 99, 100, ts=nxt(1)))
    assert ex.fills() == [] and ex.cash() == 50.0
    assert ex.rejections()[0][1] == "insufficient cash"


def test_sell_without_position_recorded_as_rejection():
    ex = BacktestExecutor(1000.0)
    ex.on_bar(SYM, bar(100, 101, 99, 100))
    ex.submit(sell())
    ex.on_bar(SYM, bar(100, 101, 99, 100, ts=nxt(1)))
    assert ex.rejections()[0][1] == "no position"


def test_buy_without_stop_rejected_at_submit():
    ex = BacktestExecutor(1000.0)
    r = ex.submit(Order(symbol=SYM, side=Side.BUY, size_quote=100.0, stop_price=None, reason="t"))
    assert isinstance(r, Rejected) and r.reason == "missing stop"


def test_set_stop_raises_stop():
    ex = BacktestExecutor(1000.0)
    opened(ex, stop=90.0)
    ex.set_stop(SYM, 98.0)
    assert ex.positions()[0].stop_price == 98.0
    closed = ex.on_bar(SYM, bar(99, 100, 97, 99, ts=nxt(2)))
    assert len(closed) == 1 and closed[0].exit_reason == "stop"


def test_add_to_position_averages_entry():
    ex = BacktestExecutor(1000.0)
    f1 = opened(ex, size=100.0, stop=90.0)
    ex.submit(buy(size=100.0, stop=180.0))
    ex.on_bar(SYM, bar(200, 201, 199, 200, ts=nxt(2)))
    f2 = ex.fills()[-1]
    p = ex.positions()[0]
    assert p.size_base == pytest.approx(f1.size_base + f2.size_base)
    expected_entry = (f1.size_base * f1.price + f2.size_base * f2.price) / p.size_base
    assert p.entry_price == pytest.approx(expected_entry)
    assert p.stop_price == 180.0
