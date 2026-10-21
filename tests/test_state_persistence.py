import json
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from cryptofx.core.models import Order, Side
from cryptofx.exec.backtest import BacktestExecutor
from cryptofx.exec.paper import PaperExecutor
from cryptofx.risk.manager import RiskManager, RiskState

T0 = datetime(2024, 1, 1, tzinfo=UTC)
SYM = "BTC/USDT"


def bar(open, high, low, close, i=0, volume=1e6):
    return pd.Series(
        {"ts": pd.Timestamp(T0 + timedelta(hours=i)), "open": open, "high": high, "low": low,
         "close": close, "volume": volume, "gap": False}
    )


def test_executor_state_roundtrip_reproduces_next_bar():
    a = BacktestExecutor(1000.0)
    a.on_bar(SYM, bar(100, 101, 99, 100, 0))
    a.submit(Order(symbol=SYM, side=Side.BUY, size_quote=100.0, stop_price=90.0, reason="t"))
    a.on_bar(SYM, bar(100, 101, 99, 100, 1))  # fills
    a.submit(Order(symbol="ETH/USDT", side=Side.BUY, size_quote=50.0, stop_price=9.0, reason="t"))
    b = BacktestExecutor.from_state(a.to_state())
    nxt = bar(110, 111, 109, 110, 2)
    eth = bar(10, 11, 9.5, 10, 2)
    for ex in (a, b):
        ex.on_bar(SYM, nxt)
        ex.on_bar("ETH/USDT", eth)
    assert a.cash() == pytest.approx(b.cash())
    assert a.equity() == pytest.approx(b.equity())
    assert [f.model_dump() for f in a.fills()] == [f.model_dump() for f in b.fills()]
    assert [p.model_dump() for p in a.positions()] == [p.model_dump() for p in b.positions()]
    assert a.pending() == b.pending() and a.trades() == b.trades()


def test_executor_state_is_json_serialisable():
    ex = BacktestExecutor(1000.0)
    ex.on_bar(SYM, bar(100, 101, 99, 100))
    ex.submit(Order(symbol=SYM, side=Side.BUY, size_quote=100.0, stop_price=90.0, reason="t"))
    ex.on_bar(SYM, bar(100, 101, 99, 100, 1))
    ex.submit(Order(symbol=SYM, side=Side.SELL, size_quote=0.0, stop_price=None, reason="x"))
    s = json.loads(json.dumps(ex.to_state()))
    assert s["cash"] == pytest.approx(ex.cash())
    assert BacktestExecutor.from_state(s).positions()[0].symbol == SYM


def test_risk_state_roundtrip_keeps_halt_and_pause():
    rm = RiskManager()
    rm.update_equity(T0, 1000.0)
    rm.update_equity(T0 + timedelta(hours=1), 969.0)  # paused
    s = json.loads(json.dumps(rm.to_state()))
    back = RiskManager.from_state(s)
    assert back.state == RiskState.PAUSED and back.paused_until == rm.paused_until
    assert back.peak_equity == 1000.0 and back.day_start_ts == rm.day_start_ts
    assert back.events == rm.events
    rm.update_equity(T0 + timedelta(days=3), 850.0)
    halted = RiskManager.from_state(rm.to_state())
    assert halted.state == RiskState.HALTED
    halted.update_equity(T0 + timedelta(days=4), 2000.0)
    assert halted.state == RiskState.HALTED


def test_paper_executor_mode():
    assert PaperExecutor(100.0).mode == "paper"
    assert BacktestExecutor(100.0).mode == "backtest"
    assert isinstance(PaperExecutor.from_state(PaperExecutor(100.0).to_state()), PaperExecutor)
