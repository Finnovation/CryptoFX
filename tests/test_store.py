import math
from datetime import UTC, datetime, timedelta

import pytest

from cryptofx.core.models import Direction, Fill, Order, Position, Side, Signal, Trade
from cryptofx.store.db import Store

T0 = datetime(2024, 1, 1, tzinfo=UTC)


def ts(i):
    return T0 + timedelta(hours=i)


def trade(pnl, symbol="BTC/USDT", reason="signal exit", i=0):
    return Trade(
        symbol=symbol, entry_ts=ts(i), exit_ts=ts(i + 1), entry_price=100.0,
        exit_price=100.0 + pnl, size_base=1.0, pnl=pnl, fees=0.2, exit_reason=reason,
    )


@pytest.fixture
def store():
    return Store(":memory:")


def test_schema_applies_twice(tmp_path):
    p = tmp_path / "x.db"
    Store(p)
    Store(p)


def test_run_and_status_roundtrip(store):
    run_id = store.start_run("paper", ["BTC/USDT", "ETH/USDT"], "1h", {"threshold": 0.6}, 500.0)
    assert run_id == 1
    s = store.status()
    assert s["mode"] == "paper" and s["symbols"] == ["BTC/USDT", "ETH/USDT"]
    assert s["tf"] == "1h" and s["params"] == {"threshold": 0.6} and s["initial_equity"] == 500.0
    assert s["started_at"].tzinfo is not None


def test_empty_reads_are_well_formed(store):
    s = store.status()
    for k in ("mode", "started_at", "initial_equity", "symbols", "tf", "params", "last_ts", "equity", "cash"):
        assert k in s and s[k] is None
    assert s["n_positions"] == 0
    assert store.equity_curve() == [] and store.trades() == [] and store.positions() == []
    assert store.latest_signals() == [] and store.risk_events() == [] and store.fills() == []
    st = store.trade_stats()
    assert st["count"] == 0 and st["profit_factor"] == 0.0 and st["win_rate"] == 0.0
    assert store.first_equity_ts() is None and store.max_drawdown() == 0.0


def test_equity_append_is_idempotent_on_ts(store):
    store.append_equity(ts(0), 100.0, 100.0, 0, {"BTC/USDT": "ranging"})
    store.append_equity(ts(0), 101.0, 99.0, 1, {"BTC/USDT": "trending"})
    curve = store.equity_curve()
    assert len(curve) == 1 and curve[0]["equity"] == 101.0 and curve[0]["ts"] == ts(0)
    assert store.status()["last_ts"] == ts(0) and store.status()["equity"] == 101.0
    assert store.first_equity_ts() == ts(0)


def test_fills_trades_positions_roundtrip(store):
    store.append_fills([Fill(symbol="BTC/USDT", side=Side.BUY, size_base=0.5, price=100.0, fee=0.05, ts=ts(1))])
    store.append_trades([trade(10.0)])
    store.replace_positions([Position(symbol="BTC/USDT", size_base=0.5, entry_price=100.0, stop_price=90.0, opened_at=ts(1), initial_risk=10.0)])
    f = store.fills()[0]
    assert f["ts"] == ts(1) and f["side"] == "buy" and f["size_base"] == 0.5
    assert store.fills(symbol="ETH/USDT") == []
    assert store.fills(since=ts(2)) == [] and len(store.fills(since=ts(1))) == 1
    t = store.trades()[0]
    assert t["pnl"] == 10.0 and t["exit_ts"] == ts(1) and t["exit_reason"] == "signal exit"
    p = store.positions()[0]
    assert p["stop_price"] == 90.0 and p["opened_at"] == ts(1)
    store.replace_positions([])
    assert store.positions() == [] and store.status()["n_positions"] == 0


def test_trade_stats_hand_values(store):
    store.append_trades([trade(10.0, i=0), trade(20.0, symbol="ETH/USDT", i=2), trade(-15.0, reason="stop", i=4)])
    st = store.trade_stats()
    assert st["count"] == 3 and st["wins"] == 2 and st["losses"] == 1
    assert st["win_rate"] == pytest.approx(2 / 3)
    assert st["profit_factor"] == pytest.approx(2.0)
    assert st["expectancy"] == pytest.approx(5.0)
    assert st["avg_win"] == pytest.approx(15.0) and st["avg_loss"] == pytest.approx(-15.0)
    assert st["gross_pnl"] == pytest.approx(15.0) and st["fees"] == pytest.approx(0.6)
    assert st["by_symbol"]["BTC/USDT"] == {"count": 2, "pnl": pytest.approx(-5.0)}
    assert st["by_exit_reason"]["stop"] == {"count": 1, "pnl": pytest.approx(-15.0)}


def test_trade_stats_profit_factor_inf_without_losses(store):
    store.append_trades([trade(5.0)])
    assert math.isinf(store.trade_stats()["profit_factor"])


def test_latest_signals_per_symbol(store):
    sig = {"trend": Signal(direction=Direction.LONG, confidence=0.8, reason="up")}
    store.append_signals(ts(0), "BTC/USDT", "trending", sig)
    store.append_signals(ts(1), "BTC/USDT", "ranging", sig)
    store.append_signals(ts(0), "ETH/USDT", "chaos", sig)
    rows = store.latest_signals()
    assert {(r["symbol"], r["ts"]) for r in rows} == {("BTC/USDT", ts(1)), ("ETH/USDT", ts(0))}
    btc = next(r for r in rows if r["symbol"] == "BTC/USDT")
    assert btc["agent"] == "trend" and btc["direction"] == "long" and btc["regime"] == "ranging"


def test_risk_events_and_rejections(store):
    store.append_risk_events([{"ts": ts(0), "kind": "paused", "detail": "x"}, {"ts": ts(1), "kind": "halted", "detail": "y"}])
    ev = store.risk_events()
    assert [e["kind"] for e in ev] == ["halted", "paused"] and ev[0]["ts"] == ts(1)
    store.append_rejections(ts(2), [(Order(symbol="BTC/USDT", side=Side.BUY, size_quote=50.0, stop_price=90.0, reason="r"), "max positions")])
    assert store.rejections()[0]["reason"] == "max positions"


def test_state_roundtrip_and_missing(store):
    store.save_state("k", {"a": [1, 2], "when": ts(0)})
    assert store.load_state("k") == {"a": [1, 2], "when": ts(0).isoformat()}
    assert store.load_state("nope") is None
    store.save_state("k", 5)
    assert store.load_state("k") == 5


def test_max_drawdown_from_equity(store):
    for i, e in enumerate([100.0, 120.0, 90.0, 130.0]):
        store.append_equity(ts(i), e, e, 0, {})
    assert store.max_drawdown() == pytest.approx(0.25)


def test_transaction_rolls_back(store):
    with pytest.raises(RuntimeError), store.transaction():
        store.append_equity(ts(0), 1.0, 1.0, 0, {})
        raise RuntimeError("boom")
    assert store.equity_curve() == []
