from datetime import UTC, datetime, timedelta
from functools import partial

import pandas as pd
import pytest
from typer.testing import CliRunner

from cryptofx import cli
from cryptofx.backtest.runner import Params
from cryptofx.data.feed import DataFeed, DataFeedError
from cryptofx.paper.trader import POLL_DELAY_S, PaperTrader
from cryptofx.risk.manager import RiskState
from cryptofx.store.db import Store
from tests.synth import uptrend

SYMS = ["BTC/USDT", "ETH/USDT"]
H = timedelta(hours=1)


class Market:
    """Synthetic exchange: candles become visible as ``visible`` advances."""

    def __init__(self, n=400):
        self.df = uptrend(n=n, noise=1.5)
        self.visible = 150  # number of closed bars the exchange has published
        self.fail = False
        self.late = timedelta(0)  # how far the exchange is behind the clock

    def rows(self, since_ms):
        out = []
        for r in self.df.iloc[: self.visible].itertuples():
            ms = int(r.ts.timestamp() * 1000)
            if ms >= since_ms:
                out.append([ms, r.open, r.high, r.low, r.close, r.volume])
        return out

    def fetch(self, symbol, tf, since_ms, limit):
        if self.fail:
            raise RuntimeError("exchange down")
        return self.rows(since_ms)

    def now(self):
        return (self.df.ts.iloc[self.visible - 1] + H + timedelta(minutes=5) + self.late).to_pydatetime()


@pytest.fixture
def market():
    return Market()


def make_trader(tmp_path, market, store=None, warmup=100):
    feed = DataFeed(root=tmp_path, fetch_fn=market.fetch, backoff_s=0,
                    now_fn=lambda: int(market.now().timestamp() * 1000))
    store = store or Store(tmp_path / "t.db")
    t = PaperTrader(feed, store, SYMS, "1h", Params(), 500.0, warmup=warmup, now_fn=market.now,
                    log=lambda m: None)
    t.bootstrap()
    return t, store


def test_bootstrap_fresh_starts_run(tmp_path, market):
    t, store = make_trader(tmp_path, market)
    assert store.status()["mode"] == "paper" and store.status()["symbols"] == SYMS
    assert t.run_once() is True
    assert store.load_state("executor") is not None and store.load_state("risk") is not None
    assert store.load_state("last_ts") == market.df.ts.iloc[149].isoformat()


def test_run_once_no_new_bar_returns_false_and_writes_nothing(tmp_path, market):
    t, store = make_trader(tmp_path, market)
    assert t.run_once() is True
    n = len(store.equity_curve())
    assert t.run_once() is False
    assert len(store.equity_curve()) == n and store.risk_events() == []


def test_run_once_new_bar_steps_and_persists(tmp_path, market):
    t, store = make_trader(tmp_path, market)
    t.run_once()
    market.visible += 1
    assert t.run_once() is True
    curve = store.equity_curve()
    assert len(curve) == 2 and curve[-1]["ts"] == market.df.ts.iloc[150]
    sig = store.latest_signals()
    assert {s["symbol"] for s in sig} == set(SYMS)
    assert {s["agent"] for s in sig} == {"trend", "meanrev", "breakout"}
    assert store.status()["last_ts"] == market.df.ts.iloc[150]
    assert store.status()["n_positions"] == len(t.engine.executor.positions())


def test_restart_resumes_identical_state(tmp_path, market):
    a, store = make_trader(tmp_path, market)
    for _ in range(3):
        a.run_once()
        market.visible += 1
    b, _ = make_trader(tmp_path, market, store=store)
    assert b.last_ts == a.last_ts
    assert b.engine.executor.to_state() == a.engine.executor.to_state()
    assert b.engine.risk.to_state() == a.engine.risk.to_state()
    assert b.run_once() is True
    row_b = store.equity_curve()[-1]
    assert a.run_once() is True  # same bar, same answer
    row_a = store.equity_curve()[-1]
    assert row_a == row_b
    assert a.engine.executor.fills() == b.engine.executor.fills()
    assert len(store.equity_curve()) == 4


def test_data_error_writes_event_and_stays_active(tmp_path, market):
    t, store = make_trader(tmp_path, market)
    t.run_once()
    market.visible += 1
    market.fail = True
    assert t.run_once() is False
    ev = store.risk_events()
    assert ev and ev[0]["kind"] == "data"
    assert t.engine.risk.state == RiskState.ACTIVE
    market.fail = False
    assert t.run_once() is True


def test_unexpected_error_halts_and_persists(tmp_path, market, monkeypatch):
    t, store = make_trader(tmp_path, market)
    t.run_once()
    market.visible += 1

    def boom(windows):
        raise ValueError("bug")

    monkeypatch.setattr(t.engine, "step", boom)
    with pytest.raises(ValueError):
        t.run_once()
    assert store.risk_events()[0]["kind"] == "error"
    assert store.load_state("risk")["state"] == "halted"
    assert t.engine.risk.state == RiskState.HALTED


def test_next_wake_is_close_plus_delay(tmp_path, market):
    t, _ = make_trader(tmp_path, market)
    now = datetime(2024, 3, 1, 10, 17, tzinfo=UTC)
    assert t.next_wake(now) == datetime(2024, 3, 1, 11, 0, tzinfo=UTC) + timedelta(seconds=POLL_DELAY_S)


def test_run_forever_steps_then_sleeps_to_next_close(tmp_path, market):
    t, store = make_trader(tmp_path, market)
    sleeps = []
    first_now = market.now()

    def sleeper(s):
        sleeps.append(s)
        market.late = H  # the clock passes the next close but the exchange publishes nothing

    t.run_forever(sleep_fn=sleeper, max_iterations=2)
    assert len(store.equity_curve()) == 1
    # first iteration stepped -> sleep until next close + delay; second: bar is late -> short retry
    assert sleeps[0] == pytest.approx((t.next_wake(first_now) - first_now).total_seconds())
    assert sleeps[1] <= 60


def test_cli_paper_once(tmp_path, market, monkeypatch):
    monkeypatch.setattr(
        cli, "DataFeed",
        lambda **kw: DataFeed(root=kw["root"], fetch_fn=market.fetch, backoff_s=0,
                              now_fn=lambda: int(market.now().timestamp() * 1000)),
    )
    monkeypatch.setattr(cli, "PaperTrader", partial(PaperTrader, now_fn=market.now))
    res = CliRunner().invoke(
        cli.app,
        ["paper", "run", "--once", "--symbols", "BTC/USDT,ETH/USDT", "--db", str(tmp_path / "p.db"),
         "--data-root", str(tmp_path), "--warmup", "100"],
    )
    assert res.exit_code == 0, res.output
    assert "stepped" in res.output
    res2 = CliRunner().invoke(
        cli.app,
        ["paper", "run", "--once", "--symbols", "BTC/USDT,ETH/USDT", "--db", str(tmp_path / "p.db"),
         "--data-root", str(tmp_path), "--warmup", "100"],
    )
    assert res2.exit_code == 0 and "no new bar" in res2.output


def test_insufficient_history_is_a_data_event(tmp_path):
    m = Market(n=120)
    m.visible = 50
    t, store = make_trader(tmp_path, m, warmup=100)
    assert t.run_once() is False
    assert store.risk_events()[0]["kind"] == "data"


def test_feed_error_type_is_reexported():
    assert DataFeedError is not None and isinstance(pd.Timestamp("2024-01-01"), pd.Timestamp)
