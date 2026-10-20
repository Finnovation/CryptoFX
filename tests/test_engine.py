import pandas as pd
import pytest

from cryptofx.agents.regime import RegimeAgent
from cryptofx.agents.trend import TrendAgent
from cryptofx.core.allocator import Allocator
from cryptofx.core.engine import Engine
from cryptofx.core.models import Direction, Regime, Side, Signal
from cryptofx.exec.backtest import BacktestExecutor
from cryptofx.risk.manager import RiskManager
from tests.synth import uptrend

SYM = "BTC/USDT"


class LongAtBar:
    """Stub agent: LONG only when the last candle timestamp equals ``ts``."""

    name = "trend"

    def __init__(self, ts: pd.Timestamp) -> None:
        self.ts = ts

    def signal(self, candles: pd.DataFrame, regime: Regime) -> Signal:
        if candles.ts.iloc[-1] == self.ts:
            return Signal(direction=Direction.LONG, confidence=1.0, reason="stub")
        return Signal(direction=Direction.FLAT, confidence=0.0, reason="stub")


def make_engine(agents, warmup=100, equity=1000.0):
    return Engine(
        agents, RegimeAgent(), Allocator(), RiskManager(), BacktestExecutor(equity), [SYM], "1h",
        warmup=warmup,
    )


def test_engine_uptrend_single_symbol_profits():
    eng = make_engine([TrendAgent()])
    curve = eng.run({SYM: uptrend(n=400)})
    assert curve.equity.iloc[-1] > 1000.0
    assert len(eng.fills) >= 1
    assert list(curve.columns) == ["equity", "cash", "n_positions", f"regime_{SYM}"]
    assert curve.index.name == "ts"


def test_engine_orders_fill_next_bar_open():
    df = uptrend(n=300)
    k = 150
    eng = make_engine([LongAtBar(df.ts.iloc[k])])
    eng.run({SYM: df})
    buys = [f for f in eng.fills if f.side == Side.BUY]
    assert len(buys) == 1
    fill = buys[0]
    assert pd.Timestamp(fill.ts) == df.ts.iloc[k + 1]
    open_next = df.open.iloc[k + 1]
    assert open_next * 1.0005 <= fill.price <= open_next * (1 + 0.0005 + 0.01)


def test_engine_signal_exit_sells_next_bar():
    df = uptrend(n=300)
    k = 150
    eng = make_engine([LongAtBar(df.ts.iloc[k])])
    eng.run({SYM: df})
    # LONG only at bar k -> score drops at k+1 -> SELL pending -> fills at k+2
    assert len(eng.trades) == 1
    t = eng.trades[0]
    assert pd.Timestamp(t.exit_ts) == df.ts.iloc[k + 2]
    assert t.exit_reason == "signal exit"


def test_engine_skips_symbol_with_long_gap():
    df = uptrend(n=300)
    k = 150
    df.loc[df.index[k - 10 : k - 5], "gap"] = True  # 5 consecutive gap bars just before k
    eng = make_engine([LongAtBar(df.ts.iloc[k])])
    eng.run({SYM: df})
    assert eng.fills == []
    assert eng.rejections == []


def test_engine_deterministic():
    a = make_engine([TrendAgent()]).run({SYM: uptrend(n=400, noise=2.0)})
    b = make_engine([TrendAgent()]).run({SYM: uptrend(n=400, noise=2.0)})
    pd.testing.assert_frame_equal(a, b)


def test_engine_raises_on_short_data():
    with pytest.raises(ValueError):
        make_engine([TrendAgent()], warmup=100).run({SYM: uptrend(n=100)})


def test_engine_aligns_symbols_on_shared_timestamps():
    a = uptrend(n=300)
    b = uptrend(n=250).copy()
    b["ts"] = b["ts"] + pd.Timedelta(hours=50)  # overlaps a on bars 50..299
    eng = Engine(
        [TrendAgent()], RegimeAgent(), Allocator(), RiskManager(), BacktestExecutor(1000.0),
        [SYM, "ETH/USDT"], "1h", warmup=100,
    )
    curve = eng.run({SYM: a, "ETH/USDT": b})
    assert len(curve) == 250 - 100
    assert "regime_ETH/USDT" in curve.columns
