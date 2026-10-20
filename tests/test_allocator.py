import pytest

from cryptofx.agents.indicators import atr
from cryptofx.core.allocator import REGIME_WEIGHTS, Allocator
from cryptofx.core.models import Direction, Regime, Signal
from tests.synth import uptrend

LONG = Signal(direction=Direction.LONG, confidence=1.0, reason="")
FLAT = Signal(direction=Direction.FLAT, confidence=0.0, reason="")
ALL_LONG = {k: LONG for k in ("trend", "meanrev", "breakout")}


def test_regime_weights_match_spec():
    assert REGIME_WEIGHTS[Regime.TRENDING] == {"trend": 1.0, "meanrev": 0.3, "breakout": 1.0}
    assert REGIME_WEIGHTS[Regime.RANGING] == {"trend": 0.3, "meanrev": 1.0, "breakout": 0.3}
    assert REGIME_WEIGHTS[Regime.CHAOS] == {"trend": 0.0, "meanrev": 0.0, "breakout": 0.0}


def test_score_trending_all_long():
    assert Allocator().score(ALL_LONG, Regime.TRENDING) == pytest.approx(1.0)


def test_score_chaos_is_zero():
    assert Allocator().score(ALL_LONG, Regime.CHAOS) == 0.0


def test_score_trending_only_meanrev():
    sig = {"trend": FLAT, "meanrev": LONG, "breakout": FLAT}
    assert Allocator().score(sig, Regime.TRENDING) == pytest.approx(0.3 / 2.3)


def test_allocate_sizes_by_atr_stop():
    df = uptrend(n=100)
    tp = Allocator(threshold=0.0).allocate(
        "BTC/USDT", ALL_LONG, Regime.TRENDING, df, equity=1000.0, risk_fraction=0.01
    )
    close = df.close.iloc[-1]
    a = atr(df.high, df.low, df.close, 14).iloc[-1]
    assert tp.symbol == "BTC/USDT"
    assert tp.stop_price == pytest.approx(close - 2 * a)
    assert tp.size_quote == pytest.approx(0.01 * 1000 * close / (2 * a))


def test_allocate_none_below_threshold():
    sig = {"trend": FLAT, "meanrev": LONG, "breakout": FLAT}
    assert Allocator(threshold=0.6).allocate(
        "BTC/USDT", sig, Regime.TRENDING, uptrend(n=100), equity=1000.0, risk_fraction=0.01
    ) is None


def test_allocate_none_when_atr_unavailable():
    assert Allocator(threshold=0.0).allocate(
        "BTC/USDT", ALL_LONG, Regime.TRENDING, uptrend(n=5), equity=1000.0, risk_fraction=0.01
    ) is None
