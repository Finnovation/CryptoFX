from cryptofx.agents.trend import TrendAgent
from cryptofx.core.models import FLAT_SIGNAL, Direction, Regime
from tests.synth import downtrend, uptrend


def test_trend_long_in_uptrend():
    s = TrendAgent().signal(uptrend(), Regime.TRENDING)
    assert s.direction == Direction.LONG
    assert s.confidence > 0.5


def test_trend_flat_in_downtrend():
    assert TrendAgent().signal(downtrend(), Regime.TRENDING).direction == Direction.FLAT


def test_trend_flat_on_short_history():
    assert TrendAgent().signal(uptrend(n=10), Regime.TRENDING) == FLAT_SIGNAL
