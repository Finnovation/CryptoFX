from cryptofx.agents.breakout import BreakoutAgent
from cryptofx.core.models import FLAT_SIGNAL, Direction, Regime
from tests.synth import flat


def test_breakout_long_on_new_high():
    df = flat(n=100)
    df.loc[df.index[-1], ["close", "high"]] = [110.0, 110.5]
    assert BreakoutAgent().signal(df, Regime.TRENDING).direction == Direction.LONG


def test_breakout_flat_in_range():
    assert BreakoutAgent().signal(flat(), Regime.RANGING).direction == Direction.FLAT


def test_breakout_short_history():
    assert BreakoutAgent().signal(flat(n=5), Regime.TRENDING) == FLAT_SIGNAL
