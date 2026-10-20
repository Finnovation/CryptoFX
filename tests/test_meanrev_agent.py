from cryptofx.agents.meanrev import MeanRevAgent
from cryptofx.core.models import FLAT_SIGNAL, Direction, Regime
from tests.synth import ranging, uptrend


def test_meanrev_long_after_oversold_dip():
    df = ranging(n=200)
    df.loc[df.index[-3:], "close"] -= [6, 9, 12]
    assert MeanRevAgent().signal(df, Regime.RANGING).direction == Direction.LONG


def test_meanrev_flat_in_uptrend():
    assert MeanRevAgent().signal(uptrend(), Regime.RANGING).direction == Direction.FLAT


def test_meanrev_short_history():
    assert MeanRevAgent().signal(ranging(n=5), Regime.RANGING) == FLAT_SIGNAL
