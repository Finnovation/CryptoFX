from cryptofx.agents.regime import RegimeAgent
from cryptofx.core.models import Regime
from tests.synth import ranging, uptrend


def test_regime_trending():
    assert RegimeAgent().classify(uptrend()) == Regime.TRENDING


def test_regime_ranging():
    assert RegimeAgent().classify(ranging()) == Regime.RANGING


def test_regime_chaos_on_vol_spike():
    df = ranging(n=300)
    df.loc[df.index[-5:], "close"] *= [1.0, 1.3, 0.75, 1.3, 0.8]
    assert RegimeAgent().classify(df) == Regime.CHAOS


def test_regime_short_history_is_ranging():
    assert RegimeAgent().classify(uptrend(n=10)) == Regime.RANGING
