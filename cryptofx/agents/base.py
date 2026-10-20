"""Agent protocol: pure function of candles (and regime) to a Signal."""

from typing import Protocol

import pandas as pd

from cryptofx.core.models import Regime, Signal


class Agent(Protocol):
    name: str

    def signal(self, candles: pd.DataFrame, regime: Regime) -> Signal: ...
