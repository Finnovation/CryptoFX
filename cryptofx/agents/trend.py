"""Trend-following agent: EMA crossover filtered by ADX."""

import pandas as pd

from cryptofx.agents.indicators import adx, ema
from cryptofx.core.models import FLAT_SIGNAL, Direction, Regime, Signal


class TrendAgent:
    name = "trend"

    def __init__(
        self, fast: int = 20, slow: int = 50, adx_n: int = 14, adx_min: float = 20.0
    ) -> None:
        self.fast = fast
        self.slow = slow
        self.adx_n = adx_n
        self.adx_min = adx_min

    def signal(self, candles: pd.DataFrame, regime: Regime) -> Signal:
        if len(candles) < self.slow + self.adx_n:
            return FLAT_SIGNAL
        fast = ema(candles.close, self.fast).iloc[-1]
        slow = ema(candles.close, self.slow).iloc[-1]
        strength = adx(candles.high, candles.low, candles.close, self.adx_n).iloc[-1]
        if pd.isna(strength):
            return FLAT_SIGNAL
        if fast > slow and strength > self.adx_min:
            conf = min(1.0, (strength - self.adx_min) / 20.0 + 0.5)
            return Signal(
                direction=Direction.LONG,
                confidence=conf,
                reason=f"ema{self.fast}>ema{self.slow}, adx={strength:.1f}",
            )
        return Signal(
            direction=Direction.FLAT,
            confidence=0.0,
            reason=f"no trend (fast>slow={fast > slow}, adx={strength:.1f})",
        )
