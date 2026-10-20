"""Breakout agent: Donchian channel breakout with a tighter exit channel."""

import pandas as pd

from cryptofx.agents.indicators import donchian
from cryptofx.core.models import FLAT_SIGNAL, Direction, Regime, Signal


class BreakoutAgent:
    name = "breakout"

    def __init__(self, entry_n: int = 20, exit_n: int = 10) -> None:
        self.entry_n = entry_n
        self.exit_n = exit_n

    def signal(self, candles: pd.DataFrame, regime: Regime) -> Signal:
        if len(candles) < self.entry_n + 1:
            return FLAT_SIGNAL
        upper, _ = donchian(candles.high, candles.low, self.entry_n)
        _, exit_lower = donchian(candles.high, candles.low, self.exit_n)
        close = candles.close
        breakout = close > upper
        last_close = close.iloc[-1]
        last_upper = upper.iloc[-1]
        if pd.isna(last_upper):
            return FLAT_SIGNAL
        if breakout.iloc[-1]:
            conf = min(1.0, 0.5 + (last_close - last_upper) / last_upper * 10)
            return Signal(
                direction=Direction.LONG,
                confidence=conf,
                reason=f"breakout above {last_upper:.2f}",
            )
        recent = bool(breakout.iloc[-self.entry_n :].any())
        if recent and not pd.isna(exit_lower.iloc[-1]) and last_close > exit_lower.iloc[-1]:
            return Signal(
                direction=Direction.LONG,
                confidence=0.5,
                reason=f"holding breakout above exit {exit_lower.iloc[-1]:.2f}",
            )
        return Signal(direction=Direction.FLAT, confidence=0.0, reason="no breakout")
