"""Mean-reversion agent: RSI oversold plus close below the lower Bollinger band."""

import pandas as pd

from cryptofx.agents.indicators import bollinger, rsi
from cryptofx.core.models import FLAT_SIGNAL, Direction, Regime, Signal


class MeanRevAgent:
    name = "meanrev"

    def __init__(
        self,
        rsi_n: int = 14,
        rsi_entry: float = 30.0,
        rsi_exit: float = 50.0,
        bb_n: int = 20,
        bb_k: float = 2.0,
    ) -> None:
        self.rsi_n = rsi_n
        self.rsi_entry = rsi_entry
        self.rsi_exit = rsi_exit
        self.bb_n = bb_n
        self.bb_k = bb_k

    def signal(self, candles: pd.DataFrame, regime: Regime) -> Signal:
        if len(candles) < max(self.rsi_n, self.bb_n) + 1:
            return FLAT_SIGNAL
        r = rsi(candles.close, self.rsi_n)
        _, _, lower = bollinger(candles.close, self.bb_n, self.bb_k)
        entry = (r < self.rsi_entry) & (candles.close < lower)
        last_rsi = r.iloc[-1]
        if pd.isna(last_rsi):
            return FLAT_SIGNAL
        if entry.iloc[-1]:
            conf = min(1.0, (self.rsi_entry - last_rsi) / self.rsi_entry + 0.5)
            return Signal(
                direction=Direction.LONG,
                confidence=conf,
                reason=f"oversold rsi={last_rsi:.1f} below lower band",
            )
        recent_entry = bool(entry.iloc[-self.bb_n :].any())
        if recent_entry and last_rsi < self.rsi_exit:
            conf = min(1.0, (self.rsi_exit - last_rsi) / self.rsi_exit + 0.5)
            return Signal(
                direction=Direction.LONG,
                confidence=conf,
                reason=f"holding reversion rsi={last_rsi:.1f}",
            )
        return Signal(direction=Direction.FLAT, confidence=0.0, reason=f"rsi={last_rsi:.1f}")
