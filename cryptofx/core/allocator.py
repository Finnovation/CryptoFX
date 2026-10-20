"""Combines weighted agent votes into a single target position per symbol."""

import pandas as pd

from cryptofx.agents.indicators import atr
from cryptofx.core.models import Direction, Regime, Signal, TargetPosition

REGIME_WEIGHTS: dict[Regime, dict[str, float]] = {
    Regime.TRENDING: {"trend": 1.0, "meanrev": 0.3, "breakout": 1.0},
    Regime.RANGING: {"trend": 0.3, "meanrev": 1.0, "breakout": 0.3},
    Regime.CHAOS: {"trend": 0.0, "meanrev": 0.0, "breakout": 0.0},
}


class Allocator:
    def __init__(
        self, threshold: float = 0.6, atr_n: int = 14, stop_atr_mult: float = 2.0
    ) -> None:
        self.threshold = threshold
        self.atr_n = atr_n
        self.stop_atr_mult = stop_atr_mult

    def score(self, signals: dict[str, Signal], regime: Regime) -> float:
        weights = REGIME_WEIGHTS[regime]
        total = sum(weights.get(name, 0.0) for name in signals)
        if total <= 0:
            return 0.0
        votes = sum(
            weights.get(name, 0.0) * s.confidence * (1.0 if s.direction == Direction.LONG else 0.0)
            for name, s in signals.items()
        )
        return votes / total

    def allocate(
        self,
        symbol: str,
        signals: dict[str, Signal],
        regime: Regime,
        candles: pd.DataFrame,
        equity: float,
        risk_fraction: float,
    ) -> TargetPosition | None:
        if self.score(signals, regime) < self.threshold:
            return None
        a = atr(candles.high, candles.low, candles.close, self.atr_n).iloc[-1]
        if pd.isna(a) or a <= 0:
            return None
        close = float(candles.close.iloc[-1])
        stop_distance = self.stop_atr_mult * float(a)
        return TargetPosition(
            symbol=symbol,
            size_quote=risk_fraction * equity * close / stop_distance,
            stop_price=close - stop_distance,
        )
