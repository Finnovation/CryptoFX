"""Regime detector: tags the market as TRENDING, RANGING or CHAOS."""

import pandas as pd

from cryptofx.agents.indicators import adx, realised_vol
from cryptofx.core.models import Regime


class RegimeAgent:
    name = "regime"

    def __init__(
        self,
        adx_n: int = 14,
        adx_trend: float = 25.0,
        vol_n: int = 20,
        vol_lookback: int = 200,
        chaos_pct: float = 0.95,
    ) -> None:
        self.adx_n = adx_n
        self.adx_trend = adx_trend
        self.vol_n = vol_n
        self.vol_lookback = vol_lookback
        self.chaos_pct = chaos_pct

    def classify(self, candles: pd.DataFrame) -> Regime:
        if len(candles) < max(self.adx_n * 2, self.vol_n):
            return Regime.RANGING
        vol = realised_vol(candles.close, self.vol_n).dropna()
        if len(vol) >= 2:
            window = vol.iloc[-self.vol_lookback :]
            last = vol.iloc[-1]
            threshold = window.quantile(self.chaos_pct)
            if last > threshold and last > 0:
                return Regime.CHAOS
        strength = adx(candles.high, candles.low, candles.close, self.adx_n).iloc[-1]
        if not pd.isna(strength) and strength > self.adx_trend:
            return Regime.TRENDING
        return Regime.RANGING
