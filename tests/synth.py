"""Synthetic candle generators for tests. Hourly bars from 2024-01-01 UTC."""

import numpy as np
import pandas as pd


def _frame(close: np.ndarray) -> pd.DataFrame:
    n = len(close)
    ts = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    open_ = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame(
        {
            "ts": ts,
            "open": open_,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.full(n, 1000.0),
            "gap": np.zeros(n, dtype=bool),
        }
    )


def uptrend(n: int = 200, start: float = 100.0, step: float = 1.0, noise: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    close = start + step * np.arange(n) + rng.normal(0, noise, n)
    return _frame(close.astype(float))


def downtrend(n: int = 200, start: float = 300.0, step: float = 1.0, noise: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    close = start - step * np.arange(n) + rng.normal(0, noise, n)
    return _frame(close.astype(float))


def ranging(n: int = 200, centre: float = 100.0, amp: float = 5.0, period: int = 20) -> pd.DataFrame:
    close = centre + amp * np.sin(2 * np.pi * np.arange(n) / period)
    return _frame(close.astype(float))


def flat(n: int = 200, price: float = 100.0) -> pd.DataFrame:
    return _frame(np.full(n, price, dtype=float))
