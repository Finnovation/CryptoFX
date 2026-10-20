"""Hand-written technical indicators. All functions take and return pd.Series."""

import numpy as np
import pandas as pd


def ema(close: pd.Series, n: int) -> pd.Series:
    return close.ewm(span=n, adjust=False).mean()


def sma(close: pd.Series, n: int) -> pd.Series:
    return close.rolling(n).mean()


def _wilder(values: pd.Series, n: int) -> pd.Series:
    """Wilder smoothing seeded with the SMA of the first n values."""
    out = pd.Series(np.nan, index=values.index, dtype=float)
    arr = values.to_numpy(dtype=float)
    if len(arr) < n:
        return out
    # find first index where n consecutive non-NaN values exist
    start = None
    for i in range(len(arr) - n + 1):
        if not np.isnan(arr[i : i + n]).any():
            start = i
            break
    if start is None:
        return out
    prev = arr[start : start + n].mean()
    out.iloc[start + n - 1] = prev
    for i in range(start + n, len(arr)):
        prev = (prev * (n - 1) + arr[i]) / n
        out.iloc[i] = prev
    return out


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = _wilder(gain.iloc[1:], n).reindex(close.index)
    avg_loss = _wilder(loss.iloc[1:], n).reindex(close.index)
    rs = avg_gain / avg_loss
    out = 100.0 - 100.0 / (1.0 + rs)
    out = out.where(avg_loss != 0, 100.0)
    out = out.where(~((avg_gain == 0) & (avg_loss == 0)), 50.0)
    out[avg_gain.isna() | avg_loss.isna()] = np.nan
    return out


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    tr.iloc[0] = np.nan
    return tr


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    tr = true_range(high, low, close)
    return _wilder(tr.iloc[1:], n).reindex(close.index)


def adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    up = high.diff()
    down = -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=high.index)
    tr = true_range(high, low, close)
    atr_s = _wilder(tr.iloc[1:], n).reindex(close.index)
    plus_s = _wilder(plus_dm.iloc[1:], n).reindex(close.index)
    minus_s = _wilder(minus_dm.iloc[1:], n).reindex(close.index)
    plus_di = 100.0 * plus_s / atr_s
    minus_di = 100.0 * minus_s / atr_s
    denom = plus_di + minus_di
    dx = (100.0 * (plus_di - minus_di).abs() / denom).where(denom != 0, 0.0)
    dx[atr_s.isna()] = np.nan
    return _wilder(dx.dropna(), n).reindex(close.index)


def bollinger(
    close: pd.Series, n: int = 20, k: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = close.rolling(n).mean()
    std = close.rolling(n).std(ddof=0)
    return mid, mid + k * std, mid - k * std


def donchian(high: pd.Series, low: pd.Series, n: int = 20) -> tuple[pd.Series, pd.Series]:
    return high.rolling(n).max().shift(1), low.rolling(n).min().shift(1)


def realised_vol(close: pd.Series, n: int = 20) -> pd.Series:
    return np.log(close).diff().rolling(n).std(ddof=0)
