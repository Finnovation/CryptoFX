"""Hand-written technical indicators. All functions take and return pd.Series."""

import numpy as np
import pandas as pd


def ema(close: pd.Series, n: int) -> pd.Series:
    return close.ewm(span=n, adjust=False).mean()


def sma(close: pd.Series, n: int) -> pd.Series:
    return close.rolling(n).mean()


def _wilder(values: pd.Series, n: int) -> pd.Series:
    """Wilder smoothing seeded with the SMA of the first n non-NaN values."""
    arr = values.to_numpy(dtype=float)
    out = np.full(len(arr), np.nan)
    if len(arr) < n:
        return pd.Series(out, index=values.index)
    valid = ~np.isnan(arr)
    # first index where n consecutive valid values start
    run = 0
    start = None
    for i, ok in enumerate(valid):
        run = run + 1 if ok else 0
        if run == n:
            start = i - n + 1
            break
    if start is None:
        return pd.Series(out, index=values.index)
    prev = arr[start : start + n].mean()
    out[start + n - 1] = prev
    k = (n - 1) / n
    for i in range(start + n, len(arr)):
        prev = prev * k + arr[i] / n
        out[i] = prev
    return pd.Series(out, index=values.index)


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
