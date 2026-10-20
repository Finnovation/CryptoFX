"""Hand-written technical indicators. All functions take and return pd.Series.

Cores run on numpy arrays; pandas is only used at the boundary so the engine's
per-bar recompute over a small window stays cheap."""

import numpy as np
import pandas as pd


def _np(s: pd.Series) -> np.ndarray:
    return s.to_numpy(dtype=float)


def _wrap(arr: np.ndarray, like: pd.Series) -> pd.Series:
    return pd.Series(arr, index=like.index)


def _wilder_np(arr: np.ndarray, n: int, offset: int = 0) -> np.ndarray:
    """Wilder smoothing seeded with the SMA of the first n valid values.

    ``offset`` is how many leading entries of ``arr`` are undefined (NaN)."""
    out = np.full(len(arr), np.nan)
    if len(arr) - offset < n:
        return out
    seg = arr[offset:]
    if np.isnan(seg[:n]).any():
        return out
    prev = seg[:n].mean()
    out[offset + n - 1] = prev
    k = (n - 1) / n
    inv = 1.0 / n
    for i in range(n, len(seg)):
        prev = prev * k + seg[i] * inv
        out[offset + i] = prev
    return out


def ema(close: pd.Series, n: int) -> pd.Series:
    return close.ewm(span=n, adjust=False).mean()


def sma(close: pd.Series, n: int) -> pd.Series:
    return close.rolling(n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    c = _np(close)
    delta = np.empty_like(c)
    delta[0] = np.nan
    delta[1:] = np.diff(c)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain[0] = loss[0] = np.nan
    avg_gain = _wilder_np(gain, n, offset=1)
    avg_loss = _wilder_np(loss, n, offset=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    out = np.where(avg_loss == 0, 100.0, out)
    out = np.where((avg_gain == 0) & (avg_loss == 0), 50.0, out)
    out = np.where(np.isnan(avg_gain) | np.isnan(avg_loss), np.nan, out)
    return _wrap(out, close)


def _tr_np(h: np.ndarray, lo: np.ndarray, c: np.ndarray) -> np.ndarray:
    tr = np.empty_like(c)
    tr[0] = np.nan
    pc = c[:-1]
    tr[1:] = np.maximum.reduce([h[1:] - lo[1:], np.abs(h[1:] - pc), np.abs(lo[1:] - pc)])
    return tr


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    return _wrap(_tr_np(_np(high), _np(low), _np(close)), close)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    tr = _tr_np(_np(high), _np(low), _np(close))
    return _wrap(_wilder_np(tr, n, offset=1), close)


def adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    h, lo, c = _np(high), _np(low), _np(close)
    up = np.empty_like(h)
    down = np.empty_like(h)
    up[0] = down[0] = np.nan
    up[1:] = np.diff(h)
    down[1:] = -np.diff(lo)
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    plus_dm[0] = minus_dm[0] = np.nan
    tr = _tr_np(h, lo, c)
    atr_s = _wilder_np(tr, n, offset=1)
    plus_s = _wilder_np(plus_dm, n, offset=1)
    minus_s = _wilder_np(minus_dm, n, offset=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        plus_di = 100.0 * plus_s / atr_s
        minus_di = 100.0 * minus_s / atr_s
        denom = plus_di + minus_di
        dx = np.where(denom != 0, 100.0 * np.abs(plus_di - minus_di) / denom, 0.0)
    dx = np.where(np.isnan(atr_s), np.nan, dx)
    first = n  # dx defined from index n (offset 1 + n-1)
    return _wrap(_wilder_np(dx, n, offset=first), close)


def bollinger(
    close: pd.Series, n: int = 20, k: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    roll = close.rolling(n)
    mid = roll.mean()
    std = roll.std(ddof=0)
    return mid, mid + k * std, mid - k * std


def donchian(high: pd.Series, low: pd.Series, n: int = 20) -> tuple[pd.Series, pd.Series]:
    return high.rolling(n).max().shift(1), low.rolling(n).min().shift(1)


def realised_vol(close: pd.Series, n: int = 20) -> pd.Series:
    return np.log(close).diff().rolling(n).std(ddof=0)
