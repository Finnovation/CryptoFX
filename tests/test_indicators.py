import numpy as np
import pandas as pd

from cryptofx.agents.indicators import adx, bollinger, donchian, ema, rsi, true_range


def test_ema_hand_values():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    assert ema(s, 3).round(4).tolist() == [1.0, 1.5, 2.25, 3.125, 4.0625]


def test_rsi_hand_value():
    s = pd.Series([1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5, 6, 5, 6, 7], dtype=float)
    assert round(rsi(s, 14).iloc[14], 2) == 71.43
    assert rsi(s, 14).iloc[:14].isna().all()


def test_rsi_extremes():
    assert rsi(pd.Series(range(1, 31), dtype=float), 14).iloc[-1] == 100.0
    assert rsi(pd.Series(range(30, 0, -1), dtype=float), 14).iloc[-1] == 0.0


def test_true_range_hand_values():
    high = pd.Series([10.0, 12.0, 11.0])
    low = pd.Series([8.0, 9.0, 7.0])
    close = pd.Series([9.0, 11.0, 8.0])
    assert true_range(high, low, close).tolist()[1:] == [3.0, 4.0]


def test_bollinger_hand_values():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    mid, up, lo = bollinger(s, 5, 2.0)
    assert mid.iloc[-1] == 3.0
    assert round(up.iloc[-1], 4) == 5.8284
    assert round(lo.iloc[-1], 4) == 0.1716


def test_donchian_excludes_current_bar():
    high = pd.Series([1.0, 2.0, 3.0, 10.0])
    low = pd.Series([0.5, 1.0, 2.0, 9.0])
    up, lo = donchian(high, low, 3)
    assert up.iloc[-1] == 3.0
    assert lo.iloc[-1] == 0.5


def test_adx_strong_trend_vs_flat():
    n = 100
    trend = pd.Series(np.arange(n, dtype=float) + 100)
    flat = pd.Series(np.full(n, 100.0))
    assert adx(trend + 1, trend - 1, trend, 14).iloc[-1] > 25
    flat_adx = adx(flat + 0.1, flat - 0.1, flat, 14).iloc[-1]
    assert flat_adx < 5 or np.isnan(flat_adx)
