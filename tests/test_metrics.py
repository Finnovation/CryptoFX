import math
from datetime import UTC, datetime

import pandas as pd
import pytest

from cryptofx.backtest.metrics import Metrics, buy_and_hold, compute_metrics, to_table
from cryptofx.core.models import Trade

T0 = datetime(2024, 1, 1, tzinfo=UTC)


def daily_index(n):
    return pd.date_range("2024-01-01", periods=n, freq="1D", tz="UTC")


def flat_equity(n=10, value=1000.0):
    return pd.Series([value] * n, index=daily_index(n))


def trade(pnl):
    return Trade(
        symbol="BTC/USDT", entry_ts=T0, exit_ts=T0, entry_price=100.0, exit_price=100.0 + pnl,
        size_base=1.0, pnl=pnl, fees=0.0, exit_reason="t",
    )


def test_max_drawdown_hand_value():
    eq = pd.Series([100.0, 120.0, 90.0, 130.0], index=daily_index(4))
    m = compute_metrics(eq, [])
    assert m.max_drawdown == pytest.approx(0.25)
    assert m.total_return == pytest.approx(0.30)


def test_profit_factor_and_win_rate():
    trades = [trade(10), trade(20), trade(-15)]
    m = compute_metrics(flat_equity(), trades)
    assert m.profit_factor == pytest.approx(2.0)
    assert m.win_rate == pytest.approx(2 / 3)
    assert m.expectancy == pytest.approx(5.0)
    assert m.avg_win == pytest.approx(15.0)
    assert m.avg_loss == pytest.approx(-15.0)
    assert m.trade_count == 3


def test_no_trades_profit_factor_zero():
    m = compute_metrics(flat_equity(), [])
    assert m.profit_factor == 0.0 and m.trade_count == 0 and m.win_rate == 0.0


def test_only_wins_profit_factor_inf():
    assert math.isinf(compute_metrics(flat_equity(), [trade(5)]).profit_factor)


def test_sharpe_of_constant_growth_is_positive():
    eq = pd.Series([1000.0 * 1.001**i for i in range(60)], index=daily_index(60))
    m = compute_metrics(eq, [])
    assert m.sharpe > 0 and m.sortino > 0 and m.cagr > 0


def test_flat_equity_sharpe_is_zero_not_nan():
    m = compute_metrics(flat_equity(), [])
    assert m.sharpe == 0.0 and m.sortino == 0.0 and m.max_drawdown == 0.0


def test_exposure_fraction():
    n_pos = pd.Series([0, 1, 1, 0], index=daily_index(4))
    assert compute_metrics(flat_equity(4), [], n_pos).exposure == pytest.approx(0.5)
    assert compute_metrics(flat_equity(4), []).exposure == 0.0


def test_monthly_returns_series():
    idx = pd.date_range("2024-01-01", periods=91, freq="1D", tz="UTC")
    eq = pd.Series([1000.0 + i for i in range(91)], index=idx)
    mr = compute_metrics(eq, []).monthly_returns
    assert isinstance(mr, pd.Series) and len(mr) >= 2
    assert (mr.dropna() > 0).all()


def test_buy_and_hold_scales_by_close():
    close = pd.Series([100.0, 110.0, 90.0], index=daily_index(3))
    bh = buy_and_hold(close, 500.0)
    assert bh.tolist() == pytest.approx([500.0, 550.0, 450.0])


def test_to_table_contains_fields():
    m = compute_metrics(flat_equity(), [trade(10)])
    assert isinstance(m, Metrics)
    txt = to_table(m, "system")
    assert "system" in txt and "max_drawdown" in txt and "profit_factor" in txt
