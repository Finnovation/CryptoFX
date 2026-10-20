from datetime import UTC, datetime

import pandas as pd
import pytest

from cryptofx.backtest.runner import BacktestResult, Params, run_backtest
from cryptofx.backtest.walkforward import PARAM_GRID, WalkForwardResult, walk_forward, windows
from tests.synth import uptrend

SYM = "BTC/USDT"


def d(y, m, day):
    return datetime(y, m, day, tzinfo=UTC)


def test_windows_roll_by_test_days():
    w = windows(d(2024, 1, 1), d(2024, 12, 31))
    assert w[0] == (d(2024, 1, 1), d(2024, 6, 29), d(2024, 8, 28))
    assert w[1][1] == d(2024, 8, 28)
    assert w[-1][2] == d(2024, 12, 31)
    assert all(ts < te for _, ts, te in w)


def test_windows_empty_when_range_shorter_than_train():
    assert windows(d(2024, 1, 1), d(2024, 3, 1)) == []


def test_param_grid_thresholds():
    assert [p.threshold for p in PARAM_GRID] == [0.5, 0.6, 0.7]


def test_run_backtest_returns_benchmark():
    df = uptrend(n=400, noise=2.0)
    res = run_backtest({SYM: df}, Params(), initial_equity=500.0, warmup=100)
    assert isinstance(res, BacktestResult)
    first_ts = res.equity.index[0]
    close0 = df.loc[df.ts == first_ts, "close"].item()
    assert res.benchmark.total_return == pytest.approx(df.close.iloc[-1] / close0 - 1)
    assert res.equity.equity.iloc[0] == pytest.approx(500.0)
    assert res.params == Params()


def test_walk_forward_only_reports_test_windows():
    df = uptrend(n=600, noise=2.0)
    res = walk_forward(
        {SYM: df}, df.ts.iloc[0].to_pydatetime(), df.ts.iloc[-1].to_pydatetime(),
        grid=[Params(threshold=0.5), Params(threshold=0.7)], train_days=8, test_days=4, warmup=50,
    )
    assert isinstance(res, WalkForwardResult)
    assert len(res.windows) >= 3
    first_test_start = pd.Timestamp(res.windows[0]["test_start"])
    assert res.stitched_equity.index.min() >= first_test_start
    assert res.stitched_equity.index.is_monotonic_increasing
    assert res.stitched_equity.index.is_unique
    for w in res.windows:
        assert w["chosen"].threshold in (0.5, 0.7)
        assert "train_sharpe" in w and w["test_metrics"] is not None


def test_walk_forward_stitches_equity_continuously():
    df = uptrend(n=600, noise=2.0)
    res = walk_forward(
        {SYM: df}, df.ts.iloc[0].to_pydatetime(), df.ts.iloc[-1].to_pydatetime(),
        grid=[Params(threshold=0.6)], train_days=8, test_days=4, warmup=50, initial_equity=500.0,
    )
    assert res.windows[0]["start_equity"] == pytest.approx(500.0)
    for prev, nxt in zip(res.windows, res.windows[1:]):
        assert nxt["start_equity"] == pytest.approx(prev["end_equity"])
    assert res.metrics.total_return == pytest.approx(
        res.stitched_equity.iloc[-1] / res.stitched_equity.iloc[0] - 1
    )
    assert res.benchmark is not None
