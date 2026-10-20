"""Walk-forward validation: choose params on a train window, report only test windows."""

from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd

from cryptofx.backtest.metrics import Metrics, compute_metrics
from cryptofx.backtest.runner import (
    DEFAULT_WARMUP,
    Params,
    benchmark_for,
    run_backtest,
    slice_with_warmup,
)
from cryptofx.core.models import Trade

PARAM_GRID = [Params(threshold=t) for t in (0.5, 0.6, 0.7)]


@dataclass
class WalkForwardResult:
    windows: list[dict]
    stitched_equity: pd.Series
    metrics: Metrics
    benchmark: Metrics
    trades: list[Trade]


def windows(
    start: datetime, end: datetime, train_days: int = 180, test_days: int = 60
) -> list[tuple[datetime, datetime, datetime]]:
    out = []
    train_start = start
    while True:
        test_start = train_start + timedelta(days=train_days)
        if test_start >= end:
            break
        test_end = min(test_start + timedelta(days=test_days), end)
        out.append((train_start, test_start, test_end))
        train_start = train_start + timedelta(days=test_days)
    return out


def _sharpe_key(m: Metrics) -> float:
    return m.sharpe if m.sharpe == m.sharpe else float("-inf")  # NaN -> -inf


def walk_forward(
    candles: dict[str, pd.DataFrame],
    start: datetime,
    end: datetime,
    grid: list[Params] = PARAM_GRID,
    initial_equity: float = 500.0,
    tf: str = "1h",
    train_days: int = 180,
    test_days: int = 60,
    warmup: int = DEFAULT_WARMUP,
) -> WalkForwardResult:
    results: list[dict] = []
    curves: list[pd.Series] = []
    trades: list[Trade] = []
    equity = initial_equity
    for train_start, test_start, test_end in windows(start, end, train_days, test_days):
        train = slice_with_warmup(candles, train_start, test_start - timedelta(seconds=1), warmup)
        best: tuple[float, Params] | None = None
        for params in grid:
            r = run_backtest(train, params, initial_equity=equity, tf=tf, warmup=warmup)
            key = _sharpe_key(r.metrics)
            if best is None or key > best[0]:
                best = (key, params)
        assert best is not None
        test = slice_with_warmup(candles, test_start, test_end, warmup)
        r = run_backtest(test, best[1], initial_equity=equity, tf=tf, warmup=warmup)
        results.append(
            {
                "train_start": train_start,
                "test_start": test_start,
                "test_end": test_end,
                "chosen": best[1],
                "train_sharpe": best[0],
                "test_metrics": r.metrics,
                "start_equity": equity,
                "end_equity": float(r.equity.equity.iloc[-1]),
            }
        )
        curves.append(r.equity.equity)
        trades.extend(r.trades)
        equity = float(r.equity.equity.iloc[-1])
    if not curves:
        raise ValueError("no walk-forward windows fit in the requested date range")
    stitched = pd.concat(curves)
    stitched = stitched[~stitched.index.duplicated(keep="last")].sort_index()
    metrics = compute_metrics(stitched, trades)
    bench = benchmark_for(candles, stitched.index, initial_equity)
    return WalkForwardResult(
        windows=results, stitched_equity=stitched, metrics=metrics, benchmark=bench, trades=trades
    )
