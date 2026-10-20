"""Builds the standard engine and runs a single backtest with a benchmark."""

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from cryptofx.agents.breakout import BreakoutAgent
from cryptofx.agents.meanrev import MeanRevAgent
from cryptofx.agents.regime import RegimeAgent
from cryptofx.agents.trend import TrendAgent
from cryptofx.backtest.metrics import Metrics, buy_and_hold, compute_metrics
from cryptofx.core.allocator import Allocator
from cryptofx.core.engine import Engine
from cryptofx.core.models import Trade
from cryptofx.exec.backtest import BacktestExecutor
from cryptofx.exec.base import Executor
from cryptofx.risk.manager import RiskManager

DEFAULT_WARMUP = 200


@dataclass(frozen=True)
class Params:
    threshold: float = 0.6
    trend_fast: int = 20
    trend_slow: int = 50


@dataclass
class BacktestResult:
    equity: pd.DataFrame
    trades: list[Trade]
    metrics: Metrics
    benchmark: Metrics
    params: Params
    risk_events: list[dict]


def build_engine(
    params: Params,
    executor: Executor,
    symbols: list[str],
    tf: str,
    warmup: int = DEFAULT_WARMUP,
    risk: RiskManager | None = None,
) -> Engine:
    agents = [
        TrendAgent(fast=params.trend_fast, slow=params.trend_slow),
        MeanRevAgent(),
        BreakoutAgent(),
    ]
    return Engine(
        agents, RegimeAgent(), Allocator(threshold=params.threshold), risk or RiskManager(),
        executor, symbols, tf, warmup=warmup,
    )


def slice_with_warmup(
    candles: dict[str, pd.DataFrame], start: datetime, end: datetime, warmup: int
) -> dict[str, pd.DataFrame]:
    """Rows in [start, end] plus ``warmup`` rows before ``start`` so the engine's
    first decision lands on ``start``."""
    out = {}
    for sym, df in candles.items():
        in_range = df[(df.ts >= pd.Timestamp(start)) & (df.ts <= pd.Timestamp(end))]
        if in_range.empty:
            out[sym] = in_range.reset_index(drop=True)
            continue
        first = in_range.index[0]
        pos = df.index.get_loc(first)
        out[sym] = df.iloc[max(0, pos - warmup) : df.index.get_loc(in_range.index[-1]) + 1]
        out[sym] = out[sym].reset_index(drop=True)
    return out


def benchmark_for(
    candles: dict[str, pd.DataFrame], index: pd.Index, initial_equity: float
) -> Metrics:
    sym = next(iter(candles))
    df = candles[sym].set_index("ts")
    close = df.loc[df.index.isin(index), "close"]
    return compute_metrics(buy_and_hold(close, initial_equity), [])


def run_backtest(
    candles: dict[str, pd.DataFrame],
    params: Params,
    initial_equity: float = 500.0,
    tf: str = "1h",
    warmup: int = DEFAULT_WARMUP,
    executor: Executor | None = None,
    risk: RiskManager | None = None,
) -> BacktestResult:
    """Run one backtest. Pass ``executor``/``risk`` to continue from existing
    state (walk-forward uses this so positions, pending orders, drawdown peaks
    and halts survive window boundaries)."""
    symbols = list(candles)
    executor = executor or BacktestExecutor(initial_equity)
    trades_before = len(executor.trades())
    engine = build_engine(params, executor, symbols, tf, warmup=warmup, risk=risk)
    curve = engine.run(candles)
    trades = engine.trades[trades_before:]
    metrics = compute_metrics(curve.equity, trades, curve.n_positions)
    bench = benchmark_for(candles, curve.index, float(curve.equity.iloc[0]))
    return BacktestResult(
        equity=curve, trades=trades, metrics=metrics, benchmark=bench, params=params,
        risk_events=list(engine.risk_events),
    )
