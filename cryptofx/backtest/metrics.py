"""Performance metrics for an equity curve and a list of closed trades."""

import math
from dataclasses import dataclass, fields

import numpy as np
import pandas as pd

from cryptofx.core.models import Trade

ANNUAL_DAYS = 365.0


@dataclass(frozen=True)
class Metrics:
    total_return: float
    cagr: float
    sharpe: float
    sortino: float
    max_drawdown: float
    profit_factor: float
    win_rate: float
    avg_win: float
    avg_loss: float
    expectancy: float
    trade_count: int
    exposure: float
    monthly_returns: pd.Series


def _daily(equity: pd.Series) -> pd.Series:
    return equity.resample("1D").last().dropna()


def _ratio(returns: pd.Series, downside_only: bool) -> float:
    if len(returns) < 2:
        return 0.0
    mean = returns.mean()
    if downside_only:
        downside = returns[returns < 0]
        denom = math.sqrt((downside**2).mean()) if len(downside) else 0.0
    else:
        denom = returns.std(ddof=0)
    if np.isnan(denom) or denom <= 1e-12:
        return math.inf if mean > 1e-12 else 0.0
    return float(mean / denom * math.sqrt(ANNUAL_DAYS))


def compute_metrics(
    equity: pd.Series, trades: list[Trade], n_positions: pd.Series | None = None
) -> Metrics:
    equity = equity.astype(float)
    start, end = float(equity.iloc[0]), float(equity.iloc[-1])
    total_return = end / start - 1.0 if start else 0.0

    days = max((equity.index[-1] - equity.index[0]).total_seconds() / 86400.0, 1e-9)
    years = days / ANNUAL_DAYS
    cagr = (end / start) ** (1 / years) - 1.0 if start > 0 and end > 0 and years > 0 else 0.0

    daily = _daily(equity)
    rets = daily.pct_change().dropna()
    sharpe = _ratio(rets, downside_only=False)
    sortino = _ratio(rets, downside_only=True)

    peak = equity.cummax()
    max_dd = float(((peak - equity) / peak).max()) if len(equity) else 0.0

    pnls = np.array([t.pnl for t in trades], dtype=float)
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    if len(pnls) == 0:
        profit_factor = 0.0
    elif len(losses) == 0:
        profit_factor = math.inf if len(wins) else 0.0
    else:
        profit_factor = float(wins.sum() / -losses.sum())
    win_rate = float(len(wins) / len(pnls)) if len(pnls) else 0.0
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0
    expectancy = float(pnls.mean()) if len(pnls) else 0.0

    exposure = float((n_positions > 0).mean()) if n_positions is not None and len(n_positions) else 0.0

    monthly = equity.resample("ME").last().pct_change()

    return Metrics(
        total_return=float(total_return),
        cagr=float(cagr),
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=max_dd,
        profit_factor=profit_factor,
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        expectancy=expectancy,
        trade_count=len(pnls),
        exposure=exposure,
        monthly_returns=monthly,
    )


def buy_and_hold(close: pd.Series, initial_equity: float) -> pd.Series:
    return initial_equity * close / float(close.iloc[0])


def to_table(m: Metrics, name: str) -> str:
    lines = [f"{'metric':<16}{name:>14}", "-" * 30]
    for f in fields(m):
        if f.name == "monthly_returns":
            continue
        v = getattr(m, f.name)
        if isinstance(v, float):
            s = "inf" if math.isinf(v) else f"{v:.4f}"
        else:
            s = str(v)
        lines.append(f"{f.name:<16}{s:>14}")
    return "\n".join(lines)
