"""CryptoFX command line."""

import math
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import typer

from cryptofx.backtest.metrics import Metrics, to_table
from cryptofx.backtest.runner import DEFAULT_WARMUP, Params, run_backtest, slice_with_warmup
from cryptofx.backtest.walkforward import walk_forward
from cryptofx.data.feed import DataFeed, DataFeedError

DEFAULT_SYMBOLS = "BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT,XRP/USDT"
TARGET_LINE = "reference: target 7-9% / realistic 0-5% per month"

app = typer.Typer(help="Multi-agent crypto trading system with hard risk limits.", no_args_is_help=True)
data_app = typer.Typer(help="Candle data management.")
app.add_typer(data_app, name="data")


def _parse_date(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=UTC)


def _symbols(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


@data_app.command("sync")
def data_sync(
    symbols: str = typer.Option(DEFAULT_SYMBOLS, help="Comma-separated symbols"),
    tf: str = typer.Option("1h"),
    since: str = typer.Option("2023-01-01"),
    exchange: str = typer.Option("binance"),
    data_root: str = typer.Option("data"),
) -> None:
    """Fetch missing candles into the local cache."""
    feed = DataFeed(exchange_id=exchange, root=Path(data_root))
    start = _parse_date(since)
    for sym in _symbols(symbols):
        try:
            added = feed.sync(sym, tf, start)
        except DataFeedError as e:
            typer.secho(f"{sym:<10} FAILED: {e}", fg=typer.colors.RED)
            raise typer.Exit(code=1) from e
        typer.echo(f"{sym:<10} +{added} rows")


def _side_by_side(system: Metrics, bench: Metrics) -> str:
    left = to_table(system, "system").splitlines()
    right = to_table(bench, "benchmark").splitlines()
    return "\n".join(f"{a:<32}{b}" for a, b in zip(left, right))


def _monthly_table(system: pd.Series, bench: pd.Series) -> str:
    df = pd.DataFrame({"system": system, "benchmark": bench}).dropna(how="all")
    lines = [f"{'month':<10}{'system':>10}{'benchmark':>12}", "-" * 32]
    for ts, row in df.iterrows():
        s = "" if math.isnan(row.system) else f"{row.system:+.1%}"
        b = "" if math.isnan(row.benchmark) else f"{row.benchmark:+.1%}"
        lines.append(f"{ts.strftime('%Y-%m'):<10}{s:>10}{b:>12}")
    lines.append(TARGET_LINE)
    return "\n".join(lines)


@app.command("backtest")
def backtest(
    from_: str = typer.Option("2023-01-01", "--from"),
    to: str = typer.Option(datetime.now(UTC).strftime("%Y-%m-%d"), "--to"),
    symbols: str = typer.Option(DEFAULT_SYMBOLS),
    tf: str = typer.Option("1h"),
    equity: float = typer.Option(500.0),
    walk_forward_: bool = typer.Option(True, "--walk-forward/--no-walk-forward"),
    threshold: float = typer.Option(0.6, help="Allocator threshold (ignored with walk-forward)"),
    warmup: int = typer.Option(DEFAULT_WARMUP),
    exchange: str = typer.Option("binance"),
    data_root: str = typer.Option("data"),
) -> None:
    """Run a backtest against BTC buy-and-hold and print honest numbers."""
    feed = DataFeed(exchange_id=exchange, root=Path(data_root))
    start, end = _parse_date(from_), _parse_date(to)
    syms = _symbols(symbols)
    try:
        candles = {s: feed.get_candles(s, tf) for s in syms}
    except DataFeedError as e:
        typer.secho(str(e), fg=typer.colors.RED)
        raise typer.Exit(code=1) from e

    if walk_forward_:
        try:
            wf = walk_forward(candles, start, end, initial_equity=equity, tf=tf, warmup=warmup)
        except ValueError as e:
            typer.secho(str(e), fg=typer.colors.RED)
            raise typer.Exit(code=1) from e
        system, bench = wf.metrics, wf.benchmark
        typer.echo(f"walk-forward: {len(wf.windows)} test windows, {len(wf.trades)} trades")
        for w in wf.windows:
            typer.echo(
                f"  {w['test_start']:%Y-%m-%d} -> {w['test_end']:%Y-%m-%d}  "
                f"threshold={w['chosen'].threshold}  "
                f"test_return={w['test_metrics'].total_return:+.2%}"
            )
    else:
        sliced = slice_with_warmup(candles, start, end, warmup)
        try:
            res = run_backtest(sliced, Params(threshold=threshold), initial_equity=equity, tf=tf,
                               warmup=warmup)
        except ValueError as e:
            typer.secho(str(e), fg=typer.colors.RED)
            raise typer.Exit(code=1) from e
        system, bench = res.metrics, res.benchmark

    typer.echo(_side_by_side(system, bench))
    typer.echo()
    typer.echo(_monthly_table(system.monthly_returns, bench.monthly_returns))
    if system.sharpe < bench.sharpe:
        typer.secho(
            "\nWARNING: system Sharpe is below BTC buy-and-hold. The system is not adding value "
            "over just buying and holding on this data.",
            fg=typer.colors.RED,
        )


if __name__ == "__main__":
    app()
