from datetime import UTC, datetime

import pandas as pd
from typer.testing import CliRunner

from cryptofx import cli
from cryptofx.data import cache
from cryptofx.data.feed import DataFeed
from tests.synth import uptrend

runner = CliRunner()
H = 3_600_000


def seed(tmp_path, df):
    cache.save(df[cache.CANDLE_COLUMNS], cache.cache_path(tmp_path, "binance", "BTC/USDT", "1h"))


def test_backtest_command_prints_metrics(tmp_path):
    seed(tmp_path, uptrend(n=600, noise=2.0))
    res = runner.invoke(
        cli.app,
        ["backtest", "--from", "2024-01-05", "--to", "2024-01-25", "--symbols", "BTC/USDT",
         "--no-walk-forward", "--data-root", str(tmp_path), "--warmup", "90"],
    )
    assert res.exit_code == 0, res.output
    assert "system" in res.output and "benchmark" in res.output
    assert "max_drawdown" in res.output and "profit_factor" in res.output
    assert "target 7-9%" in res.output and "realistic 0-5%" in res.output


def test_backtest_warns_when_below_benchmark(tmp_path):
    # Uptrend with a 4-bar hole every 50 bars: the gap guard keeps the system flat
    # (Sharpe 0) while buy-and-hold on the filled series rises.
    df = uptrend(n=600)
    keep = [i for i in range(600) if i % 50 >= 4]
    seed(tmp_path, df.iloc[keep])
    res = runner.invoke(
        cli.app,
        ["backtest", "--from", "2024-01-05", "--to", "2024-01-25", "--symbols", "BTC/USDT",
         "--no-walk-forward", "--data-root", str(tmp_path), "--warmup", "90"],
    )
    assert res.exit_code == 0, res.output
    assert "trade_count                  0" in res.output
    assert "WARNING" in res.output and "buy-and-hold" in res.output


def test_backtest_without_data_fails_cleanly(tmp_path):
    res = runner.invoke(
        cli.app, ["backtest", "--symbols", "BTC/USDT", "--no-walk-forward", "--data-root", str(tmp_path)]
    )
    assert res.exit_code != 0
    assert "data sync" in res.output


def test_data_sync_reports_rows(tmp_path, monkeypatch):
    def fetch(symbol, tf, since_ms, limit):
        return [[since_ms + i * H, 1, 2, 0.5, 1.5, 10.0] for i in range(5)] if since_ms < 5 * H else []

    monkeypatch.setattr(
        cli, "DataFeed", lambda **kw: DataFeed(root=kw["root"], fetch_fn=fetch, backoff_s=0)
    )
    res = runner.invoke(
        cli.app,
        ["data", "sync", "--symbols", "BTC/USDT,ETH/USDT", "--since", "1970-01-01",
         "--data-root", str(tmp_path)],
    )
    assert res.exit_code == 0, res.output
    assert "BTC/USDT" in res.output and "ETH/USDT" in res.output and "5" in res.output
    df = pd.read_parquet(cache.cache_path(tmp_path, "binance", "ETH/USDT", "1h"))
    assert len(df) == 5 and df.ts.iloc[0] == pd.Timestamp(datetime(1970, 1, 1, tzinfo=UTC))
