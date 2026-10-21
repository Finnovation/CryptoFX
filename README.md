# CryptoFX

Multi-agent crypto trading system with hard risk limits, built to prove or
disprove an edge on paper before a cent goes live.

## Read this first

This is not a money printer. Retail algorithmic trading mostly loses or breaks
even after fees. The point of this project is an honest pipeline: backtest,
paper trading and live trading share one execution path, and the reports say
plainly whether the system beats simply buying and holding BTC.

Live trading is locked behind a go-live gate (minimum 42 days of paper
trading, 50+ trades, profit factor above 1.3, drawdown under 10%). There is no
bypass flag. Risk limits live in `cryptofx/risk/limits.py` as code, not config.

Design spec: `docs/superpowers/specs/2026-10-20-crypto-trading-system-design.md`

## Install

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

## Fetch data

Public Binance candles, no account needed. Cached to `data/` as Parquet.

```bash
cryptofx data sync --since 2023-01-01            # 1h, five majors
cryptofx data sync --tf 4h --since 2022-01-01
```

## Backtest

```bash
cryptofx backtest --from 2023-01-01 --to 2026-10-01 --walk-forward
cryptofx backtest --from 2024-01-01 --to 2024-12-31 --no-walk-forward --threshold 0.6
```

Walk-forward picks the allocator threshold on a 180-day train window and
reports only the following 60-day test window, rolled forward. The output
shows the system next to BTC buy-and-hold, monthly returns against a
`target 7-9% / realistic 0-5%` reference line, and a red warning when the
system's Sharpe is below buy-and-hold.

## Paper trade and watch it

Two processes share one SQLite file. Start the trader, then the dashboard.

```bash
cryptofx paper run                      # resumes from cryptofx.db if it exists
cryptofx paper run --once               # one bar then exit (cron-friendly)
cryptofx web                            # http://127.0.0.1:8000
cryptofx gate                           # go-live checklist, exit 1 while locked
cryptofx risk reset --confirm --reason "reviewed the halt, resuming"
```

The dashboard shows wallet, equity curve, a candlestick chart with the
system's overlays and trade markers, what every agent is voting right now,
open positions, trade history with win/loss stats, risk events and the
go-live checklist. It polls every 10 seconds and works on a phone.

**Phone access.** The server binds to localhost on purpose. Use Tailscale
(install it on the box and your phone, then open `http://<tailscale-ip>:8000`)
or an SSH tunnel (`ssh -L 8000:127.0.0.1:8000 user@box`). If you must bind
`--host 0.0.0.0`, set `CRYPTOFX_TOKEN=<long random string>` first; the page
will ask for it once and remember it.

**Keep it running.** A minimal systemd unit:

```ini
[Unit]
Description=CryptoFX paper trader
After=network-online.target
[Service]
WorkingDirectory=/home/you/CryptoFX
ExecStart=/home/you/CryptoFX/.venv/bin/cryptofx paper run
Restart=always
RestartSec=30
[Install]
WantedBy=default.target
```

If the trader crashes it halts itself before exiting; on restart it stays
halted until you run `cryptofx risk reset`. That is deliberate.

## Architecture

```
DataFeed -> RegimeAgent + {Trend, MeanRev, Breakout} -> Allocator -> RiskManager -> Executor
```

- Agents are pure functions of candles. No LLMs in the signal path.
- The allocator needs weighted agreement (score >= threshold) to size a trade.
- The risk manager can pause (3% daily loss) or halt (10% drawdown) trading
  and vetoes every buy that breaks a limit. Sells are never blocked.
- Executors: backtest and paper share one fill model; live (Kraken) is Plan 3,
  locked behind the gate.

## Status

Plan 1 (backtest core) is complete and reviewed. The first real backtest is
recorded as NOT RUN in `docs/superpowers/results/` because the build
environment could not reach any exchange API; run the two commands above on a
machine that can. Plan 2 adds the SQLite trade log, paper
trading loop, HTML report, go-live gate command and the Kraken live executor.
