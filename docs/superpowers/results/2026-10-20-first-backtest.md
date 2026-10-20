# First backtest — 2026-10-20

## Status: NOT RUN (data fetch blocked)

The cloud session's network policy refused connections to every exchange
public API tried (`api.binance.com`, `data-api.binance.vision`,
`api.kraken.com`, `www.okx.com`), so `cryptofx data sync` could not fetch
candles and no real backtest was produced. No numbers are recorded here on
purpose: nothing in this file is to be made up.

## To run it

On a machine with internet access, or after adding `api.binance.com` (and
optionally `data-api.binance.vision`) to the environment's allowed domains:

```bash
. .venv/bin/activate
cryptofx data sync --since 2023-01-01
cryptofx backtest --from 2023-01-01 --to 2026-10-01 --walk-forward
```

Expected runtime: a 1h, five-symbol walk-forward over three years is on the
order of 30-60 minutes (about 4ms per bar per symbol, roughly ten passes over
the data). Use `--tf 4h` for a first look in a few minutes.

Paste the two tables and the monthly block below verbatim, whatever they say.

## Results

_pending_
