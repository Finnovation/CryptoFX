"""Candle data feed backed by ccxt with a local Parquet cache."""

import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import pandas as pd

from cryptofx.data import cache

FetchFn = Callable[[str, str, int, int], list[list]]
PAGE_LIMIT = 1000
TF_MS = {"1h": 3_600_000, "4h": 4 * 3_600_000}


class DataFeedError(Exception):
    pass


class DataFeed:
    def __init__(
        self,
        exchange_id: str = "binance",
        root: Path = Path("data"),
        fetch_fn: FetchFn | None = None,
        retries: int = 3,
        backoff_s: float = 1.0,
    ) -> None:
        self.exchange_id = exchange_id
        self.root = Path(root)
        self.retries = retries
        self.backoff_s = backoff_s
        self._fetch = fetch_fn or self._ccxt_fetch_fn()

    def _ccxt_fetch_fn(self) -> FetchFn:
        import ccxt

        ex = getattr(ccxt, self.exchange_id)({"enableRateLimit": True})

        def fetch(symbol: str, tf: str, since_ms: int, limit: int) -> list[list]:
            return ex.fetch_ohlcv(symbol, tf, since=since_ms, limit=limit)

        return fetch

    def _fetch_page(self, symbol: str, tf: str, since_ms: int) -> list[list]:
        last_err: Exception | None = None
        for attempt in range(self.retries):
            try:
                return self._fetch(symbol, tf, since_ms, PAGE_LIMIT)
            except Exception as e:  # noqa: BLE001 - any transport error is retried
                last_err = e
                time.sleep(self.backoff_s * 2**attempt)
        raise DataFeedError(f"fetch failed for {symbol} {tf} after {self.retries} tries: {last_err}")

    def sync(self, symbol: str, tf: str, since: datetime) -> int:
        if tf not in TF_MS:
            raise DataFeedError(f"unsupported timeframe {tf!r}")
        path = cache.cache_path(self.root, self.exchange_id, symbol, tf)
        existing = cache.load(path)
        since_ms = int(since.timestamp() * 1000)
        if existing is not None and len(existing):
            since_ms = max(since_ms, int(existing.ts.iloc[-1].timestamp() * 1000) + TF_MS[tf])
        new_rows: list[list] = []
        while True:
            page = self._fetch_page(symbol, tf, since_ms)
            if not page:
                break
            new_rows.extend(page)
            next_ms = page[-1][0] + TF_MS[tf]
            if len(page) < PAGE_LIMIT or next_ms <= since_ms:
                break
            since_ms = next_ms
        if not new_rows:
            return 0
        new = pd.DataFrame(new_rows, columns=cache.CANDLE_COLUMNS)
        new["ts"] = pd.to_datetime(new["ts"], unit="ms", utc=True)
        merged = cache.merge(existing, new)
        cache.save(merged, path)
        return len(merged) - (len(existing) if existing is not None else 0)

    def get_candles(
        self,
        symbol: str,
        tf: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        path = cache.cache_path(self.root, self.exchange_id, symbol, tf)
        df = cache.load(path)
        if df is None or df.empty:
            raise DataFeedError(f"no cached candles for {symbol} {tf}; run data sync first")
        if start is not None:
            df = df[df.ts >= pd.Timestamp(start)]
        if end is not None:
            df = df[df.ts <= pd.Timestamp(end)]
        return cache.fill_gaps(df.reset_index(drop=True), tf)

    @staticmethod
    def max_gap_bars(df: pd.DataFrame) -> int:
        if "gap" not in df.columns or df.empty:
            return 0
        best = run = 0
        for g in df["gap"].to_numpy():
            run = run + 1 if g else 0
            best = max(best, run)
        return best
