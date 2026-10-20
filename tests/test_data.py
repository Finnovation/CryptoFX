from datetime import UTC, datetime

import pandas as pd
import pytest

from cryptofx.data.cache import CANDLE_COLUMNS, cache_path, fill_gaps, merge
from cryptofx.data.feed import DataFeed, DataFeedError

H = 3_600_000


def _rows(start_ms, n, step_ms=H):
    return [[start_ms + i * step_ms, 100 + i, 101 + i, 99 + i, 100.5 + i, 10.0] for i in range(n)]


def to_df(rows):
    df = pd.DataFrame(rows, columns=CANDLE_COLUMNS)
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df


def test_cache_path_replaces_slash(tmp_path):
    assert cache_path(tmp_path, "binance", "BTC/USDT", "1h") == tmp_path / "binance" / "BTC-USDT" / "1h.parquet"


def test_merge_sorts_and_dedups():
    a = to_df(_rows(2 * H, 3))  # hours 2,3,4
    b = to_df(_rows(0, 3))  # hours 0,1,2
    b.loc[2, "close"] = 999.0  # overlapping hour 2, new wins
    out = merge(a, b)
    assert out.ts.is_monotonic_increasing and out.ts.is_unique and len(out) == 5
    assert out.loc[out.ts == pd.Timestamp(2 * H, unit="ms", tz="UTC"), "close"].item() == 999.0


def test_merge_with_none_existing():
    assert len(merge(None, to_df(_rows(0, 2)))) == 2


def test_fill_gaps_marks_missing_hour():
    df = to_df(_rows(0, 2) + _rows(3 * H, 1))  # hours 0,1,3
    out = fill_gaps(df, "1h")
    assert len(out) == 4
    assert out.gap.tolist() == [False, False, True, False]
    assert out.loc[2, "close"] == out.loc[1, "close"]
    assert out.loc[2, "open"] == out.loc[1, "close"]
    assert out.loc[2, "volume"] == 0


def test_sync_pages_and_caches(tmp_path):
    calls = []

    def fetch(symbol, tf, since_ms, limit):
        calls.append(since_ms)
        return _rows(since_ms, 1000) if since_ms < 1000 * H else []

    feed = DataFeed(root=tmp_path, fetch_fn=fetch, backoff_s=0)
    added = feed.sync("BTC/USDT", "1h", datetime(1970, 1, 1, tzinfo=UTC))
    assert added == 1000
    assert cache_path(tmp_path, "binance", "BTC/USDT", "1h").exists()
    assert feed.sync("BTC/USDT", "1h", datetime(1970, 1, 1, tzinfo=UTC)) == 0


def test_sync_failure_leaves_cache_untouched(tmp_path):
    def boom(*a):
        raise RuntimeError("net")

    feed = DataFeed(root=tmp_path, fetch_fn=boom, backoff_s=0)
    with pytest.raises(DataFeedError):
        feed.sync("BTC/USDT", "1h", datetime(2024, 1, 1, tzinfo=UTC))
    assert not cache_path(tmp_path, "binance", "BTC/USDT", "1h").exists()


def test_get_candles_slices_and_fills(tmp_path):
    rows = _rows(0, 5) + _rows(6 * H, 4)  # hours 0-4, 6-9 ; hour 5 missing

    def fetch(symbol, tf, since_ms, limit):
        return [r for r in rows if r[0] >= since_ms]

    feed = DataFeed(root=tmp_path, fetch_fn=fetch, backoff_s=0)
    feed.sync("BTC/USDT", "1h", datetime(1970, 1, 1, tzinfo=UTC))
    df = feed.get_candles(
        "BTC/USDT", "1h", start=datetime(1970, 1, 1, 2, tzinfo=UTC), end=datetime(1970, 1, 1, 7, tzinfo=UTC)
    )
    assert df.ts.iloc[0] == pd.Timestamp(2 * H, unit="ms", tz="UTC")
    assert df.ts.iloc[-1] == pd.Timestamp(7 * H, unit="ms", tz="UTC")
    assert len(df) == 6 and df.gap.sum() == 1


def test_get_candles_without_cache_raises(tmp_path):
    with pytest.raises(DataFeedError):
        DataFeed(root=tmp_path, fetch_fn=lambda *a: []).get_candles("BTC/USDT", "1h")


def test_max_gap_bars():
    df = pd.DataFrame({"gap": [False, True, True, False, True, True, True, False]})
    assert DataFeed.max_gap_bars(df) == 3
    assert DataFeed.max_gap_bars(pd.DataFrame({"close": [1.0]})) == 0
