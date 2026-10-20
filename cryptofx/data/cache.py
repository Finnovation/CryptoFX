"""Parquet candle cache helpers."""

import os
import tempfile
from pathlib import Path

import pandas as pd

CANDLE_COLUMNS = ["ts", "open", "high", "low", "close", "volume"]
TF_TO_PANDAS = {"1h": "1h", "4h": "4h"}


def cache_path(root: Path, exchange: str, symbol: str, tf: str) -> Path:
    return Path(root) / exchange / symbol.replace("/", "-") / f"{tf}.parquet"


def load(path: Path) -> pd.DataFrame | None:
    if not Path(path).exists():
        return None
    return pd.read_parquet(path)


def save(df: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    try:
        df.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def merge(existing: pd.DataFrame | None, new: pd.DataFrame) -> pd.DataFrame:
    frames = [f for f in (existing, new) if f is not None and len(f)]
    if not frames:
        return pd.DataFrame(columns=CANDLE_COLUMNS)
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset="ts", keep="last").sort_values("ts")
    return out[CANDLE_COLUMNS].reset_index(drop=True)


def fill_gaps(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    if tf not in TF_TO_PANDAS:
        raise ValueError(f"unsupported timeframe {tf!r}")
    if df.empty:
        out = df.copy()
        out["gap"] = pd.Series(dtype=bool)
        return out
    freq = TF_TO_PANDAS[tf]
    full = pd.date_range(df.ts.iloc[0], df.ts.iloc[-1], freq=freq, tz="UTC")
    out = df.set_index("ts").reindex(full)
    out["gap"] = out["close"].isna()
    prev_close = out["close"].ffill()
    for col in ("open", "high", "low", "close"):
        out[col] = out[col].where(~out["gap"], prev_close)
    out["volume"] = out["volume"].fillna(0.0)
    out.index.name = "ts"
    return out.reset_index()[CANDLE_COLUMNS + ["gap"]]
