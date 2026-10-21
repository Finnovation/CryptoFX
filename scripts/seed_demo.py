"""Seed a demo trade log and candle cache so the dashboard can be seen without
waiting for live bars. Synthetic data only. Usage:

    python scripts/seed_demo.py /tmp/demo
    cryptofx web --db /tmp/demo/cryptofx.db --data-root /tmp/demo --symbols BTC/USDT,ETH/USDT
"""

import sys
from pathlib import Path

from cryptofx.backtest.runner import Params
from cryptofx.data.feed import DataFeed
from cryptofx.paper.trader import PaperTrader
from cryptofx.store.db import Store
from tests.synth import uptrend

root = Path(sys.argv[1])
root.mkdir(parents=True, exist_ok=True)
syms = ["BTC/USDT", "ETH/USDT"]
frames = {"BTC/USDT": uptrend(n=700, start=60000.0, step=20.0, noise=60.0),
          "ETH/USDT": uptrend(n=700, start=3000.0, step=1.0, noise=3.0)}
visible = {"n": 300}  # the fake exchange publishes bars as this grows; sync fills the cache


def fetch(symbol, tf, since_ms, limit):
    df = frames[symbol].iloc[: visible["n"]]
    return [[int(r.ts.timestamp() * 1000), r.open, r.high, r.low, r.close, r.volume]
            for r in df.itertuples() if int(r.ts.timestamp() * 1000) >= since_ms]


def now():
    return (frames["BTC/USDT"].ts.iloc[visible["n"] - 1] + __import__("pandas").Timedelta(hours=1, minutes=5)).to_pydatetime()


feed = DataFeed(root=root, fetch_fn=fetch, backoff_s=0, now_fn=lambda: int(now().timestamp() * 1000))
store = Store(root / "cryptofx.db")
trader = PaperTrader(feed, store, syms, "1h", Params(), 500.0, warmup=200, now_fn=now, log=lambda m: None)
trader.bootstrap()
for _ in range(380):
    trader.run_once()
    visible["n"] += 1
print(f"seeded {root}: {len(store.equity_curve())} bars, {store.trade_stats()['count']} trades, "
      f"{len(store.positions())} open")
