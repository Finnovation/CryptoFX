"""Read-only JSON API over the trade log, plus the dashboard page."""

import math
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse

from cryptofx.agents.indicators import bollinger, donchian, ema
from cryptofx.data.feed import DataFeed, DataFeedError
from cryptofx.risk.gate import GoLiveGate
from cryptofx.risk.manager import RiskManager
from cryptofx.store.db import Store

STATIC = Path(__file__).with_name("static")
OVERLAY_WARMUP = 60


def _clean(obj: Any) -> Any:
    """Make a payload strict-JSON safe: inf/nan -> None, datetimes -> ISO."""
    if isinstance(obj, float):
        return None if math.isnan(obj) or math.isinf(obj) else obj
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_clean(v) for v in obj]
    return obj


def create_app(
    store: Store, feed: DataFeed, tf: str, symbols: list[str], token: str | None = None
) -> FastAPI:
    app = FastAPI(title="CryptoFX", docs_url=None, redoc_url=None)

    async def require_token(request: Request) -> None:
        if token is None:
            return
        header = request.headers.get("authorization", "")
        if header != f"Bearer {token}":
            raise HTTPException(status_code=401, detail="unauthorized")

    @app.exception_handler(HTTPException)
    async def http_error(_: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse({"error": exc.detail}, status_code=exc.status_code)

    def last_close(symbol: str) -> float | None:
        try:
            return float(feed.get_candles(symbol, tf).close.iloc[-1])
        except DataFeedError:
            return None

    def risk_view() -> dict:
        state = store.load_state("risk")
        if state is None:
            return {"risk_state": "active", "paused_until": None, "peak_equity": None}
        rm = RiskManager.from_state(state)
        return {
            "risk_state": rm.state.value,
            "paused_until": rm.paused_until,
            "peak_equity": rm.peak_equity,
        }

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC / "index.html", media_type="text/html")

    @app.get("/api/symbols", dependencies=[Depends(require_token)])
    async def api_symbols() -> dict:
        return {"symbols": symbols, "tf": tf}

    @app.get("/api/status", dependencies=[Depends(require_token)])
    async def api_status() -> JSONResponse:
        st = store.status()
        rv = risk_view()
        equity = st["equity"]
        peak = rv["peak_equity"]
        drawdown = (peak - equity) / peak if equity is not None and peak else 0.0
        initial = st["initial_equity"]
        total_return = equity / initial - 1 if equity is not None and initial else None
        body = {
            **st,
            **rv,
            "drawdown": drawdown,
            "total_return": total_return,
            "gate": [c.__dict__ for c in GoLiveGate(store).evaluate()],
        }
        return JSONResponse(_clean(body))

    @app.get("/api/equity", dependencies=[Depends(require_token)])
    async def api_equity(limit: int = 5000) -> JSONResponse:
        return JSONResponse(_clean(store.equity_curve(limit=limit)))

    @app.get("/api/trades", dependencies=[Depends(require_token)])
    async def api_trades(limit: int = 500) -> JSONResponse:
        return JSONResponse(_clean({"trades": store.trades(limit=limit), "stats": store.trade_stats()}))

    @app.get("/api/positions", dependencies=[Depends(require_token)])
    async def api_positions() -> JSONResponse:
        out = []
        for p in store.positions():
            mark = last_close(p["symbol"])
            pnl = (mark - p["entry_price"]) * p["size_base"] if mark is not None else None
            pct = mark / p["entry_price"] - 1 if mark is not None and p["entry_price"] else None
            out.append({**p, "mark": mark, "unrealised_pnl": pnl, "unrealised_pct": pct})
        return JSONResponse(_clean(out))

    @app.get("/api/signals", dependencies=[Depends(require_token)])
    async def api_signals() -> JSONResponse:
        grouped: dict[str, dict] = {}
        for row in store.latest_signals():
            g = grouped.setdefault(row["symbol"], {"ts": row["ts"], "regime": row["regime"], "agents": []})
            g["agents"].append(
                {"agent": row["agent"], "direction": row["direction"], "confidence": row["confidence"],
                 "reason": row["reason"]}
            )
        return JSONResponse(_clean(grouped))

    @app.get("/api/risk-events", dependencies=[Depends(require_token)])
    async def api_risk_events(limit: int = 100) -> JSONResponse:
        return JSONResponse(_clean(store.risk_events(limit=limit)))

    @app.get("/api/candles", dependencies=[Depends(require_token)])
    async def api_candles(symbol: str, limit: int = 300) -> JSONResponse:
        try:
            df = feed.get_candles(symbol, tf)
        except DataFeedError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        df = df.tail(limit + OVERLAY_WARMUP).reset_index(drop=True)
        e20, e50 = ema(df.close, 20), ema(df.close, 50)
        _, bb_up, bb_lo = bollinger(df.close, 20, 2.0)
        d_up, d_lo = donchian(df.high, df.low, 20)
        df = df.tail(limit)
        times = [int(t.timestamp()) for t in df.ts]

        def series(s: pd.Series) -> list[dict]:
            vals = s.loc[df.index].tolist()
            return [{"time": t, "value": None if pd.isna(v) else float(v)} for t, v in zip(times, vals)]

        candles = [
            {"time": t, "open": float(r.open), "high": float(r.high), "low": float(r.low),
             "close": float(r.close), "volume": float(r.volume)}
            for t, r in zip(times, df.itertuples())
        ]
        first_ts = df.ts.iloc[0].to_pydatetime()
        markers = []
        for f in store.fills(symbol=symbol, since=first_ts):
            t = int(f["ts"].timestamp())
            if f["side"] == "buy":
                markers.append({"time": t, "position": "belowBar", "shape": "arrowUp",
                                "color": "#22c55e", "text": "buy"})
            else:
                markers.append({"time": t, "position": "aboveBar", "shape": "arrowDown",
                                "color": "#ef4444", "text": "sell"})
        for tr in store.trades(limit=10_000):
            if tr["symbol"] == symbol and tr["exit_reason"] == "stop" and tr["exit_ts"] >= first_ts:
                markers.append({"time": int(tr["exit_ts"].timestamp()), "position": "aboveBar",
                                "shape": "circle", "color": "#f59e0b", "text": "stop"})
        markers.sort(key=lambda m: m["time"])
        stop_price = next((p["stop_price"] for p in store.positions() if p["symbol"] == symbol), None)
        return JSONResponse(
            _clean({
                "symbol": symbol,
                "tf": tf,
                "candles": candles,
                "overlays": {
                    "ema20": series(e20), "ema50": series(e50),
                    "bb_upper": series(bb_up), "bb_lower": series(bb_lo),
                    "donchian_upper": series(d_up), "donchian_lower": series(d_lo),
                },
                "markers": markers,
                "stop_price": stop_price,
            })
        )

    return app
