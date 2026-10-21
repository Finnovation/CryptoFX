from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from cryptofx.core.models import Direction, Fill, Position, Side, Signal, Trade
from cryptofx.data import cache
from cryptofx.data.feed import DataFeed
from cryptofx.risk.manager import RiskManager
from cryptofx.store.db import Store
from cryptofx.web.app import create_app
from tests.synth import uptrend

SYMS = ["BTC/USDT", "ETH/USDT"]


def seeded(tmp_path, token=None):
    store = Store(tmp_path / "w.db")
    df = uptrend(n=400, noise=1.0)
    for s in SYMS:
        cache.save(df[cache.CANDLE_COLUMNS], cache.cache_path(tmp_path, "binance", s, "1h"))
    feed = DataFeed(root=tmp_path, fetch_fn=lambda *a: [])
    store.start_run("paper", SYMS, "1h", {"threshold": 0.6}, 500.0)
    t_fill = df.ts.iloc[300].to_pydatetime()
    t_stop = df.ts.iloc[320].to_pydatetime()
    for i in range(250, 400):
        store.append_equity(df.ts.iloc[i].to_pydatetime(), 500.0 + i - 250, 400.0, 1, {"BTC/USDT": "trending"})
    store.append_fills([Fill(symbol="BTC/USDT", side=Side.BUY, size_base=0.01, price=350.0, fee=0.1, ts=t_fill)])
    store.append_trades([
        Trade(symbol="BTC/USDT", entry_ts=t_fill, exit_ts=t_stop, entry_price=350.0, exit_price=340.0,
              size_base=0.01, pnl=-0.1, fees=0.2, exit_reason="stop"),
        Trade(symbol="ETH/USDT", entry_ts=t_fill, exit_ts=t_stop, entry_price=10.0, exit_price=12.0,
              size_base=1.0, pnl=2.0, fees=0.02, exit_reason="signal exit"),
    ])
    store.replace_positions([Position(symbol="BTC/USDT", size_base=0.01, entry_price=350.0, stop_price=340.0,
                                      opened_at=t_fill, initial_risk=10.0)])
    store.append_signals(df.ts.iloc[-1].to_pydatetime(), "BTC/USDT", "trending",
                         {"trend": Signal(direction=Direction.LONG, confidence=0.9, reason="up")})
    store.append_risk_events([{"ts": t_stop, "kind": "paused", "detail": "daily loss"}])
    rm = RiskManager()
    rm.update_equity(df.ts.iloc[250].to_pydatetime(), 500.0)
    store.save_state("risk", rm.to_state())
    app = create_app(store, feed, "1h", SYMS, token=token)
    return TestClient(app), store, df, t_fill, t_stop


def test_status_empty_db_is_well_formed(tmp_path):
    client = TestClient(create_app(Store(":memory:"), DataFeed(root=tmp_path, fetch_fn=lambda *a: []), "1h", SYMS))
    r = client.get("/api/status")
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] is None and body["equity"] is None and body["risk_state"] == "active"
    assert body["drawdown"] == 0.0 and body["total_return"] is None
    assert len(body["gate"]) == 6 and not any(c["passed"] for c in body["gate"])
    for route in ("/api/equity", "/api/trades", "/api/positions", "/api/signals", "/api/risk-events"):
        assert client.get(route).status_code == 200, route
    assert client.get("/api/trades").json()["stats"]["count"] == 0
    assert client.get("/api/symbols").json() == {"symbols": SYMS, "tf": "1h"}


def test_status_after_seed_has_equity_and_gate(tmp_path):
    client, _store, _df, *_ = seeded(tmp_path)
    body = client.get("/api/status").json()
    assert body["mode"] == "paper" and body["equity"] == 649.0 and body["cash"] == 400.0
    assert body["initial_equity"] == 500.0 and body["total_return"] == pytest.approx(649 / 500 - 1)
    assert body["risk_state"] == "active" and body["peak_equity"] == 500.0
    assert body["n_positions"] == 1 and body["last_ts"].startswith("2024-01-17")
    assert isinstance(body["gate"], list) and body["gate"][0]["name"] == "paper_days"


def test_equity_trades_positions_signals_risk_events_routes(tmp_path):
    client, _store, df, _t_fill, _t_stop = seeded(tmp_path)
    eq = client.get("/api/equity?limit=10").json()
    assert len(eq) == 10 and eq[-1]["equity"] == 649.0
    tr = client.get("/api/trades").json()
    assert tr["stats"]["count"] == 2 and tr["stats"]["win_rate"] == 0.5
    assert tr["trades"][0]["symbol"] in SYMS and "exit_reason" in tr["trades"][0]
    pos = client.get("/api/positions").json()
    assert pos[0]["symbol"] == "BTC/USDT" and pos[0]["mark"] == pytest.approx(df.close.iloc[-1])
    assert pos[0]["unrealised_pnl"] == pytest.approx((df.close.iloc[-1] - 350.0) * 0.01)
    sig = client.get("/api/signals").json()
    assert sig["BTC/USDT"]["regime"] == "trending" and sig["BTC/USDT"]["agents"][0]["agent"] == "trend"
    ev = client.get("/api/risk-events").json()
    assert ev[0]["kind"] == "paused"


def test_candles_payload_shape_and_marker_alignment(tmp_path):
    client, _store, df, t_fill, t_stop = seeded(tmp_path)
    r = client.get("/api/candles?symbol=BTC/USDT&limit=150")
    assert r.status_code == 200
    body = r.json()
    c = body["candles"]
    assert len(c) == 150 and set(c[0]) == {"time", "open", "high", "low", "close", "volume"}
    assert c[-1]["time"] == int(df.ts.iloc[-1].timestamp())
    for key in ("ema20", "ema50", "bb_upper", "bb_lower", "donchian_upper", "donchian_lower"):
        assert len(body["overlays"][key]) == 150 and body["overlays"][key][-1]["value"] is not None
    times = {x["time"] for x in c}
    buy = next(m for m in body["markers"] if m["text"] == "buy")
    stop = next(m for m in body["markers"] if m["text"] == "stop")
    assert buy["time"] == int(t_fill.timestamp()) and buy["time"] in times
    assert stop["time"] == int(t_stop.timestamp()) and stop["position"] == "aboveBar"
    assert body["stop_price"] == 340.0


def test_candles_unknown_symbol_404_json(tmp_path):
    client, *_ = seeded(tmp_path)
    r = client.get("/api/candles?symbol=DOGE/USDT")
    assert r.status_code == 404
    assert r.json()["error"].startswith("no cached candles for DOGE/USDT 1h")


def test_inf_profit_factor_serialises(tmp_path):
    store = Store(":memory:")
    store.append_trades([Trade(symbol="BTC/USDT", entry_ts=datetime(2024, 1, 1, tzinfo=UTC),
                               exit_ts=datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=1), entry_price=1.0,
                               exit_price=2.0, size_base=1.0, pnl=1.0, fees=0.0, exit_reason="x")])
    client = TestClient(create_app(store, DataFeed(root=tmp_path, fetch_fn=lambda *a: []), "1h", SYMS))
    r = client.get("/api/trades")
    assert r.status_code == 200 and r.json()["stats"]["profit_factor"] is None


def test_token_enforced_only_when_configured(tmp_path):
    client, *_ = seeded(tmp_path)
    assert client.get("/api/status").status_code == 200
    secured, *_ = seeded(tmp_path, token="s3cret")
    assert secured.get("/api/status").status_code == 401
    assert secured.get("/api/status").json() == {"error": "unauthorized"}
    assert secured.get("/api/status", headers={"Authorization": "Bearer s3cret"}).status_code == 200
    assert secured.get("/").status_code == 200


def test_index_served(tmp_path):
    client, *_ = seeded(tmp_path)
    r = client.get("/")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    assert 'id="chart"' in r.text and "PAPER" in r.text
    assert "lightweight-charts@4.2.0" in r.text
    lib = client.get("/static/vendor/lightweight-charts.standalone.production.js")
    assert lib.status_code == 200 and "Lightweight Charts" in lib.text[:300]
    for el in ("chart", "agents", "trades", "positions", "stats", "gate", "wallet", "equity"):
        assert f'id="{el}"' in r.text, el
