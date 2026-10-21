"""SQLite trade log. One writer (the trader), many readers (the web API)."""

import json
import math
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from cryptofx.core.models import Fill, Order, Position, Signal, Trade

SCHEMA = Path(__file__).with_name("schema.sql").read_text()


def _iso(ts: datetime) -> str:
    return ts.isoformat()


def _dt(s: str | None) -> datetime | None:
    return datetime.fromisoformat(s) if s else None


class Store:
    def __init__(self, path: Path | str) -> None:
        self.path = str(path)
        self._conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        if self.path != ":memory:":
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(SCHEMA)
        self._in_tx = False

    # -- transactions ---------------------------------------------------
    @contextmanager
    def transaction(self):
        if self._in_tx:
            yield
            return
        self._conn.execute("BEGIN")
        self._in_tx = True
        try:
            yield
            self._conn.execute("COMMIT")
        except BaseException:
            self._conn.execute("ROLLBACK")
            raise
        finally:
            self._in_tx = False

    # -- writes ---------------------------------------------------------
    def start_run(
        self, mode: str, symbols: list[str], tf: str, params: dict, initial_equity: float
    ) -> int:
        cur = self._conn.execute(
            "INSERT INTO runs (mode, started_at, symbols, tf, params_json, initial_equity)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (mode, _iso(datetime.now().astimezone()), json.dumps(symbols), tf,
             json.dumps(params), initial_equity),
        )
        return int(cur.lastrowid)

    def append_equity(
        self, ts: datetime, equity: float, cash: float, n_positions: int, regimes: dict[str, str]
    ) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO equity (ts, equity, cash, n_positions, regimes_json)"
            " VALUES (?, ?, ?, ?, ?)",
            (_iso(ts), equity, cash, n_positions, json.dumps(regimes)),
        )

    def append_fills(self, fills: list[Fill]) -> None:
        self._conn.executemany(
            "INSERT INTO fills (ts, symbol, side, size_base, price, fee) VALUES (?, ?, ?, ?, ?, ?)",
            [(_iso(f.ts), f.symbol, f.side.value, f.size_base, f.price, f.fee) for f in fills],
        )

    def append_trades(self, trades: list[Trade]) -> None:
        self._conn.executemany(
            "INSERT INTO trades (symbol, entry_ts, exit_ts, entry_price, exit_price, size_base,"
            " pnl, fees, exit_reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [(t.symbol, _iso(t.entry_ts), _iso(t.exit_ts), t.entry_price, t.exit_price,
              t.size_base, t.pnl, t.fees, t.exit_reason) for t in trades],
        )

    def replace_positions(self, positions: list[Position]) -> None:
        with self.transaction():
            self._conn.execute("DELETE FROM positions")
            self._conn.executemany(
                "INSERT INTO positions (symbol, size_base, entry_price, stop_price, opened_at,"
                " initial_risk) VALUES (?, ?, ?, ?, ?, ?)",
                [(p.symbol, p.size_base, p.entry_price, p.stop_price, _iso(p.opened_at),
                  p.initial_risk) for p in positions],
            )

    def append_signals(
        self, ts: datetime, symbol: str, regime: str, signals: dict[str, Signal]
    ) -> None:
        self._conn.executemany(
            "INSERT OR REPLACE INTO signals (ts, symbol, agent, direction, confidence, reason,"
            " regime) VALUES (?, ?, ?, ?, ?, ?, ?)",
            [(_iso(ts), symbol, agent, s.direction.value, s.confidence, s.reason, regime)
             for agent, s in signals.items()],
        )

    def append_risk_events(self, events: list[dict]) -> None:
        self._conn.executemany(
            "INSERT INTO risk_events (ts, kind, detail) VALUES (?, ?, ?)",
            [(_iso(e["ts"]) if isinstance(e["ts"], datetime) else e["ts"], e["kind"], e["detail"])
             for e in events],
        )

    def append_rejections(self, ts: datetime, rejections: list[tuple[Order, str]]) -> None:
        self._conn.executemany(
            "INSERT INTO rejections (ts, symbol, side, size_quote, reason) VALUES (?, ?, ?, ?, ?)",
            [(_iso(ts), o.symbol, o.side.value, o.size_quote, r) for o, r in rejections],
        )

    def save_state(self, key: str, obj: Any) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO state (key, json) VALUES (?, ?)",
            (key, json.dumps(obj, default=_json_default)),
        )

    def load_state(self, key: str) -> Any | None:
        row = self._conn.execute("SELECT json FROM state WHERE key = ?", (key,)).fetchone()
        return json.loads(row["json"]) if row else None

    # -- reads ----------------------------------------------------------
    def _rows(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        return self._conn.execute(sql, params).fetchall()

    def status(self) -> dict:
        run = self._conn.execute("SELECT * FROM runs ORDER BY id DESC LIMIT 1").fetchone()
        last = self._conn.execute("SELECT * FROM equity ORDER BY ts DESC LIMIT 1").fetchone()
        n_pos = self._conn.execute("SELECT COUNT(*) AS n FROM positions").fetchone()["n"]
        return {
            "mode": run["mode"] if run else None,
            "started_at": _dt(run["started_at"]) if run else None,
            "initial_equity": run["initial_equity"] if run else None,
            "symbols": json.loads(run["symbols"]) if run else None,
            "tf": run["tf"] if run else None,
            "params": json.loads(run["params_json"]) if run else None,
            "last_ts": _dt(last["ts"]) if last else None,
            "equity": last["equity"] if last else None,
            "cash": last["cash"] if last else None,
            "n_positions": int(n_pos),
        }

    def equity_curve(self, limit: int = 5000) -> list[dict]:
        rows = self._rows(
            "SELECT * FROM (SELECT * FROM equity ORDER BY ts DESC LIMIT ?) ORDER BY ts ASC",
            (limit,),
        )
        return [
            {"ts": _dt(r["ts"]), "equity": r["equity"], "cash": r["cash"],
             "n_positions": r["n_positions"], "regimes": json.loads(r["regimes_json"])}
            for r in rows
        ]

    def first_equity_ts(self) -> datetime | None:
        row = self._conn.execute("SELECT MIN(ts) AS ts FROM equity").fetchone()
        return _dt(row["ts"]) if row and row["ts"] else None

    def max_drawdown(self) -> float:
        peak = 0.0
        worst = 0.0
        for r in self._rows("SELECT equity FROM equity ORDER BY ts ASC"):
            e = float(r["equity"])
            peak = max(peak, e)
            if peak > 0:
                worst = max(worst, (peak - e) / peak)
        return worst

    def trades(self, limit: int = 500) -> list[dict]:
        rows = self._rows("SELECT * FROM trades ORDER BY exit_ts DESC, id DESC LIMIT ?", (limit,))
        return [self._trade_dict(r) for r in rows]

    @staticmethod
    def _trade_dict(r: sqlite3.Row) -> dict:
        d = dict(r)
        d["entry_ts"] = _dt(d["entry_ts"])
        d["exit_ts"] = _dt(d["exit_ts"])
        return d

    def trade_stats(self) -> dict:
        rows = self._rows("SELECT symbol, pnl, fees, exit_reason FROM trades")
        pnls = [float(r["pnl"]) for r in rows]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        if not pnls:
            pf = 0.0
        elif not losses:
            pf = math.inf if wins else 0.0
        else:
            pf = sum(wins) / -sum(losses)
        by_symbol: dict[str, dict] = {}
        by_reason: dict[str, dict] = {}
        for r in rows:
            for key, bucket in ((r["symbol"], by_symbol), (r["exit_reason"], by_reason)):
                b = bucket.setdefault(key, {"count": 0, "pnl": 0.0})
                b["count"] += 1
                b["pnl"] += float(r["pnl"])
        return {
            "count": len(pnls),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / len(pnls) if pnls else 0.0,
            "profit_factor": pf,
            "expectancy": sum(pnls) / len(pnls) if pnls else 0.0,
            "avg_win": sum(wins) / len(wins) if wins else 0.0,
            "avg_loss": sum(losses) / len(losses) if losses else 0.0,
            "gross_pnl": sum(pnls),
            "fees": sum(float(r["fees"]) for r in rows),
            "by_symbol": by_symbol,
            "by_exit_reason": by_reason,
        }

    def positions(self) -> list[dict]:
        out = []
        for r in self._rows("SELECT * FROM positions ORDER BY symbol"):
            d = dict(r)
            d["opened_at"] = _dt(d["opened_at"])
            out.append(d)
        return out

    def latest_signals(self) -> list[dict]:
        rows = self._rows(
            "SELECT s.* FROM signals s JOIN (SELECT symbol, MAX(ts) AS ts FROM signals GROUP BY"
            " symbol) m ON s.symbol = m.symbol AND s.ts = m.ts ORDER BY s.symbol, s.agent"
        )
        out = []
        for r in rows:
            d = dict(r)
            d["ts"] = _dt(d["ts"])
            out.append(d)
        return out

    def risk_events(self, limit: int = 100) -> list[dict]:
        rows = self._rows("SELECT * FROM risk_events ORDER BY ts DESC, id DESC LIMIT ?", (limit,))
        return [{"id": r["id"], "ts": _dt(r["ts"]), "kind": r["kind"], "detail": r["detail"]} for r in rows]

    def rejections(self, limit: int = 100) -> list[dict]:
        rows = self._rows("SELECT * FROM rejections ORDER BY ts DESC, id DESC LIMIT ?", (limit,))
        out = []
        for r in rows:
            d = dict(r)
            d["ts"] = _dt(d["ts"])
            out.append(d)
        return out

    def fills(self, symbol: str | None = None, since: datetime | None = None) -> list[dict]:
        sql = "SELECT * FROM fills WHERE 1=1"
        params: list = []
        if symbol is not None:
            sql += " AND symbol = ?"
            params.append(symbol)
        if since is not None:
            sql += " AND ts >= ?"
            params.append(_iso(since))
        sql += " ORDER BY ts ASC, id ASC"
        out = []
        for r in self._rows(sql, tuple(params)):
            d = dict(r)
            d["ts"] = _dt(d["ts"])
            out.append(d)
        return out


def _json_default(o: Any) -> Any:
    if isinstance(o, datetime):
        return o.isoformat()
    return str(o)
