from datetime import UTC, datetime, timedelta

from typer.testing import CliRunner

from cryptofx import cli
from cryptofx.core.models import Trade
from cryptofx.risk.gate import GoLiveGate
from cryptofx.risk.manager import RiskManager
from cryptofx.store.db import Store

T0 = datetime(2024, 1, 1, tzinfo=UTC)
NAMES = ["paper_days", "closed_trades", "profit_factor", "max_drawdown", "expectancy", "no_halt_30d"]


def trade(pnl, i):
    return Trade(symbol="BTC/USDT", entry_ts=T0 + timedelta(hours=i), exit_ts=T0 + timedelta(hours=i + 1),
                 entry_price=100.0, exit_price=100.0 + pnl, size_base=1.0, pnl=pnl, fees=0.0,
                 exit_reason="x")


def seed_passing(store, days=50, n_trades=60, win=10.0, loss=-5.0, dd_equities=None, halt_days_ago=None):
    for d in range(days + 1):
        e = 500.0 + d if dd_equities is None else dd_equities[min(d, len(dd_equities) - 1)]
        store.append_equity(T0 + timedelta(days=d), e, e, 0, {})
    trades = [trade(win if i % 3 else loss, i) for i in range(n_trades)]
    store.append_trades(trades)
    if halt_days_ago is not None:
        store.append_risk_events([{"ts": T0 + timedelta(days=days - halt_days_ago), "kind": "halted", "detail": "dd"}])


def test_passing_store_passes():
    s = Store(":memory:")
    seed_passing(s)
    g = GoLiveGate(s)
    assert [c.name for c in g.evaluate()] == NAMES
    assert all(c.passed for c in g.evaluate()) and g.passes()


def test_empty_store_fails_all():
    g = GoLiveGate(Store(":memory:"))
    assert not any(c.passed for c in g.evaluate()) and not g.passes()


def test_each_criterion_blocks_alone():
    cases = {
        "paper_days": {"days": 41},
        "closed_trades": {"n_trades": 49},
        "profit_factor": {"win": 6.0, "loss": -10.0},  # 2 wins of 6 per loss of 10 -> pf 1.2
        "max_drawdown": {"dd_equities": [500.0, 600.0, 530.0, 700.0]},
        "expectancy": {"win": 1.0, "loss": -5.0},
        "no_halt_30d": {"halt_days_ago": 10},
    }
    for name, kw in cases.items():
        s = Store(":memory:")
        seed_passing(s, **kw)
        failed = [c.name for c in GoLiveGate(s).evaluate() if not c.passed]
        # negative expectancy implies profit factor < 1, so that pair always fails together
        expected = ["profit_factor", "expectancy"] if name == "expectancy" else [name]
        assert failed == expected, (name, failed)


def test_halt_older_than_30_days_is_fine():
    s = Store(":memory:")
    seed_passing(s, halt_days_ago=31)
    assert GoLiveGate(s).passes()


def test_cli_gate_exit_codes(tmp_path):
    db = tmp_path / "g.db"
    s = Store(db)
    res = CliRunner().invoke(cli.app, ["gate", "--db", str(db)])
    assert res.exit_code == 1 and "[FAIL]" in res.output and "paper_days" in res.output
    seed_passing(s)
    res = CliRunner().invoke(cli.app, ["gate", "--db", str(db)])
    assert res.exit_code == 0 and "[FAIL]" not in res.output and res.output.count("[PASS]") == 6


def test_cli_risk_reset_requires_confirm_and_reason(tmp_path):
    db = tmp_path / "r.db"
    s = Store(db)
    rm = RiskManager()
    rm.update_equity(T0, 1000.0)
    rm.update_equity(T0 + timedelta(days=1), 850.0)
    s.save_state("risk", rm.to_state())
    r = CliRunner().invoke(cli.app, ["risk", "reset", "--reason", "checked", "--db", str(db)])
    assert r.exit_code != 0 and s.load_state("risk")["state"] == "halted"
    r = CliRunner().invoke(cli.app, ["risk", "reset", "--confirm", "--reason", "  ", "--db", str(db)])
    assert r.exit_code != 0 and s.load_state("risk")["state"] == "halted"


def test_cli_risk_reset_clears_halt(tmp_path):
    db = tmp_path / "r.db"
    s = Store(db)
    rm = RiskManager()
    rm.update_equity(T0, 1000.0)
    rm.update_equity(T0 + timedelta(days=1), 850.0)
    s.save_state("risk", rm.to_state())
    r = CliRunner().invoke(cli.app, ["risk", "reset", "--confirm", "--reason", "reviewed, resuming", "--db", str(db)])
    assert r.exit_code == 0, r.output
    assert s.load_state("risk")["state"] == "active"
    assert s.risk_events()[0]["kind"] == "reset" and "reviewed" in s.risk_events()[0]["detail"]
    r2 = CliRunner().invoke(cli.app, ["risk", "reset", "--confirm", "--reason", "again", "--db", str(db)])
    assert r2.exit_code != 0 and "not halted" in r2.output.lower()


def test_error_halt_blocks_gate():
    s = Store(":memory:")
    seed_passing(s)
    s.append_risk_events([{"ts": T0 + timedelta(days=45), "kind": "error", "detail": "crash"}])
    failed = [c.name for c in GoLiveGate(s).evaluate() if not c.passed]
    assert failed == ["no_halt_30d"]


def test_flood_of_data_events_does_not_hide_halt():
    s = Store(":memory:")
    seed_passing(s, halt_days_ago=5)
    s.append_risk_events([{"ts": T0 + timedelta(days=49, seconds=i), "kind": "data", "detail": "x"}
                          for i in range(12_000)])
    assert not GoLiveGate(s).passes()
