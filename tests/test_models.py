from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from cryptofx.core.models import Direction, Position, Signal


def test_signal_confidence_bounds():
    with pytest.raises(ValidationError):
        Signal(direction=Direction.LONG, confidence=1.5, reason="x")


def test_direction_has_no_short():
    assert {d.value for d in Direction} == {"long", "flat"}


def test_position_stop_is_mutable():
    p = Position(
        symbol="BTC/USDT",
        size_base=0.01,
        entry_price=100.0,
        stop_price=90.0,
        opened_at=datetime(2024, 1, 1, tzinfo=UTC),
        initial_risk=10.0,
    )
    p.stop_price = 95.0
    assert p.stop_price == 95.0
