import math

import numpy as np

from envs.dynamics import DynamicsConfig, MOVE, TURN, UAVState, advance


def test_turn_changes_heading_and_then_moves() -> None:
    state = UAVState(10.0, 10.0, 5.0, 0.0)
    result = advance(state, TURN, np.array([0.5], dtype=np.float32), DynamicsConfig())
    assert result.heading == pytest.approx(math.pi / 6)
    assert result.speed == pytest.approx(5.0)
    assert result.y > state.y


def test_move_changes_speed_without_changing_heading() -> None:
    state = UAVState(10.0, 10.0, 0.0, 0.0)
    result = advance(state, MOVE, np.array([0.5], dtype=np.float32), DynamicsConfig())
    assert result.speed == pytest.approx(2.0)
    assert result.heading == pytest.approx(0.0)
    assert result.x == pytest.approx(12.0)


def test_parameter_and_speed_are_clipped() -> None:
    config = DynamicsConfig(max_acceleration=4.0, max_speed=6.0)
    result = advance(UAVState(0.0, 0.0, 5.0, 0.0), MOVE, np.array([99.0]), config)
    assert result.speed == pytest.approx(6.0)


import pytest
