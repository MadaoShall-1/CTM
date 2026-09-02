import numpy as np

from envs.dynamics import CATCH, MOVE
from envs.relay_navigation import RelayNavigationEnv


def make_env(require_catch_action: bool = True) -> RelayNavigationEnv:
    env = RelayNavigationEnv(
        map_size=200.0,
        goal_radius=5.0,
        relay_radius=5.0,
        min_start_goal_distance=10.0,
        require_catch_action=require_catch_action,
    )
    env.reset(
        options={
            "start": [50.0, 50.0],
            "relay_goal": [50.0, 50.0],
            "final_goal": [150.0, 50.0],
            "heading": 0.0,
        }
    )
    return env


def test_relay_requires_catch_and_switches_active_goal() -> None:
    env = make_env()
    before = env.current_goal
    _, reward, terminated, _, info = env.step((MOVE, np.array([0.0], dtype=np.float32)))
    assert env.current_phase == 0
    assert reward == -1.0 and not terminated
    np.testing.assert_allclose(env.current_goal, before)

    obs, reward, terminated, _, info = env.step((CATCH, np.array([0.0], dtype=np.float32)))
    assert env.current_phase == 1 and info["phase_changed"]
    assert reward == -1.0 and not terminated
    np.testing.assert_allclose(obs["desired_goal"], [150.0, 50.0])


def test_final_goal_cannot_succeed_before_relay_phase() -> None:
    env = make_env()
    env.state.x = 150.0
    env.state.y = 50.0
    _, reward, terminated, _, info = env.step((MOVE, np.array([0.0], dtype=np.float32)))
    assert reward == -1.0
    assert not terminated and not info["is_success"]


def test_optional_automatic_pickup_switches_on_contact() -> None:
    env = make_env(require_catch_action=False)
    obs, _, _, _, info = env.step((MOVE, np.array([0.0], dtype=np.float32)))
    assert env.current_phase == 1 and info["phase_changed"]
    np.testing.assert_allclose(obs["desired_goal"], [150.0, 50.0])


def test_delivery_succeeds_only_after_pickup() -> None:
    env = make_env()
    env.step((CATCH, np.array([0.0], dtype=np.float32)))
    env.state.x = 150.0
    env.state.y = 50.0
    _, reward, terminated, truncated, info = env.step(
        (MOVE, np.array([0.0], dtype=np.float32))
    )
    assert reward == 0.0
    assert terminated and not truncated and info["is_success"]
