import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from envs import MultiRelayNavigationEnv
from envs.dynamics import CATCH, MOVE


@pytest.mark.parametrize("num_relays", [0, 1, 2, 4, 8])
def test_supported_horizons_pass_gymnasium_check(num_relays) -> None:
    env = MultiRelayNavigationEnv(
        num_relays=num_relays,
        min_start_goal_distance=10.0,
        max_episode_steps=20,
    )
    check_env(env, skip_render_check=True)
    obs, _ = env.reset(seed=3)
    assert env.num_phases == num_relays + 1
    assert env.current_phase == 0
    assert obs["observation"].shape == (7 + num_relays,)


def test_relays_must_be_completed_in_order_before_final_success() -> None:
    env = MultiRelayNavigationEnv(
        num_relays=2, map_size=200.0, goal_radius=3.0, relay_radius=3.0,
        min_start_goal_distance=10.0, max_episode_steps=20)
    obs, _ = env.reset(seed=1, options={
        "start": [20.0, 20.0],
        "relay_goals": [[20.0, 20.0], [80.0, 20.0]],
        "final_goal": [150.0, 20.0],
        "heading": 0.0,
    })
    obs, reward, terminated, _, info = env.step((CATCH, np.array([0.0], np.float32)))
    assert env.current_phase == 1 and info["phase_changed"]
    np.testing.assert_allclose(obs["desired_goal"], [80.0, 20.0])
    assert reward == -1.0 and not terminated

    # Being at the final goal cannot skip the second relay.
    env.state.x, env.state.y = 150.0, 20.0
    _, reward, terminated, _, _ = env.step((MOVE, np.array([0.0], np.float32)))
    assert reward == -1.0 and not terminated and env.current_phase == 1

    env.state.x, env.state.y = 80.0, 20.0
    obs, _, _, _, _ = env.step((CATCH, np.array([0.0], np.float32)))
    assert env.current_phase == 2
    np.testing.assert_allclose(obs["desired_goal"], [150.0, 20.0])
    env.state.x, env.state.y = 150.0, 20.0
    _, reward, terminated, truncated, info = env.step((MOVE, np.array([0.0], np.float32)))
    assert reward == 0.0 and terminated and not truncated and info["is_success"]


def test_zero_relay_variant_is_direct_sparse_navigation() -> None:
    env = MultiRelayNavigationEnv(
        num_relays=0, map_size=100.0, goal_radius=3.0,
        min_start_goal_distance=5.0)
    env.reset(options={
        "start": [10.0, 10.0], "relay_goals": [],
        "final_goal": [12.0, 10.0], "heading": 0.0,
    })
    assert env.action_space.spaces[0].n == 2
    _, reward, terminated, _, info = env.step((MOVE, np.array([0.5], np.float32)))
    assert reward == 0.0 and terminated and info["is_success"]
