import math

import numpy as np

from envs.direct_navigation import DirectNavigationEnv
from envs.dynamics import MOVE, TURN


def test_reset_is_seed_deterministic_and_space_valid() -> None:
    env = DirectNavigationEnv()
    first, _ = env.reset(seed=7)
    second, _ = env.reset(seed=7)
    for key in first:
        np.testing.assert_allclose(first[key], second[key])
    assert env.observation_space.contains(first)


def test_sparse_reward_and_success_detection() -> None:
    env = DirectNavigationEnv(goal_radius=5.0)
    env.reset(options={"start": [10.0, 10.0], "goal": [12.0, 10.0], "heading": 0.0})
    _, reward, terminated, truncated, info = env.step((MOVE, np.array([0.5], dtype=np.float32)))
    assert reward == 0.0
    assert terminated and not truncated and info["is_success"]


def test_out_of_bounds_terminates() -> None:
    env = DirectNavigationEnv(
        map_size=100.0, goal_radius=2.0, min_start_goal_distance=10.0,
        boundary_mode="terminate")
    env.reset(options={"start": [99.0, 50.0], "goal": [10.0, 10.0], "heading": 0.0, "speed": 5.0})
    _, reward, terminated, _, info = env.step((TURN, np.array([0.0], dtype=np.float32)))
    assert reward == -1.0
    assert terminated and info["out_of_bounds"]


def test_default_boundary_clips_without_early_termination() -> None:
    env = DirectNavigationEnv(map_size=100.0, goal_radius=2.0, min_start_goal_distance=10.0)
    env.reset(options={"start": [99.0, 50.0], "goal": [10.0, 10.0], "heading": 0.0, "speed": 5.0})
    obs, reward, terminated, truncated, info = env.step(
        (TURN, np.array([0.0], dtype=np.float32)))
    assert reward == -1.0 and not terminated and not truncated
    assert info["boundary_hit"] and not info["out_of_bounds"]
    np.testing.assert_allclose(obs["achieved_goal"], [100.0, 50.0])
    assert env.observation_space.contains(obs)


def test_repeated_boundary_hits_cannot_shorten_sparse_failure_return() -> None:
    env = DirectNavigationEnv(
        map_size=100.0, goal_radius=2.0, min_start_goal_distance=10.0,
        max_episode_steps=3)
    env.reset(options={"start": [99.0, 50.0], "goal": [10.0, 10.0], "heading": 0.0, "speed": 5.0})
    total_reward = 0.0
    for step in range(3):
        _, reward, terminated, truncated, info = env.step(
            (TURN, np.array([0.0], dtype=np.float32)))
        total_reward += reward
        assert not terminated and info["boundary_hit"]
        assert truncated == (step == 2)
    assert total_reward == -3.0


def test_compute_reward_supports_batches() -> None:
    env = DirectNavigationEnv(goal_radius=1.0)
    achieved = np.asarray([[0.0, 0.0], [3.0, 0.0]], dtype=np.float32)
    desired = np.asarray([[0.5, 0.0], [0.0, 0.0]], dtype=np.float32)
    np.testing.assert_array_equal(env.compute_reward(achieved, desired, None), [0.0, -1.0])


def test_scripted_controller_solves_direct_navigation() -> None:
    env = DirectNavigationEnv(map_size=500.0, goal_radius=15.0, max_episode_steps=100)
    obs, _ = env.reset(
        options={"start": [50.0, 50.0], "goal": [400.0, 350.0], "heading": 0.0}
    )
    success = False
    for _ in range(100):
        position = obs["achieved_goal"]
        delta = obs["desired_goal"] - position
        target_heading = math.atan2(float(delta[1]), float(delta[0]))
        heading = float(obs["observation"][3])
        error = (target_heading - heading + math.pi) % (2 * math.pi) - math.pi
        if abs(error) > 0.08:
            action = (TURN, np.array([np.clip(error / env.dynamics.max_turn_angle, -1, 1)], dtype=np.float32))
        else:
            remaining = float(np.linalg.norm(delta))
            parameter = 1.0 if remaining > 80.0 else -1.0
            action = (MOVE, np.array([parameter], dtype=np.float32))
        obs, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            success = info["is_success"]
            break
    assert success
