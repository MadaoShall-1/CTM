import numpy as np
import pytest
import torch

from agents import MPDQNAgent, PDQNConfig
from agents.common import goal_observation_to_vector
from envs import DirectNavigationEnv
from replay import ReplayBuffer, ReplayBufferConfig


def make_buffer(capacity=5, seed=3):
    return ReplayBuffer(ReplayBufferConfig(
        capacity=capacity,
        state_dim=3,
        parameter_dim=2,
        num_actions=2,
        seed=seed,
    ))


def add_transition(buffer, value, *, terminated=False, truncated=False):
    return buffer.add(
        state=np.full(3, value, dtype=np.float32),
        action=value % 2,
        action_parameters=np.array([value, -value], dtype=np.float32),
        reward=float(-value),
        next_state=np.full(3, value + 1, dtype=np.float32),
        terminated=terminated,
        truncated=truncated,
    )


def test_circular_buffer_overwrites_oldest_physical_slot() -> None:
    buffer = make_buffer(capacity=3)
    for value in range(5):
        add_transition(buffer, value)
    assert len(buffer) == 3 and buffer.full
    assert buffer.total_added == 5
    assert buffer.position == 2
    # Physical slots 0 and 1 have been overwritten by transitions 3 and 4.
    np.testing.assert_array_equal(buffer.states[:, 0], [3.0, 4.0, 2.0])


def test_sampling_is_seed_deterministic() -> None:
    first = make_buffer(seed=17)
    second = make_buffer(seed=17)
    for value in range(5):
        add_transition(first, value)
        add_transition(second, value)
    np.testing.assert_array_equal(first.sample_indices(4), second.sample_indices(4))


def test_truncation_bootstraps_by_default_but_can_be_terminal() -> None:
    buffer = make_buffer()
    add_transition(buffer, 0, truncated=True)
    add_transition(buffer, 1, terminated=True)
    bootstrapped = buffer.batch_from_indices([0, 1])
    terminal_time_limit = buffer.batch_from_indices(
        [0, 1], bootstrap_on_truncation=False)
    torch.testing.assert_close(bootstrapped.dones, torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(terminal_time_limit.dones, torch.tensor([1.0, 1.0]))


def test_validation_rejects_bad_transition() -> None:
    buffer = make_buffer()
    with pytest.raises(ValueError, match="state must have shape"):
        buffer.add(
            state=np.zeros(2), action=0, action_parameters=np.zeros(2), reward=-1,
            next_state=np.zeros(3), terminated=False, truncated=False)
    with pytest.raises(ValueError, match="both terminated and truncated"):
        buffer.add(
            state=np.zeros(3), action=0, action_parameters=np.zeros(2), reward=-1,
            next_state=np.zeros(3), terminated=True, truncated=True)


def test_save_restore_preserves_data_and_rng(tmp_path) -> None:
    buffer = make_buffer(capacity=4, seed=23)
    for value in range(6):
        add_transition(buffer, value, truncated=value == 5)
    path = tmp_path / "replay.npz"
    buffer.save(path)
    restored = ReplayBuffer.load(path)
    assert restored.position == buffer.position
    assert restored.size == buffer.size
    assert restored.total_added == buffer.total_added
    np.testing.assert_array_equal(restored.states, buffer.states)
    np.testing.assert_array_equal(restored.truncated, buffer.truncated)
    # RNG state is restored, so the next sample is identical.
    np.testing.assert_array_equal(restored.sample_indices(3), buffer.sample_indices(3))


def test_real_uav_replay_batch_updates_mpdqn() -> None:
    env = DirectNavigationEnv(max_episode_steps=20)
    obs, _ = env.reset(seed=9)
    state_dim = goal_observation_to_vector(obs).size
    agent = MPDQNAgent(PDQNConfig(
        state_dim=state_dim,
        parameter_sizes=(1, 1),
        hidden_sizes=(16,),
        seed=9,
    ))
    buffer = ReplayBuffer(ReplayBufferConfig(
        capacity=32,
        state_dim=state_dim,
        parameter_dim=2,
        num_actions=2,
        seed=9,
    ))
    for _ in range(12):
        state = goal_observation_to_vector(obs)
        selection = agent.select_action(state, epsilon=1.0, parameter_noise_std=0.1)
        next_obs, reward, terminated, truncated, _ = env.step(selection.environment_action())
        buffer.add_selection(
            state=state,
            selection=selection,
            reward=reward,
            next_state=goal_observation_to_vector(next_obs),
            terminated=terminated,
            truncated=truncated,
        )
        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()
    metrics = agent.update(buffer.sample(8))
    assert np.isfinite(metrics["q_loss"])
    assert np.isfinite(metrics["parameter_actor_loss"])
