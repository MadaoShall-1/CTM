import numpy as np
import pytest

from agents import MPDQNAgent, PDQNConfig
from agents.common import goal_observation_to_vector
from envs import DirectNavigationEnv
from replay import (
    FutureGoalRelabeler,
    GoalTransition,
    HERReplayBuffer,
    ReplayBufferConfig,
)


def observation(position, desired=(90.0, 90.0), step=0):
    position = np.asarray(position, dtype=np.float32)
    desired = np.asarray(desired, dtype=np.float32)
    state = np.array(
        [position[0], position[1], 0.0, 0.0, np.linalg.norm(position - desired), step],
        dtype=np.float32,
    )
    return {
        "observation": state,
        "achieved_goal": position,
        "desired_goal": desired,
    }


def make_episode(*, final_truncated=False):
    positions = [(10.0, 10.0), (20.0, 10.0), (30.0, 10.0), (40.0, 10.0)]
    transitions = []
    for index in range(3):
        transitions.append(GoalTransition(
            observation=observation(positions[index], step=index),
            action=index % 2,
            action_parameters=np.array([0.2, -0.4], dtype=np.float32),
            reward=-1.0,
            next_observation=observation(positions[index + 1], step=index + 1),
            terminated=False,
            truncated=final_truncated and index == 2,
            phase=0,
        ))
    return transitions


def test_future_strategy_only_uses_later_achieved_goals() -> None:
    env = DirectNavigationEnv(goal_radius=0.1)
    relabeled = FutureGoalRelabeler(her_k=4, seed=4).relabel_episode(
        make_episode(), env.compute_reward)
    assert len(relabeled) == 3 + 2 + 1
    assert all(item.future_index >= item.source_index for item in relabeled)
    for item in relabeled:
        expected = make_episode()[item.future_index].next_observation["achieved_goal"]
        np.testing.assert_allclose(item.relabeled_goal, expected)


def test_relabeling_replaces_both_goals_and_recomputes_reward() -> None:
    env = DirectNavigationEnv(goal_radius=0.1)
    episode = make_episode()
    relabeled = FutureGoalRelabeler(her_k=4, seed=2).relabel_episode(
        episode, env.compute_reward)
    for item in relabeled:
        transition = item.transition
        np.testing.assert_allclose(
            transition.observation["desired_goal"], item.relabeled_goal)
        np.testing.assert_allclose(
            transition.next_observation["desired_goal"], item.relabeled_goal)
        expected_reward = env.compute_reward(
            transition.next_observation["achieved_goal"], item.relabeled_goal, None)
        assert transition.reward == expected_reward
        assert transition.terminated == (expected_reward == 0.0)


def test_virtual_success_overrides_source_time_limit() -> None:
    env = DirectNavigationEnv(goal_radius=0.1)
    relabeled = FutureGoalRelabeler(her_k=1, seed=1).relabel_episode(
        make_episode(final_truncated=True), env.compute_reward)
    final = next(item for item in relabeled if item.source_index == 2)
    assert final.transition.reward == 0.0
    assert final.transition.terminated
    assert not final.transition.truncated


def test_goal_validator_excludes_invalid_future_goal() -> None:
    env = DirectNavigationEnv(map_size=100.0, goal_radius=1.0)
    episode = make_episode()
    bad_next = observation((120.0, 10.0), step=3)
    episode[-1] = GoalTransition(
        observation=episode[-1].observation,
        action=episode[-1].action,
        action_parameters=episode[-1].action_parameters,
        reward=-1.0,
        next_observation=bad_next,
        terminated=True,
        truncated=False,
    )
    validator = env.observation_space["desired_goal"].contains
    relabeled = FutureGoalRelabeler(her_k=4, seed=0).relabel_episode(
        episode, env.compute_reward, goal_validator=validator)
    assert all(np.all(item.relabeled_goal <= 100.0) for item in relabeled)
    assert len(relabeled) == 2 + 1


def test_her_buffer_stores_original_and_relabelled_transitions() -> None:
    env = DirectNavigationEnv(goal_radius=0.1)
    config = ReplayBufferConfig(
        capacity=32,
        state_dim=8,
        parameter_dim=2,
        num_actions=2,
        seed=5,
    )
    buffer = HERReplayBuffer(config, her_k=4)
    counts = buffer.add_episode(
        make_episode(),
        env.compute_reward,
        goal_validator=env.observation_space["desired_goal"].contains,
    )
    assert counts == {
        "original_transition_count": 3,
        "her_relabel_count": 6,
        "stored_transition_count": 9,
        "total_her_relabel_count": 6,
    }
    assert len(buffer) == 9
    assert buffer.metrics["her_relabel_count"] == 6
    batch = buffer.sample(9)
    assert batch.states.shape == (9, 8)
    # Original desired goals are (90,90); HER transitions have path goals.
    stored_goals = buffer.replay.states[:9, -2:]
    assert np.sum(np.all(stored_goals == [90.0, 90.0], axis=1)) == 3


def test_relabeling_is_seed_reproducible() -> None:
    env = DirectNavigationEnv(goal_radius=0.1)
    episode = make_episode()
    first = FutureGoalRelabeler(her_k=2, seed=44).relabel_episode(
        episode, env.compute_reward)
    second = FutureGoalRelabeler(her_k=2, seed=44).relabel_episode(
        episode, env.compute_reward)
    assert [(item.source_index, item.future_index) for item in first] == [
        (item.source_index, item.future_index) for item in second]


def test_empty_episode_is_rejected_by_buffer() -> None:
    config = ReplayBufferConfig(8, 8, 2, 2)
    buffer = HERReplayBuffer(config)
    with pytest.raises(ValueError, match="empty episode"):
        buffer.add_episode([], lambda achieved, desired, info: -1.0)


def test_real_direct_episode_relabels_and_updates_agent() -> None:
    env = DirectNavigationEnv(max_episode_steps=6)
    obs, _ = env.reset(
        seed=12,
        options={"start": [500.0, 500.0], "goal": [1500.0, 1500.0], "heading": 0.0},
    )
    state_dim = goal_observation_to_vector(obs).size
    agent = MPDQNAgent(PDQNConfig(
        state_dim=state_dim,
        parameter_sizes=(1, 1),
        hidden_sizes=(16,),
        seed=12,
    ))
    transitions = []
    while True:
        selection = agent.select_action(
            goal_observation_to_vector(obs), epsilon=1.0, parameter_noise_std=0.1)
        next_obs, reward, terminated, truncated, info = env.step(
            selection.environment_action())
        transitions.append(GoalTransition(
            observation=obs,
            action=selection.discrete_action,
            action_parameters=selection.all_parameters,
            reward=reward,
            next_observation=next_obs,
            terminated=terminated,
            truncated=truncated,
            info=info,
            phase=env.current_phase,
        ))
        obs = next_obs
        if terminated or truncated:
            break

    buffer = HERReplayBuffer(ReplayBufferConfig(
        capacity=64,
        state_dim=state_dim,
        parameter_dim=2,
        num_actions=2,
        seed=12,
    ), her_k=2)
    counts = buffer.add_episode(
        transitions,
        env.compute_reward,
        goal_validator=env.observation_space["desired_goal"].contains,
    )
    assert counts["her_relabel_count"] > 0
    metrics = agent.update(buffer.sample(8))
    assert np.isfinite(metrics["q_loss"])
    assert np.isfinite(metrics["parameter_actor_loss"])
