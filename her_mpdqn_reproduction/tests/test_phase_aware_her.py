import numpy as np
import pytest

from agents import MPDQNAgent, PDQNConfig
from agents.common import goal_observation_to_vector
from envs import RelayNavigationEnv
from envs.dynamics import CATCH, MOVE
from replay import (
    FutureGoalRelabeler,
    GoalTransition,
    PhaseAwareFutureGoalRelabeler,
    PhaseAwareHERReplayBuffer,
    ReplayBufferConfig,
)


def observation(position, desired, phase, step):
    position = np.asarray(position, dtype=np.float32)
    desired = np.asarray(desired, dtype=np.float32)
    # Relay paper state plus the explicit Markov phase bit.
    state = np.array([
        position[0], position[1], 0.0, 0.0,
        np.linalg.norm(position - np.array([90.0, 10.0])),
        np.linalg.norm(position - np.array([30.0, 10.0])),
        step, phase,
    ], dtype=np.float32)
    return {
        "observation": state,
        "achieved_goal": position,
        "desired_goal": desired,
    }


def two_phase_episode():
    positions = [(10, 10), (20, 10), (30, 10), (50, 10), (70, 10)]
    phases = [0, 0, 1, 1]
    transitions = []
    for index, phase in enumerate(phases):
        desired = (30, 10) if phase == 0 else (90, 10)
        next_desired = (90, 10) if index == 1 else desired
        transitions.append(GoalTransition(
            observation=observation(positions[index], desired, phase, index),
            action=CATCH if index == 1 else MOVE,
            action_parameters=np.array([0.2, -0.1], dtype=np.float32),
            reward=-1.0,
            next_observation=observation(
                positions[index + 1], next_desired,
                1 if index >= 1 else 0, index + 1),
            terminated=False,
            truncated=index == len(phases) - 1,
            phase=phase,
        ))
    return transitions


def reward(achieved, desired, info):
    del info
    return 0.0 if np.linalg.norm(achieved - desired) <= 0.1 else -1.0


def test_phase_aware_her_never_assigns_phase1_goal_to_phase0() -> None:
    episode = two_phase_episode()
    relabeler = PhaseAwareFutureGoalRelabeler(her_k=4, seed=3)
    relabeled = relabeler.relabel_episode(episode, reward)
    assert len(relabeled) == 2 + 1 + 2 + 1
    for item in relabeled:
        assert episode[item.future_index].phase == episode[item.source_index].phase
    phase1_goals = {
        tuple(episode[index].next_observation["achieved_goal"])
        for index in (2, 3)
    }
    phase0_items = [item for item in relabeled if episode[item.source_index].phase == 0]
    assert all(tuple(item.relabeled_goal) not in phase1_goals for item in phase0_items)
    assert relabeler.last_cross_phase_filtered == 4


def test_ordinary_her_can_cross_phase_showing_ablation_is_real() -> None:
    episode = two_phase_episode()
    ordinary = FutureGoalRelabeler(her_k=4, seed=3).relabel_episode(episode, reward)
    assert any(
        episode[item.source_index].phase == 0
        and episode[item.future_index].phase == 1
        for item in ordinary
    )


def test_phase_sequence_must_be_monotonic_and_cannot_skip() -> None:
    episode = two_phase_episode()
    decreasing = list(episode)
    source = decreasing[-1]
    decreasing[-1] = GoalTransition(
        observation=source.observation,
        action=source.action,
        action_parameters=source.action_parameters,
        reward=source.reward,
        next_observation=source.next_observation,
        terminated=source.terminated,
        truncated=source.truncated,
        phase=0,
    )
    with pytest.raises(ValueError, match="non-decreasing"):
        PhaseAwareFutureGoalRelabeler().relabel_episode(decreasing, reward)

    skipped = list(episode)
    source = skipped[2]
    skipped[2] = GoalTransition(
        observation=source.observation,
        action=source.action,
        action_parameters=source.action_parameters,
        reward=source.reward,
        next_observation=source.next_observation,
        terminated=source.terminated,
        truncated=source.truncated,
        phase=2,
    )
    with pytest.raises(ValueError, match="cannot skip"):
        PhaseAwareFutureGoalRelabeler().relabel_episode(skipped, reward)


def test_phase_aware_buffer_reports_filter_audit_counts() -> None:
    buffer = PhaseAwareHERReplayBuffer(ReplayBufferConfig(
        capacity=64,
        state_dim=10,
        parameter_dim=2,
        num_actions=3,
        seed=7,
    ), her_k=4)
    counts = buffer.add_episode(two_phase_episode(), reward)
    assert counts["original_transition_count"] == 4
    assert counts["her_relabel_count"] == 6
    assert counts["cross_phase_goal_count_filtered"] == 4
    assert buffer.metrics["cross_phase_goal_count_filtered"] == 4
    stored_phases = buffer.replay.states[:10, 7]
    # Four originals plus three relabeled transitions in each phase.
    assert np.sum(stored_phases == 0) == 5
    assert np.sum(stored_phases == 1) == 5


def test_real_relay_boundary_transition_is_phase0_then_phase1() -> None:
    env = RelayNavigationEnv(
        map_size=200.0,
        goal_radius=5.0,
        relay_radius=5.0,
        min_start_goal_distance=10.0,
        max_episode_steps=6,
    )
    obs, _ = env.reset(seed=8, options={
        "start": [50.0, 50.0],
        "relay_goal": [50.0, 50.0],
        "final_goal": [150.0, 50.0],
        "heading": 0.0,
    })
    transitions = []
    actions = [CATCH, MOVE, MOVE, MOVE, MOVE, MOVE]
    parameters = [0.0, 1.0, 1.0, 1.0, -1.0, -1.0]
    for action, parameter in zip(actions, parameters):
        source_phase = env.current_phase
        next_obs, step_reward, terminated, truncated, info = env.step(
            (action, np.array([parameter], dtype=np.float32)))
        transitions.append(GoalTransition(
            observation=obs,
            action=action,
            action_parameters=np.array([parameter, 0.0], dtype=np.float32),
            reward=step_reward,
            next_observation=next_obs,
            terminated=terminated,
            truncated=truncated,
            info=info,
            phase=source_phase,
        ))
        obs = next_obs
        if terminated or truncated:
            break
    assert transitions[0].phase == 0
    assert all(transition.phase == 1 for transition in transitions[1:])

    state_dim = goal_observation_to_vector(transitions[0].observation).size
    buffer = PhaseAwareHERReplayBuffer(ReplayBufferConfig(
        capacity=64,
        state_dim=state_dim,
        parameter_dim=2,
        num_actions=3,
        seed=8,
    ), her_k=2)
    counts = buffer.add_episode(
        transitions,
        env.compute_reward,
        goal_validator=env.observation_space["desired_goal"].contains,
    )
    assert counts["her_relabel_count"] > 0
    assert counts["cross_phase_goal_count_filtered"] > 0

    agent = MPDQNAgent(PDQNConfig(
        state_dim=state_dim,
        parameter_sizes=(1, 1, 0),
        hidden_sizes=(16,),
        seed=8,
    ))
    metrics = agent.update(buffer.sample(min(8, len(buffer))))
    assert np.isfinite(metrics["q_loss"])
    assert np.isfinite(metrics["parameter_actor_loss"])
