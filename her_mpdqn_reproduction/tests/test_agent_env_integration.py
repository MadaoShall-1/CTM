import numpy as np
import pytest

from agents import MPDQNAgent, PDQNAgent, PDQNConfig, TransitionBatch
from agents.common import goal_observation_to_vector
from envs import DirectNavigationEnv, RelayNavigationEnv


def collect_random_batch(env, agent, size=8):
    observations = []
    actions = []
    parameters = []
    rewards = []
    next_observations = []
    dones = []
    obs, _ = env.reset(seed=5)
    for _ in range(size):
        state = goal_observation_to_vector(obs)
        selection = agent.select_action(state, epsilon=1.0, parameter_noise_std=0.1)
        next_obs, reward, terminated, truncated, _ = env.step(selection.environment_action())
        observations.append(state)
        actions.append(selection.discrete_action)
        parameters.append(selection.all_parameters)
        rewards.append(reward)
        next_observations.append(goal_observation_to_vector(next_obs))
        dones.append(terminated or truncated)
        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()
    return TransitionBatch.from_numpy(
        states=np.asarray(observations),
        actions=np.asarray(actions),
        action_parameters=np.asarray(parameters),
        rewards=np.asarray(rewards),
        next_states=np.asarray(next_observations),
        dones=np.asarray(dones),
    )


@pytest.mark.parametrize("agent_class", [PDQNAgent, MPDQNAgent])
@pytest.mark.parametrize(
    ("env", "parameter_sizes"),
    [
        (DirectNavigationEnv(max_episode_steps=12), (1, 1)),
        (RelayNavigationEnv(max_episode_steps=12), (1, 1, 0)),
    ],
)
def test_agent_interacts_with_uav_env_and_updates(agent_class, env, parameter_sizes) -> None:
    obs, _ = env.reset(seed=1)
    state_dim = goal_observation_to_vector(obs).size
    agent = agent_class(PDQNConfig(
        state_dim=state_dim,
        parameter_sizes=parameter_sizes,
        hidden_sizes=(16,),
        seed=13,
    ))
    batch = collect_random_batch(env, agent)
    metrics = agent.update(batch)
    assert np.isfinite(metrics["q_loss"])
    assert np.isfinite(metrics["parameter_actor_loss"])
