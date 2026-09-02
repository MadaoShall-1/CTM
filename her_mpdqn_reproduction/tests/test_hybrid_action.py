import numpy as np
import pytest
import torch
from torch import nn

from agents import MPDQNAgent, PDQNAgent, PDQNConfig, ParameterizedActionSpec, TransitionBatch
from agents.networks import build_multipass_parameters, multipass_q_values


class ParameterSensitiveQ(nn.Module):
    """Q outputs deliberately depend on all supplied parameter dimensions."""

    def forward(self, states, parameters):
        del states
        first = parameters[:, 0] + 10.0 * parameters[:, 1]
        second = 20.0 * parameters[:, 0] + parameters[:, 1]
        return torch.stack((first, second), dim=1)


def test_parameter_spec_handles_parameterless_catch() -> None:
    spec = ParameterizedActionSpec((1, 1, 0))
    assert spec.num_actions == 3
    assert spec.total_parameter_dim == 2
    np.testing.assert_array_equal(spec.selected_parameter(np.array([0.2, -0.3]), 2), [0.0])
    np.testing.assert_array_equal(
        spec.masks().numpy(),
        [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
    )


def test_multipass_masks_irrelevant_parameters_exactly() -> None:
    spec = ParameterizedActionSpec((1, 1))
    parameters = torch.tensor([[0.25, -0.75], [0.5, 0.2]])
    masked = build_multipass_parameters(parameters, spec)
    expected = torch.tensor([
        [[0.25, 0.0], [0.0, -0.75]],
        [[0.5, 0.0], [0.0, 0.2]],
    ])
    torch.testing.assert_close(masked, expected)


def test_mpdqn_q_value_ignores_unrelated_parameter() -> None:
    spec = ParameterizedActionSpec((1, 1))
    network = ParameterSensitiveQ()
    states = torch.zeros(1, 3)
    first = multipass_q_values(network, states, torch.tensor([[0.4, -0.8]]), spec)
    changed = multipass_q_values(network, states, torch.tensor([[0.4, 0.9]]), spec)
    assert first[0, 0].item() == pytest.approx(changed[0, 0].item())
    # Action 1 legitimately changes because its own parameter changed.
    assert first[0, 1].item() != pytest.approx(changed[0, 1].item())
    # A joint P-DQN pass would allow the unrelated second parameter to alter Q0.
    joint_first = network(states, torch.tensor([[0.4, -0.8]]))
    joint_changed = network(states, torch.tensor([[0.4, 0.9]]))
    assert joint_first[0, 0].item() != pytest.approx(joint_changed[0, 0].item())


@pytest.mark.parametrize("agent_class", [PDQNAgent, MPDQNAgent])
def test_agent_update_returns_finite_losses_and_updates_targets(agent_class) -> None:
    config = PDQNConfig(
        state_dim=5,
        parameter_sizes=(1, 1),
        hidden_sizes=(32, 16),
        tau=0.25,
        seed=11,
    )
    agent = agent_class(config)
    rng = np.random.default_rng(3)
    batch = TransitionBatch.from_numpy(
        states=rng.normal(size=(16, 5)).astype(np.float32),
        actions=rng.integers(0, 2, size=16),
        action_parameters=rng.uniform(-1, 1, size=(16, 2)).astype(np.float32),
        rewards=rng.choice([-1.0, 0.0], size=16).astype(np.float32),
        next_states=rng.normal(size=(16, 5)).astype(np.float32),
        dones=rng.integers(0, 2, size=16).astype(np.float32),
    )
    target_before = [value.detach().clone() for value in agent.target_q_network.parameters()]
    metrics = agent.update(batch)
    assert metrics["update_steps"] == 1.0
    for name in ("q_loss", "parameter_actor_loss", "mean_q", "mean_target_q"):
        assert np.isfinite(metrics[name])
    assert any(
        not torch.equal(before, after)
        for before, after in zip(target_before, agent.target_q_network.parameters())
    )


@pytest.mark.parametrize("agent_class", [PDQNAgent, MPDQNAgent])
def test_action_selection_is_bounded_and_reproducible(agent_class) -> None:
    config = PDQNConfig(
        state_dim=4,
        parameter_sizes=(1, 1, 0),
        hidden_sizes=(16,),
        seed=29,
    )
    first = agent_class(config)
    second = agent_class(config)
    state = np.array([0.1, -0.2, 0.3, 0.4], dtype=np.float32)
    selected_first = first.select_action(state, epsilon=1.0, parameter_noise_std=0.2)
    selected_second = second.select_action(state, epsilon=1.0, parameter_noise_std=0.2)
    assert selected_first.discrete_action == selected_second.discrete_action
    np.testing.assert_allclose(selected_first.all_parameters, selected_second.all_parameters)
    assert np.all(np.abs(selected_first.all_parameters) <= 1.0)
    assert selected_first.selected_parameter.shape == (1,)
