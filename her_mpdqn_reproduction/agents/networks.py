"""PyTorch networks and MP-DQN masking primitives."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from .common import ParameterizedActionSpec


def build_mlp(input_dim: int, hidden_sizes: Sequence[int], output_dim: int) -> nn.Sequential:
    if input_dim <= 0 or output_dim <= 0:
        raise ValueError("Network input and output dimensions must be positive")
    layers: list[nn.Module] = []
    previous = input_dim
    for size in hidden_sizes:
        if size <= 0:
            raise ValueError("Hidden layer sizes must be positive")
        linear = nn.Linear(previous, int(size))
        nn.init.kaiming_uniform_(linear.weight, nonlinearity="relu")
        nn.init.zeros_(linear.bias)
        layers.extend((linear, nn.ReLU()))
        previous = int(size)
    output = nn.Linear(previous, output_dim)
    nn.init.uniform_(output.weight, -3e-3, 3e-3)
    nn.init.zeros_(output.bias)
    layers.append(output)
    return nn.Sequential(*layers)


class ParameterActor(nn.Module):
    """Deterministic actor producing all valid action parameters in [-1, 1]."""

    def __init__(
        self,
        state_dim: int,
        parameter_dim: int,
        hidden_sizes: Sequence[int] = (128, 64),
    ) -> None:
        super().__init__()
        self.network = build_mlp(state_dim, hidden_sizes, parameter_dim)

    def forward(self, states: Tensor) -> Tensor:
        return torch.tanh(self.network(states))


class QNetwork(nn.Module):
    """Shared Q network returning one value per discrete action."""

    def __init__(
        self,
        state_dim: int,
        parameter_dim: int,
        num_actions: int,
        hidden_sizes: Sequence[int] = (128, 64),
    ) -> None:
        super().__init__()
        self.network = build_mlp(state_dim + parameter_dim, hidden_sizes, num_actions)

    def forward(self, states: Tensor, action_parameters: Tensor) -> Tensor:
        if states.ndim != 2 or action_parameters.ndim != 2:
            raise ValueError("Q-network inputs must be rank-two batches")
        return self.network(torch.cat((states, action_parameters), dim=-1))


def build_multipass_parameters(parameters: Tensor, spec: ParameterizedActionSpec) -> Tensor:
    """Return [batch, action, parameter] inputs with unrelated slots zeroed."""

    if parameters.ndim != 2 or parameters.shape[-1] != spec.total_parameter_dim:
        raise ValueError(
            f"Expected parameters [batch, {spec.total_parameter_dim}], got {tuple(parameters.shape)}")
    masks = spec.masks(device=parameters.device).to(parameters.dtype)
    return parameters[:, None, :] * masks[None, :, :]


def multipass_q_values(
    q_network: nn.Module,
    states: Tensor,
    parameters: Tensor,
    spec: ParameterizedActionSpec,
) -> Tensor:
    """Evaluate K masked passes and return only diagonal Q_kk values."""

    masked = build_multipass_parameters(parameters, spec)
    batch_size = states.shape[0]
    repeated_states = states[:, None, :].expand(-1, spec.num_actions, -1)
    all_outputs = q_network(
        repeated_states.reshape(batch_size * spec.num_actions, -1),
        masked.reshape(batch_size * spec.num_actions, -1),
    ).reshape(batch_size, spec.num_actions, spec.num_actions)
    return all_outputs.diagonal(dim1=1, dim2=2)
