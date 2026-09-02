"""Parameterized Deep Q-Network baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import copy

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .common import (
    ActionSelection,
    ParameterizedActionSpec,
    TransitionBatch,
    save_checkpoint,
    seed_everything,
    soft_update,
)
from .networks import ParameterActor, QNetwork


@dataclass(frozen=True)
class PDQNConfig:
    state_dim: int
    parameter_sizes: tuple[int, ...]
    hidden_sizes: tuple[int, ...] = (128, 64)
    gamma: float = 0.99
    tau: float = 0.005
    q_learning_rate: float = 1e-3
    parameter_learning_rate: float = 1e-4
    gradient_clip_norm: float = 10.0
    seed: int = 0
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.state_dim <= 0:
            raise ValueError("state_dim must be positive")
        ParameterizedActionSpec(self.parameter_sizes)
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if not 0.0 < self.tau <= 1.0:
            raise ValueError("tau must be in (0, 1]")
        if self.q_learning_rate <= 0 or self.parameter_learning_rate <= 0:
            raise ValueError("Learning rates must be positive")


class PDQNAgent:
    """P-DQN with joint action-parameter input to a shared Q-network."""

    algorithm_name = "pdqn"

    def __init__(self, config: PDQNConfig) -> None:
        self.config = config
        self.spec = ParameterizedActionSpec(config.parameter_sizes)
        self.device = torch.device(config.device)
        self.rng = seed_everything(config.seed)

        self.parameter_actor = ParameterActor(
            config.state_dim, self.spec.total_parameter_dim, config.hidden_sizes).to(self.device)
        self.q_network = QNetwork(
            config.state_dim,
            self.spec.total_parameter_dim,
            self.spec.num_actions,
            config.hidden_sizes,
        ).to(self.device)
        self.target_parameter_actor = copy.deepcopy(self.parameter_actor).to(self.device).eval()
        self.target_q_network = copy.deepcopy(self.q_network).to(self.device).eval()
        for network in (self.target_parameter_actor, self.target_q_network):
            for parameter in network.parameters():
                parameter.requires_grad_(False)

        self.q_optimizer = torch.optim.Adam(
            self.q_network.parameters(), lr=config.q_learning_rate)
        self.parameter_optimizer = torch.optim.Adam(
            self.parameter_actor.parameters(), lr=config.parameter_learning_rate)
        self.update_steps = 0

    def _q_values(self, network: nn.Module, states: Tensor, parameters: Tensor) -> Tensor:
        return network(states, parameters)

    @torch.no_grad()
    def q_values(self, states: np.ndarray | Tensor, parameters: np.ndarray | Tensor) -> np.ndarray:
        state_tensor = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        parameter_tensor = torch.as_tensor(parameters, dtype=torch.float32, device=self.device)
        if state_tensor.ndim == 1:
            state_tensor = state_tensor.unsqueeze(0)
        if parameter_tensor.ndim == 1:
            parameter_tensor = parameter_tensor.unsqueeze(0)
        return self._q_values(self.q_network, state_tensor, parameter_tensor).cpu().numpy()

    @torch.no_grad()
    def select_action(
        self,
        state: np.ndarray,
        *,
        epsilon: float = 0.0,
        parameter_noise_std: float = 0.0,
    ) -> ActionSelection:
        if not 0.0 <= epsilon <= 1.0 or parameter_noise_std < 0.0:
            raise ValueError("Invalid exploration settings")
        state_array = np.asarray(state, dtype=np.float32)
        if state_array.shape != (self.config.state_dim,):
            raise ValueError(f"Expected state shape {(self.config.state_dim,)}, got {state_array.shape}")
        state_tensor = torch.as_tensor(state_array, device=self.device).unsqueeze(0)
        parameters = self.parameter_actor(state_tensor).squeeze(0).cpu().numpy()
        if parameter_noise_std:
            parameters += self.rng.normal(0.0, parameter_noise_std, size=parameters.shape)
            parameters = np.clip(parameters, -1.0, 1.0).astype(np.float32)
        q_values = self.q_values(state_array, parameters)[0]
        if self.rng.random() < epsilon:
            discrete_action = int(self.rng.integers(self.spec.num_actions))
        else:
            discrete_action = int(np.argmax(q_values))
        return ActionSelection(
            discrete_action=discrete_action,
            selected_parameter=self.spec.selected_parameter(parameters, discrete_action),
            all_parameters=parameters.copy(),
            q_values=q_values.copy(),
        )

    def update(self, batch: TransitionBatch) -> dict[str, float]:
        batch = batch.to(self.device)
        if batch.states.shape[1] != self.config.state_dim:
            raise ValueError("Batch state dimension does not match agent configuration")
        if batch.action_parameters.shape[1] != self.spec.total_parameter_dim:
            raise ValueError("Batch parameter dimension does not match action specification")

        with torch.no_grad():
            next_parameters = self.target_parameter_actor(batch.next_states)
            next_q_values = self._q_values(
                self.target_q_network, batch.next_states, next_parameters)
            next_q = next_q_values.max(dim=1).values
            targets = batch.rewards + self.config.gamma * (1.0 - batch.dones) * next_q

        predicted_all = self._q_values(
            self.q_network, batch.states, batch.action_parameters)
        predicted = predicted_all.gather(1, batch.actions[:, None]).squeeze(1)
        q_loss = F.mse_loss(predicted, targets)
        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        q_gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.q_network.parameters(), self.config.gradient_clip_norm)
        self.q_optimizer.step()

        for parameter in self.q_network.parameters():
            parameter.requires_grad_(False)
        actor_parameters = self.parameter_actor(batch.states)
        actor_q_values = self._q_values(self.q_network, batch.states, actor_parameters)
        parameter_actor_loss = -actor_q_values.sum(dim=1).mean()
        self.parameter_optimizer.zero_grad(set_to_none=True)
        parameter_actor_loss.backward()
        parameter_gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.parameter_actor.parameters(), self.config.gradient_clip_norm)
        self.parameter_optimizer.step()
        for parameter in self.q_network.parameters():
            parameter.requires_grad_(True)

        soft_update(self.target_q_network, self.q_network, self.config.tau)
        soft_update(self.target_parameter_actor, self.parameter_actor, self.config.tau)
        self.update_steps += 1
        return {
            "q_loss": float(q_loss.detach().cpu()),
            "parameter_actor_loss": float(parameter_actor_loss.detach().cpu()),
            "q_gradient_norm": float(torch.as_tensor(q_gradient_norm).cpu()),
            "parameter_gradient_norm": float(torch.as_tensor(parameter_gradient_norm).cpu()),
            "mean_q": float(predicted.detach().mean().cpu()),
            "mean_target_q": float(targets.detach().mean().cpu()),
            "update_steps": float(self.update_steps),
        }

    def checkpoint(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm_name,
            "config": asdict(self.config),
            "parameter_actor": self.parameter_actor.state_dict(),
            "q_network": self.q_network.state_dict(),
            "target_parameter_actor": self.target_parameter_actor.state_dict(),
            "target_q_network": self.target_q_network.state_dict(),
            "parameter_optimizer": self.parameter_optimizer.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "update_steps": self.update_steps,
            "exploration_rng_state": self.rng.bit_generator.state,
        }

    def save(self, path: str | Path) -> None:
        save_checkpoint(path, self.checkpoint())

    def load(self, path: str | Path) -> None:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        if payload.get("algorithm") != self.algorithm_name:
            raise ValueError(
                f"Checkpoint algorithm {payload.get('algorithm')!r} does not match {self.algorithm_name!r}")
        for name in (
            "parameter_actor", "q_network", "target_parameter_actor", "target_q_network"):
            getattr(self, name).load_state_dict(payload[name])
        self.parameter_optimizer.load_state_dict(payload["parameter_optimizer"])
        self.q_optimizer.load_state_dict(payload["q_optimizer"])
        self.update_steps = int(payload["update_steps"])
        if "exploration_rng_state" in payload:
            self.rng.bit_generator.state = dict(payload["exploration_rng_state"])
