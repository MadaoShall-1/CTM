"""Shared data structures and utilities for parameterized-action agents."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import random

import numpy as np
import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class ParameterizedActionSpec:
    """Map each discrete action to its continuous parameter-vector slice.

    Direct Navigation uses ``(1, 1)`` for MOVE and TURN. Relay Navigation uses
    ``(1, 1, 0)`` because CATCH has no continuous parameter in the paper.
    """

    parameter_sizes: tuple[int, ...]

    def __post_init__(self) -> None:
        sizes = tuple(int(size) for size in self.parameter_sizes)
        if not sizes or any(size < 0 for size in sizes):
            raise ValueError("parameter_sizes must be a non-empty sequence of non-negative integers")
        if sum(sizes) <= 0:
            raise ValueError("At least one action must have a continuous parameter")
        object.__setattr__(self, "parameter_sizes", sizes)

    @property
    def num_actions(self) -> int:
        return len(self.parameter_sizes)

    @property
    def total_parameter_dim(self) -> int:
        return sum(self.parameter_sizes)

    def parameter_slice(self, action: int) -> slice:
        if action < 0 or action >= self.num_actions:
            raise IndexError(f"Discrete action {action} is out of range")
        start = sum(self.parameter_sizes[:action])
        return slice(start, start + self.parameter_sizes[action])

    def selected_parameter(self, parameters: np.ndarray, action: int) -> np.ndarray:
        values = np.asarray(parameters, dtype=np.float32)
        if values.shape != (self.total_parameter_dim,):
            raise ValueError(
                f"Expected parameter vector {(self.total_parameter_dim,)}, got {values.shape}")
        selected = values[self.parameter_slice(action)]
        # The Gymnasium relay interface has a fixed dummy scalar for CATCH.
        return selected.copy() if selected.size else np.zeros(1, dtype=np.float32)

    def masks(self, *, device: torch.device | str | None = None) -> Tensor:
        masks = torch.zeros(
            self.num_actions, self.total_parameter_dim, dtype=torch.float32, device=device)
        for action in range(self.num_actions):
            masks[action, self.parameter_slice(action)] = 1.0
        return masks


@dataclass(frozen=True)
class ActionSelection:
    discrete_action: int
    selected_parameter: np.ndarray
    all_parameters: np.ndarray
    q_values: np.ndarray

    def environment_action(self) -> tuple[int, np.ndarray]:
        return self.discrete_action, self.selected_parameter.copy()


@dataclass
class TransitionBatch:
    """Tensor batch consumed by P-DQN and MP-DQN updates."""

    states: Tensor
    actions: Tensor
    action_parameters: Tensor
    rewards: Tensor
    next_states: Tensor
    dones: Tensor

    def __post_init__(self) -> None:
        batch_size = self.states.shape[0]
        if self.states.ndim != 2 or self.next_states.shape != self.states.shape:
            raise ValueError("states and next_states must have shape [batch, state_dim]")
        if self.action_parameters.ndim != 2 or self.action_parameters.shape[0] != batch_size:
            raise ValueError("action_parameters must have shape [batch, parameter_dim]")
        for name in ("actions", "rewards", "dones"):
            value = getattr(self, name)
            if value.numel() != batch_size:
                raise ValueError(f"{name} must contain one value per transition")

    @classmethod
    def from_numpy(
        cls,
        *,
        states: np.ndarray,
        actions: np.ndarray,
        action_parameters: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
        device: torch.device | str = "cpu",
    ) -> "TransitionBatch":
        return cls(
            states=torch.as_tensor(states, dtype=torch.float32, device=device),
            actions=torch.as_tensor(actions, dtype=torch.long, device=device).reshape(-1),
            action_parameters=torch.as_tensor(
                action_parameters, dtype=torch.float32, device=device),
            rewards=torch.as_tensor(rewards, dtype=torch.float32, device=device).reshape(-1),
            next_states=torch.as_tensor(next_states, dtype=torch.float32, device=device),
            dones=torch.as_tensor(dones, dtype=torch.float32, device=device).reshape(-1),
        )

    def to(self, device: torch.device | str) -> "TransitionBatch":
        return TransitionBatch(**{
            name: getattr(self, name).to(device)
            for name in (
                "states", "actions", "action_parameters", "rewards", "next_states", "dones")
        })


def goal_observation_to_vector(observation: Mapping[str, Any]) -> np.ndarray:
    """Flatten a GoalEnv observation for goal-conditioned value networks."""

    state = np.asarray(observation["observation"], dtype=np.float32).reshape(-1)
    goal = np.asarray(observation["desired_goal"], dtype=np.float32).reshape(-1)
    return np.concatenate([state, goal], dtype=np.float32)


def seed_everything(seed: int) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return np.random.default_rng(seed)


@torch.no_grad()
def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    if not 0.0 < tau <= 1.0:
        raise ValueError("tau must be in (0, 1]")
    for target_parameter, source_parameter in zip(target.parameters(), source.parameters()):
        target_parameter.lerp_(source_parameter, tau)


def save_checkpoint(path: str | Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(path)
