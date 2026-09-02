"""Fixed-capacity replay buffer with explicit Gymnasium terminal semantics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from agents.common import ActionSelection, TransitionBatch


@dataclass(frozen=True)
class ReplayBufferConfig:
    capacity: int
    state_dim: int
    parameter_dim: int
    num_actions: int
    seed: int = 0

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        if self.state_dim <= 0 or self.parameter_dim <= 0 or self.num_actions <= 0:
            raise ValueError("Replay dimensions must be positive")


class ReplayBuffer:
    """Preallocated circular replay memory.

    ``terminated`` and ``truncated`` are stored separately. By default samples
    bootstrap across time-limit truncation but not true MDP termination, which
    matches Gymnasium semantics and prevents a time limit from becoming a fake
    terminal state.
    """

    FORMAT_VERSION = 1

    def __init__(self, config: ReplayBufferConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        # Zero initialization keeps unused capacity deterministic and prevents
        # checkpointing uninitialized process-memory bytes.
        self.states = np.zeros((config.capacity, config.state_dim), dtype=np.float32)
        self.actions = np.zeros(config.capacity, dtype=np.int64)
        self.action_parameters = np.zeros(
            (config.capacity, config.parameter_dim), dtype=np.float32)
        self.rewards = np.zeros(config.capacity, dtype=np.float32)
        self.next_states = np.zeros((config.capacity, config.state_dim), dtype=np.float32)
        self.terminated = np.zeros(config.capacity, dtype=np.bool_)
        self.truncated = np.zeros(config.capacity, dtype=np.bool_)
        self.position = 0
        self.size = 0
        self.total_added = 0

    def __len__(self) -> int:
        return self.size

    @property
    def full(self) -> bool:
        return self.size == self.config.capacity

    def _vector(self, value: np.ndarray, dimension: int, name: str) -> np.ndarray:
        array = np.asarray(value, dtype=np.float32)
        if array.shape != (dimension,):
            raise ValueError(f"{name} must have shape {(dimension,)}, got {array.shape}")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} contains non-finite values")
        return array

    def add(
        self,
        *,
        state: np.ndarray,
        action: int,
        action_parameters: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
        truncated: bool,
    ) -> int:
        """Add one transition and return its physical storage index."""

        if isinstance(action, (bool, np.bool_)) or not isinstance(action, (int, np.integer)):
            raise TypeError("action must be an integer discrete-action index")
        action = int(action)
        if not 0 <= action < self.config.num_actions:
            raise ValueError(f"action must lie in [0, {self.config.num_actions})")
        reward = float(reward)
        if not np.isfinite(reward):
            raise ValueError("reward must be finite")
        if bool(terminated) and bool(truncated):
            raise ValueError("A transition cannot be both terminated and truncated")

        index = self.position
        self.states[index] = self._vector(state, self.config.state_dim, "state")
        self.actions[index] = action
        self.action_parameters[index] = self._vector(
            action_parameters, self.config.parameter_dim, "action_parameters")
        self.rewards[index] = reward
        self.next_states[index] = self._vector(
            next_state, self.config.state_dim, "next_state")
        self.terminated[index] = bool(terminated)
        self.truncated[index] = bool(truncated)

        self.position = (self.position + 1) % self.config.capacity
        self.size = min(self.size + 1, self.config.capacity)
        self.total_added += 1
        return index

    def add_selection(
        self,
        *,
        state: np.ndarray,
        selection: ActionSelection,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
        truncated: bool,
    ) -> int:
        """Add the complete parameter vector returned by an agent policy."""

        return self.add(
            state=state,
            action=selection.discrete_action,
            action_parameters=selection.all_parameters,
            reward=reward,
            next_state=next_state,
            terminated=terminated,
            truncated=truncated,
        )

    def sample_indices(self, batch_size: int) -> np.ndarray:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if batch_size > self.size:
            raise ValueError(
                f"Cannot sample {batch_size} transitions from a buffer containing {self.size}")
        return self.rng.choice(self.size, size=batch_size, replace=False)

    def batch_from_indices(
        self,
        indices: np.ndarray,
        *,
        device: torch.device | str = "cpu",
        bootstrap_on_truncation: bool = True,
    ) -> TransitionBatch:
        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        if indices.size == 0:
            raise ValueError("indices must not be empty")
        if np.any(indices < 0) or np.any(indices >= self.size):
            raise IndexError("Replay indices are outside the currently stored range")
        dones = self.terminated[indices].copy()
        if not bootstrap_on_truncation:
            dones = np.logical_or(dones, self.truncated[indices])
        return TransitionBatch.from_numpy(
            states=self.states[indices],
            actions=self.actions[indices],
            action_parameters=self.action_parameters[indices],
            rewards=self.rewards[indices],
            next_states=self.next_states[indices],
            dones=dones.astype(np.float32),
            device=device,
        )

    def sample(
        self,
        batch_size: int,
        *,
        device: torch.device | str = "cpu",
        bootstrap_on_truncation: bool = True,
    ) -> TransitionBatch:
        return self.batch_from_indices(
            self.sample_indices(batch_size),
            device=device,
            bootstrap_on_truncation=bootstrap_on_truncation,
        )

    def state_dict(self) -> dict[str, Any]:
        """Return a complete state, including circular position and RNG state."""

        return {
            "format_version": self.FORMAT_VERSION,
            "config": asdict(self.config),
            "position": self.position,
            "size": self.size,
            "total_added": self.total_added,
            "rng_state": self.rng.bit_generator.state,
            "states": self.states.copy(),
            "actions": self.actions.copy(),
            "action_parameters": self.action_parameters.copy(),
            "rewards": self.rewards.copy(),
            "next_states": self.next_states.copy(),
            "terminated": self.terminated.copy(),
            "truncated": self.truncated.copy(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if int(state["format_version"]) != self.FORMAT_VERSION:
            raise ValueError("Unsupported replay format version")
        if dict(state["config"]) != asdict(self.config):
            raise ValueError("Replay checkpoint configuration does not match this buffer")
        for name in (
            "states", "actions", "action_parameters", "rewards",
            "next_states", "terminated", "truncated"):
            source = np.asarray(state[name])
            destination = getattr(self, name)
            if source.shape != destination.shape:
                raise ValueError(f"Replay array {name} has incompatible shape")
            destination[...] = source.astype(destination.dtype, copy=False)
        self.position = int(state["position"])
        self.size = int(state["size"])
        self.total_added = int(state["total_added"])
        if not 0 <= self.position < self.config.capacity or not 0 <= self.size <= self.config.capacity:
            raise ValueError("Replay checkpoint has invalid position or size")
        self.rng.bit_generator.state = dict(state["rng_state"])

    def save(self, path: str | Path) -> None:
        """Save replay without pickle using compressed NumPy arrays plus JSON."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = self.state_dict()
        metadata = {
            key: state[key]
            for key in (
                "format_version", "config", "position", "size", "total_added", "rng_state")
        }
        arrays = {
            key: state[key]
            for key in (
                "states", "actions", "action_parameters", "rewards",
                "next_states", "terminated", "truncated")
        }
        np.savez_compressed(path, metadata=np.asarray(json.dumps(metadata)), **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "ReplayBuffer":
        with np.load(Path(path), allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
            buffer = cls(ReplayBufferConfig(**metadata["config"]))
            state = dict(metadata)
            for key in (
                "states", "actions", "action_parameters", "rewards",
                "next_states", "terminated", "truncated"):
                state[key] = archive[key]
            buffer.load_state_dict(state)
        return buffer
