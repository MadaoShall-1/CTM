"""Algorithm-facing observation transformations kept outside environment physics."""

from __future__ import annotations

from collections.abc import Mapping

from gymnasium import spaces
import numpy as np


class GoalObservationEncoder:
    """Normalize state and desired goal to [-1, 1] and concatenate them.

    HER calls the same encoder after replacing ``desired_goal``, ensuring real
    and hindsight transitions use identical state preprocessing.
    """

    def __init__(self, observation_space: spaces.Dict) -> None:
        state_space = observation_space["observation"]
        goal_space = observation_space["desired_goal"]
        if not isinstance(state_space, spaces.Box) or not isinstance(goal_space, spaces.Box):
            raise TypeError("GoalObservationEncoder requires Box state and goal spaces")
        self.low = np.concatenate((state_space.low.reshape(-1), goal_space.low.reshape(-1))).astype(
            np.float32)
        self.high = np.concatenate((state_space.high.reshape(-1), goal_space.high.reshape(-1))).astype(
            np.float32)
        if not np.all(np.isfinite(self.low)) or not np.all(np.isfinite(self.high)):
            raise ValueError("Observation normalization bounds must be finite")
        if np.any(self.high <= self.low):
            raise ValueError("Every observation bound must have positive range")

    @property
    def output_dim(self) -> int:
        return int(self.low.size)

    def __call__(self, observation: Mapping[str, np.ndarray]) -> np.ndarray:
        raw = np.concatenate((
            np.asarray(observation["observation"], dtype=np.float32).reshape(-1),
            np.asarray(observation["desired_goal"], dtype=np.float32).reshape(-1),
        ))
        if raw.shape != self.low.shape:
            raise ValueError(f"Expected flattened observation {self.low.shape}, got {raw.shape}")
        normalized = 2.0 * (raw - self.low) / (self.high - self.low) - 1.0
        return np.clip(normalized, -1.0, 1.0).astype(np.float32)
