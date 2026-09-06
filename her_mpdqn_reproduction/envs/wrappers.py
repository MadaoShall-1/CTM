"""Algorithm-facing observation transformations kept outside environment physics."""

from __future__ import annotations

from collections.abc import Mapping

from gymnasium import spaces
import numpy as np


class GoalObservationEncoder:
    """Normalize the paper state, optionally followed by the HER goal.

    The paper's P-DQN/MP-DQN baselines consume only ``s``.  Goal-conditioned
    HER variants consume ``(s, g)``.  Relay environments expose an explicit
    phase for CT-WM trajectory analysis, but the paper baseline state does not
    include it, so callers can exclude that coordinate without removing it
    from the environment API.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        *,
        include_goal: bool = True,
        excluded_state_indices: tuple[int, ...] = (),
    ) -> None:
        state_space = observation_space["observation"]
        goal_space = observation_space["desired_goal"]
        if not isinstance(state_space, spaces.Box) or not isinstance(goal_space, spaces.Box):
            raise TypeError("GoalObservationEncoder requires Box state and goal spaces")
        state_low = state_space.low.reshape(-1)
        state_high = state_space.high.reshape(-1)
        excluded = {index % state_low.size for index in excluded_state_indices}
        self.state_indices = np.asarray(
            [index for index in range(state_low.size) if index not in excluded], dtype=np.int64)
        if not self.state_indices.size:
            raise ValueError("Cannot exclude every state coordinate")
        self.include_goal = bool(include_goal)
        lows = [state_low[self.state_indices]]
        highs = [state_high[self.state_indices]]
        if self.include_goal:
            lows.append(goal_space.low.reshape(-1))
            highs.append(goal_space.high.reshape(-1))
        self.low = np.concatenate(lows).astype(np.float32)
        self.high = np.concatenate(highs).astype(np.float32)
        if not np.all(np.isfinite(self.low)) or not np.all(np.isfinite(self.high)):
            raise ValueError("Observation normalization bounds must be finite")
        if np.any(self.high <= self.low):
            raise ValueError("Every observation bound must have positive range")

    @property
    def output_dim(self) -> int:
        return int(self.low.size)

    def __call__(self, observation: Mapping[str, np.ndarray]) -> np.ndarray:
        state = np.asarray(observation["observation"], dtype=np.float32).reshape(-1)
        parts = [state[self.state_indices]]
        if self.include_goal:
            parts.append(np.asarray(observation["desired_goal"], dtype=np.float32).reshape(-1))
        raw = np.concatenate(parts)
        if raw.shape != self.low.shape:
            raise ValueError(f"Expected flattened observation {self.low.shape}, got {raw.shape}")
        normalized = 2.0 * (raw - self.low) / (self.high - self.low) - 1.0
        return np.clip(normalized, -1.0, 1.0).astype(np.float32)
