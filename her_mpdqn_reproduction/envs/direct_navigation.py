"""Goal-conditioned direct UAV navigation environment."""

from __future__ import annotations

from typing import Any
import math

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .dynamics import DynamicsConfig, MOVE, TURN, UAVState, advance, distance


class DirectNavigationEnv(gym.Env):
    """Sparse-reward navigation in a continuous two-dimensional square.

    Actions are ``(discrete_action, parameter)`` where action 0 is MOVE with
    normalized acceleration and action 1 is TURN with normalized angle.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        map_size: float = 2000.0,
        goal_radius: float = 100.0,
        max_episode_steps: int = 100,
        min_start_goal_distance: float = 400.0,
        boundary_mode: str = "clip",
        dynamics: DynamicsConfig | None = None,
    ) -> None:
        super().__init__()
        if map_size <= 0 or goal_radius <= 0 or goal_radius >= map_size / 2:
            raise ValueError("Invalid map size or goal radius")
        if max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")
        self.map_size = float(map_size)
        self.goal_radius = float(goal_radius)
        self.max_episode_steps = int(max_episode_steps)
        self.min_start_goal_distance = float(min_start_goal_distance)
        if boundary_mode not in {"clip", "terminate"}:
            raise ValueError("boundary_mode must be 'clip' or 'terminate'")
        self.boundary_mode = boundary_mode
        self.dynamics = dynamics or DynamicsConfig()

        self.action_space = spaces.Tuple(
            (spaces.Discrete(2), spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32))
        )
        observation_low = np.asarray(
            [0.0, 0.0, self.dynamics.min_speed, -math.pi, 0.0, 0.0],
            dtype=np.float32,
        )
        observation_high = np.asarray(
            [
                self.map_size,
                self.map_size,
                self.dynamics.max_speed,
                math.pi,
                math.sqrt(2.0) * self.map_size,
                float(self.max_episode_steps),
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Box(observation_low, observation_high, dtype=np.float32),
                "achieved_goal": spaces.Box(0.0, self.map_size, shape=(2,), dtype=np.float32),
                "desired_goal": spaces.Box(0.0, self.map_size, shape=(2,), dtype=np.float32),
            }
        )
        self.state = UAVState(0.0, 0.0, 0.0, 0.0)
        self.goal = np.zeros(2, dtype=np.float32)
        self.elapsed_steps = 0

    @property
    def current_phase(self) -> int:
        return 0

    @property
    def num_phases(self) -> int:
        return 1

    @property
    def current_goal(self) -> np.ndarray:
        return self.goal.copy()

    def _sample_start_and_goal(self) -> tuple[np.ndarray, np.ndarray]:
        for _ in range(10_000):
            start = self.np_random.uniform(0.0, self.map_size, size=2).astype(np.float32)
            goal = self.np_random.uniform(0.0, self.map_size, size=2).astype(np.float32)
            if float(distance(start, goal)) >= self.min_start_goal_distance:
                return start, goal
        raise RuntimeError("Could not sample start and goal with requested separation")

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}
        sampled_start, sampled_goal = self._sample_start_and_goal()
        start = np.asarray(options.get("start", sampled_start), dtype=np.float32)
        goal = np.asarray(options.get("goal", sampled_goal), dtype=np.float32)
        if start.shape != (2,) or goal.shape != (2,):
            raise ValueError("start and goal must be two-dimensional coordinates")
        if np.any(start < 0) or np.any(start > self.map_size):
            raise ValueError("start must lie inside the map")
        if np.any(goal < 0) or np.any(goal > self.map_size):
            raise ValueError("goal must lie inside the map")
        heading = float(options.get("heading", self.np_random.uniform(-math.pi, math.pi)))
        speed = float(options.get("speed", 0.0))
        self.state = UAVState(float(start[0]), float(start[1]), speed, heading)
        self.goal = goal.copy()
        self.elapsed_steps = 0
        observation = self._get_obs()
        return observation, self._get_info(is_success=self._at_goal(), out_of_bounds=False)

    def _get_obs(self) -> dict[str, np.ndarray]:
        achieved_goal = self.state.position
        vector = np.asarray(
            [
                self.state.x,
                self.state.y,
                self.state.speed,
                self.state.heading,
                float(distance(achieved_goal, self.goal)),
                float(self.elapsed_steps),
            ],
            dtype=np.float32,
        )
        return {
            "observation": vector,
            "achieved_goal": achieved_goal.copy(),
            "desired_goal": self.goal.copy(),
        }

    def _at_goal(self) -> bool:
        return bool(distance(self.state.position, self.goal) <= self.goal_radius)

    def _out_of_bounds(self) -> bool:
        position = self.state.position
        return bool(np.any(position < 0.0) or np.any(position > self.map_size))

    def _handle_boundary(self) -> bool:
        """Apply configured boundary behavior and return whether a hit occurred."""
        boundary_hit = self._out_of_bounds()
        if boundary_hit and self.boundary_mode == "clip":
            self.state.x = float(np.clip(self.state.x, 0.0, self.map_size))
            self.state.y = float(np.clip(self.state.y, 0.0, self.map_size))
        return boundary_hit

    def _get_info(self, *, is_success: bool, out_of_bounds: bool) -> dict[str, Any]:
        return {
            "is_success": bool(is_success),
            "out_of_bounds": bool(out_of_bounds),
            "phase": self.current_phase,
            "goal_radius": self.goal_radius,
        }

    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: dict[str, Any] | list[dict[str, Any]] | None,
    ) -> float | np.ndarray:
        del info
        rewards = np.where(distance(achieved_goal, desired_goal) <= self.goal_radius, 0.0, -1.0)
        return float(rewards) if np.ndim(rewards) == 0 else rewards.astype(np.float32)

    def step(
        self, action: tuple[int, np.ndarray]
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        if not self.action_space.contains(action):
            # Parameters outside [-1, 1] are deliberately accepted and clipped;
            # malformed discrete actions or shapes are still rejected by dynamics.
            discrete_action, parameter = action
            if not self.action_space.spaces[0].contains(discrete_action):
                raise ValueError(f"Invalid discrete action: {discrete_action}")
        else:
            discrete_action, parameter = action
        self.state = advance(self.state, int(discrete_action), parameter, self.dynamics)
        self.elapsed_steps += 1
        boundary_hit = self._handle_boundary()
        is_success = self._at_goal()
        out_of_bounds = boundary_hit and self.boundary_mode == "terminate"
        terminated = bool(is_success or out_of_bounds)
        truncated = bool(self.elapsed_steps >= self.max_episode_steps and not terminated)
        observation = self._get_obs()
        reward = float(self.compute_reward(observation["achieved_goal"], self.goal, None))
        info = self._get_info(is_success=is_success, out_of_bounds=out_of_bounds)
        info["boundary_hit"] = boundary_hit
        return observation, reward, terminated, truncated, info
