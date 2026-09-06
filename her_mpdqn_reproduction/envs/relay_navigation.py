"""Relay (supply pickup and delivery) UAV navigation environment."""

from __future__ import annotations

from typing import Any
import math

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .direct_navigation import DirectNavigationEnv
from .dynamics import CATCH, DynamicsConfig, UAVState, advance, distance


class RelayNavigationEnv(DirectNavigationEnv):
    """Two-stage sparse task: reach/catch a supply, then deliver it.

    The paper requires an explicit CATCH action.  ``require_catch_action=False``
    provides the automatic relay transition described by some benchmark
    formulations, but the paper-faithful default is ``True``.
    """

    def __init__(
        self,
        map_size: float = 2000.0,
        goal_radius: float = 100.0,
        relay_radius: float = 100.0,
        max_episode_steps: int = 100,
        min_start_goal_distance: float = 400.0,
        boundary_mode: str = "clip",
        dynamics: DynamicsConfig | None = None,
        require_catch_action: bool = True,
    ) -> None:
        super().__init__(
            map_size=map_size,
            goal_radius=goal_radius,
            max_episode_steps=max_episode_steps,
            min_start_goal_distance=min_start_goal_distance,
            boundary_mode=boundary_mode,
            dynamics=dynamics,
        )
        if relay_radius <= 0:
            raise ValueError("relay_radius must be positive")
        self.relay_radius = float(relay_radius)
        self.require_catch_action = bool(require_catch_action)
        self.action_space = spaces.Tuple(
            (spaces.Discrete(3), spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32))
        )
        low = np.asarray(
            [0.0, 0.0, self.dynamics.min_speed, -math.pi, 0.0, 0.0, 0.0, 0.0],
            dtype=np.float32,
        )
        high = np.asarray(
            [
                self.map_size,
                self.map_size,
                self.dynamics.max_speed,
                math.pi,
                math.sqrt(2.0) * self.map_size,
                math.sqrt(2.0) * self.map_size,
                float(self.max_episode_steps),
                1.0,
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Box(low, high, dtype=np.float32),
                "achieved_goal": spaces.Box(0.0, self.map_size, shape=(2,), dtype=np.float32),
                "desired_goal": spaces.Box(0.0, self.map_size, shape=(2,), dtype=np.float32),
            }
        )
        self.relay_goal = np.zeros(2, dtype=np.float32)
        self.final_goal = np.zeros(2, dtype=np.float32)
        self.phase = 0

    @property
    def current_phase(self) -> int:
        return self.phase

    @property
    def num_phases(self) -> int:
        return 2

    @property
    def current_goal(self) -> np.ndarray:
        return (self.relay_goal if self.phase == 0 else self.final_goal).copy()

    def _sample_three_points(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        for _ in range(10_000):
            points = self.np_random.uniform(0.0, self.map_size, size=(3, 2)).astype(np.float32)
            if (
                distance(points[0], points[1]) >= self.min_start_goal_distance
                and distance(points[1], points[2]) >= self.min_start_goal_distance
            ):
                return points[0], points[1], points[2]
        raise RuntimeError("Could not sample separated start, relay, and final goals")

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        # Seed Gymnasium without generating and discarding a direct-navigation task.
        gym.Env.reset(self, seed=seed)
        options = options or {}
        sampled_start, sampled_relay, sampled_final = self._sample_three_points()
        start = np.asarray(options.get("start", sampled_start), dtype=np.float32)
        relay = np.asarray(options.get("relay_goal", sampled_relay), dtype=np.float32)
        final = np.asarray(options.get("final_goal", sampled_final), dtype=np.float32)
        for name, point in (("start", start), ("relay_goal", relay), ("final_goal", final)):
            if point.shape != (2,) or np.any(point < 0) or np.any(point > self.map_size):
                raise ValueError(f"{name} must be a two-dimensional point inside the map")
        self.relay_goal = relay.copy()
        self.final_goal = final.copy()
        self.goal = self.final_goal  # Compatibility with the direct environment internals.
        self.phase = 0
        self.elapsed_steps = 0
        heading = float(options.get("heading", self.np_random.uniform(-math.pi, math.pi)))
        speed = float(options.get("speed", 0.0))
        self.state = UAVState(float(start[0]), float(start[1]), speed, heading)
        observation = self._get_obs()
        return observation, self._get_info(is_success=False, out_of_bounds=False)

    def _at_relay(self) -> bool:
        return bool(distance(self.state.position, self.relay_goal) <= self.relay_radius)

    def _at_goal(self) -> bool:
        return bool(self.phase == 1 and distance(self.state.position, self.final_goal) <= self.goal_radius)

    def _get_obs(self) -> dict[str, np.ndarray]:
        achieved_goal = self.state.position
        active_goal = self.current_goal
        vector = np.asarray(
            [
                self.state.x,
                self.state.y,
                self.state.speed,
                self.state.heading,
                float(distance(achieved_goal, self.final_goal)),
                float(distance(achieved_goal, self.relay_goal)),
                float(self.elapsed_steps),
                float(self.phase),
            ],
            dtype=np.float32,
        )
        return {
            "observation": vector,
            "achieved_goal": achieved_goal.copy(),
            "desired_goal": active_goal,
        }

    def _get_info(self, *, is_success: bool, out_of_bounds: bool) -> dict[str, Any]:
        return {
            "is_success": bool(is_success),
            "out_of_bounds": bool(out_of_bounds),
            "phase": self.phase,
            "num_phases": self.num_phases,
            "relay_reached": self._at_relay() if self.phase == 0 else True,
            "carrying_supply": self.phase == 1,
            "goal_radius": self.goal_radius,
        }

    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: dict[str, Any] | list[dict[str, Any]] | None,
    ) -> float | np.ndarray:
        """Recompute real/HER rewards with stage-event consistency.

        During phase 0, reaching a spatial goal is not enough when CATCH is
        required: the relabelled transition must itself execute CATCH near the
        relabelled goal.  Phase 1 remains an ordinary spatial delivery goal.
        This prevents HER from labelling MOVE/TURN as successful pickup events.
        """
        if not isinstance(info, dict) or not info.get("is_her", False):
            return super().compute_reward(achieved_goal, desired_goal, info)
        source_phase = int(info.get("her_source_phase", info.get("phase", 0)))
        if source_phase > 0 or not self.require_catch_action:
            return super().compute_reward(achieved_goal, desired_goal, info)
        before = np.asarray(
            info.get("her_achieved_goal_before", achieved_goal), dtype=np.float32)
        action = int(info.get("her_source_action", -1))
        success = action == CATCH and distance(before, desired_goal) <= self.relay_radius
        rewards = np.where(success, 0.0, -1.0)
        return float(rewards) if np.ndim(rewards) == 0 else rewards.astype(np.float32)

    def step(
        self, action: tuple[int, np.ndarray]
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        discrete_action, parameter = action
        if not self.action_space.spaces[0].contains(discrete_action):
            raise ValueError(f"Invalid discrete action: {discrete_action}")

        was_phase = self.phase
        at_relay_before_action = self._at_relay()
        if self.phase == 0 and self.require_catch_action and discrete_action == CATCH and at_relay_before_action:
            self.phase = 1

        self.state = advance(self.state, int(discrete_action), parameter, self.dynamics)
        self.elapsed_steps += 1
        boundary_hit = self._handle_boundary()
        if self.phase == 0 and not self.require_catch_action and self._at_relay():
            self.phase = 1

        is_success = self._at_goal()
        out_of_bounds = boundary_hit and self.boundary_mode == "terminate"
        terminated = bool(is_success or out_of_bounds)
        truncated = bool(self.elapsed_steps >= self.max_episode_steps and not terminated)
        observation = self._get_obs()
        # The paper gives no intermediate success reward: only final delivery is 0.
        reward = 0.0 if is_success else -1.0
        info = self._get_info(is_success=is_success, out_of_bounds=out_of_bounds)
        info["boundary_hit"] = boundary_hit
        info["phase_changed"] = was_phase != self.phase
        return observation, reward, terminated, truncated, info
