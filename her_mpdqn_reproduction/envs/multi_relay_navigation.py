"""Configurable long-horizon multi-relay UAV navigation."""

from __future__ import annotations

from typing import Any
import math

from gymnasium import spaces
import gymnasium as gym
import numpy as np

from .direct_navigation import DirectNavigationEnv
from .dynamics import CATCH, DynamicsConfig, UAVState, advance, distance


class MultiRelayNavigationEnv(DirectNavigationEnv):
    """Navigate through N ordered relay points and then the final goal.

    Each relay requires a valid CATCH by default. Rewards remain -1 until the
    final goal is reached; increasing N never changes reward density.
    """

    def __init__(
        self,
        num_relays: int = 1,
        map_size: float = 2000.0,
        goal_radius: float = 100.0,
        relay_radius: float = 100.0,
        max_episode_steps: int = 100,
        min_start_goal_distance: float = 400.0,
        boundary_mode: str = "clip",
        dynamics: DynamicsConfig | None = None,
        require_catch_action: bool = True,
    ) -> None:
        if int(num_relays) != num_relays or num_relays < 0:
            raise ValueError("num_relays must be a non-negative integer")
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
        self.num_relays = int(num_relays)
        self.relay_radius = float(relay_radius)
        self.require_catch_action = bool(require_catch_action)
        self.relay_goals = np.zeros((self.num_relays, 2), dtype=np.float32)
        self.final_goal = np.zeros(2, dtype=np.float32)
        self.phase = 0
        num_actions = 3 if self.num_relays else 2
        self.action_space = spaces.Tuple((
            spaces.Discrete(num_actions),
            spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32),
        ))
        # [x,y,v,theta,d_final,d_relay_1..d_relay_N,steps,phase]
        low = np.asarray(
            [0.0, 0.0, self.dynamics.min_speed, -math.pi, 0.0]
            + [0.0] * self.num_relays + [0.0, 0.0], dtype=np.float32)
        high = np.asarray(
            [self.map_size, self.map_size, self.dynamics.max_speed, math.pi,
             math.sqrt(2.0) * self.map_size]
            + [math.sqrt(2.0) * self.map_size] * self.num_relays
            + [float(self.max_episode_steps), float(max(1, self.num_relays))], dtype=np.float32)
        self.observation_space = spaces.Dict({
            "observation": spaces.Box(low, high, dtype=np.float32),
            "achieved_goal": spaces.Box(0.0, self.map_size, shape=(2,), dtype=np.float32),
            "desired_goal": spaces.Box(0.0, self.map_size, shape=(2,), dtype=np.float32),
        })

    @property
    def current_phase(self) -> int:
        return self.phase

    @property
    def num_phases(self) -> int:
        return self.num_relays + 1

    @property
    def current_goal(self) -> np.ndarray:
        if self.phase < self.num_relays:
            return self.relay_goals[self.phase].copy()
        return self.final_goal.copy()

    def _sample_route(self) -> np.ndarray:
        count = self.num_relays + 2  # start + relays + final
        for _ in range(10_000):
            points = self.np_random.uniform(
                0.0, self.map_size, size=(count, 2)).astype(np.float32)
            segment_distances = distance(points[:-1], points[1:])
            if np.all(segment_distances >= self.min_start_goal_distance):
                return points
        raise RuntimeError("Could not sample a route with requested segment separation")

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        gym.Env.reset(self, seed=seed)
        options = options or {}
        sampled = self._sample_route()
        start = np.asarray(options.get("start", sampled[0]), dtype=np.float32)
        relays = np.asarray(options.get("relay_goals", sampled[1:-1]), dtype=np.float32)
        if self.num_relays == 0:
            relays = relays.reshape(0, 2)
        final = np.asarray(options.get("final_goal", sampled[-1]), dtype=np.float32)
        if start.shape != (2,) or final.shape != (2,) or relays.shape != (self.num_relays, 2):
            raise ValueError("start/final must be [2] and relay_goals must be [num_relays, 2]")
        for name, points in (("start", start[None]), ("relay_goals", relays), ("final_goal", final[None])):
            if np.any(points < 0) or np.any(points > self.map_size):
                raise ValueError(f"{name} must lie inside the map")
        self.relay_goals = relays.copy()
        self.final_goal = final.copy()
        self.goal = self.final_goal
        self.phase = 0
        self.elapsed_steps = 0
        self.state = UAVState(
            float(start[0]), float(start[1]),
            float(options.get("speed", 0.0)),
            float(options.get("heading", self.np_random.uniform(-math.pi, math.pi))),
        )
        obs = self._get_obs()
        return obs, self._get_info(is_success=False, out_of_bounds=False)

    def _at_current_relay(self) -> bool:
        return bool(
            self.phase < self.num_relays
            and distance(self.state.position, self.relay_goals[self.phase]) <= self.relay_radius)

    def _at_goal(self) -> bool:
        return bool(
            self.phase == self.num_relays
            and distance(self.state.position, self.final_goal) <= self.goal_radius)

    def _get_obs(self) -> dict[str, np.ndarray]:
        achieved = self.state.position
        relay_distances = [float(distance(achieved, goal)) for goal in self.relay_goals]
        vector = np.asarray([
            self.state.x, self.state.y, self.state.speed, self.state.heading,
            float(distance(achieved, self.final_goal)),
            *relay_distances,
            float(self.elapsed_steps), float(self.phase),
        ], dtype=np.float32)
        return {
            "observation": vector,
            "achieved_goal": achieved.copy(),
            "desired_goal": self.current_goal,
        }

    def _get_info(self, *, is_success: bool, out_of_bounds: bool) -> dict[str, Any]:
        return {
            "is_success": bool(is_success),
            "out_of_bounds": bool(out_of_bounds),
            "phase": self.phase,
            "num_phases": self.num_phases,
            "relays_completed": min(self.phase, self.num_relays),
            "all_relays_completed": self.phase == self.num_relays,
            "goal_radius": self.goal_radius,
        }

    def compute_reward(self, achieved_goal, desired_goal, info):
        """Use CATCH-consistent virtual success for unfinished relay stages."""
        if not isinstance(info, dict) or not info.get("is_her", False):
            return super().compute_reward(achieved_goal, desired_goal, info)
        source_phase = int(info.get("her_source_phase", info.get("phase", 0)))
        if source_phase >= self.num_relays or not self.require_catch_action:
            return super().compute_reward(achieved_goal, desired_goal, info)
        before = np.asarray(
            info.get("her_achieved_goal_before", achieved_goal), dtype=np.float32)
        success = (
            int(info.get("her_source_action", -1)) == CATCH
            and distance(before, desired_goal) <= self.relay_radius
        )
        rewards = np.where(success, 0.0, -1.0)
        return float(rewards) if np.ndim(rewards) == 0 else rewards.astype(np.float32)

    def step(self, action):
        discrete_action, parameter = action
        if not self.action_space.spaces[0].contains(discrete_action):
            raise ValueError(f"Invalid discrete action: {discrete_action}")
        previous_phase = self.phase
        if (
            self.require_catch_action and discrete_action == CATCH
            and self._at_current_relay()
        ):
            self.phase += 1
        self.state = advance(self.state, int(discrete_action), parameter, self.dynamics)
        self.elapsed_steps += 1
        boundary_hit = self._handle_boundary()
        if not self.require_catch_action and self._at_current_relay():
            self.phase += 1
        success = self._at_goal()
        out_of_bounds = boundary_hit and self.boundary_mode == "terminate"
        terminated = bool(success or out_of_bounds)
        truncated = bool(self.elapsed_steps >= self.max_episode_steps and not terminated)
        obs = self._get_obs()
        info = self._get_info(is_success=success, out_of_bounds=out_of_bounds)
        info["boundary_hit"] = boundary_hit
        info["phase_changed"] = self.phase != previous_phase
        return obs, 0.0 if success else -1.0, terminated, truncated, info
