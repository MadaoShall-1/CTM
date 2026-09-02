"""Goal-transition contracts and ordinary future-strategy HER relabeling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np


GoalObservation = Mapping[str, np.ndarray]
ComputeReward = Callable[[np.ndarray, np.ndarray, Any], float | np.ndarray]
GoalValidator = Callable[[np.ndarray], bool]


def copy_goal_observation(observation: GoalObservation) -> dict[str, np.ndarray]:
    required = ("observation", "achieved_goal", "desired_goal")
    missing = [key for key in required if key not in observation]
    if missing:
        raise KeyError(f"Goal observation is missing keys: {missing}")
    copied = {
        key: np.asarray(value, dtype=np.float32).copy()
        for key, value in observation.items()
    }
    if copied["achieved_goal"].shape != copied["desired_goal"].shape:
        raise ValueError("achieved_goal and desired_goal must have matching shapes")
    for key, value in copied.items():
        if not np.all(np.isfinite(value)):
            raise ValueError(f"Observation field {key} contains non-finite values")
    return copied


@dataclass(frozen=True)
class GoalTransition:
    """One environment step retained in episode order for HER.

    ``phase`` is the task phase at the source state, before executing the
    action. This convention keeps the relay-acquiring CATCH transition in
    phase 0 even though its next observation is already in phase 1.
    """

    observation: GoalObservation
    action: int
    action_parameters: np.ndarray
    reward: float
    next_observation: GoalObservation
    terminated: bool
    truncated: bool
    info: Mapping[str, Any] = field(default_factory=dict)
    phase: int = 0

    def __post_init__(self) -> None:
        observation = copy_goal_observation(self.observation)
        next_observation = copy_goal_observation(self.next_observation)
        if observation["achieved_goal"].shape != next_observation["achieved_goal"].shape:
            raise ValueError("Goal shape must remain constant within a transition")
        parameters = np.asarray(self.action_parameters, dtype=np.float32).copy()
        if parameters.ndim != 1 or not np.all(np.isfinite(parameters)):
            raise ValueError("action_parameters must be a finite one-dimensional vector")
        if not np.isfinite(float(self.reward)):
            raise ValueError("reward must be finite")
        if self.terminated and self.truncated:
            raise ValueError("A transition cannot be both terminated and truncated")
        object.__setattr__(self, "observation", observation)
        object.__setattr__(self, "next_observation", next_observation)
        object.__setattr__(self, "action", int(self.action))
        object.__setattr__(self, "action_parameters", parameters)
        object.__setattr__(self, "reward", float(self.reward))
        object.__setattr__(self, "terminated", bool(self.terminated))
        object.__setattr__(self, "truncated", bool(self.truncated))
        object.__setattr__(self, "info", dict(self.info))
        object.__setattr__(self, "phase", int(self.phase))


@dataclass(frozen=True)
class RelabeledTransition:
    transition: GoalTransition
    source_index: int
    future_index: int
    relabeled_goal: np.ndarray


class FutureGoalRelabeler:
    """Sample up to ``her_k`` unique achieved goals from t+1..T."""

    def __init__(self, her_k: int = 4, seed: int = 0) -> None:
        if her_k < 0:
            raise ValueError("her_k must be non-negative")
        self.her_k = int(her_k)
        self.rng = np.random.default_rng(seed)

    def _future_candidates(
        self,
        episode: Sequence[GoalTransition],
        source_index: int,
        goal_validator: GoalValidator | None,
    ) -> list[int]:
        candidates = []
        # transition j ends at state s_(j+1), so j >= source_index is future
        # relative to the source state s_t.
        for future_index in range(source_index, len(episode)):
            goal = episode[future_index].next_observation["achieved_goal"]
            if goal_validator is None or bool(goal_validator(goal)):
                candidates.append(future_index)
        return candidates

    def relabel_episode(
        self,
        episode: Sequence[GoalTransition],
        compute_reward: ComputeReward,
        *,
        goal_validator: GoalValidator | None = None,
    ) -> list[RelabeledTransition]:
        if not episode or self.her_k == 0:
            return []
        relabeled: list[RelabeledTransition] = []
        for source_index, source in enumerate(episode):
            candidates = self._future_candidates(episode, source_index, goal_validator)
            count = min(self.her_k, len(candidates))
            if count == 0:
                continue
            sampled = self.rng.choice(candidates, size=count, replace=False)
            for future_index in np.atleast_1d(sampled):
                future_index = int(future_index)
                goal = np.asarray(
                    episode[future_index].next_observation["achieved_goal"],
                    dtype=np.float32,
                ).copy()
                observation = copy_goal_observation(source.observation)
                next_observation = copy_goal_observation(source.next_observation)
                observation["desired_goal"] = goal.copy()
                next_observation["desired_goal"] = goal.copy()
                reward_value = compute_reward(
                    next_observation["achieved_goal"], goal, dict(source.info))
                reward_array = np.asarray(reward_value, dtype=np.float32)
                if reward_array.shape != ():
                    raise ValueError("compute_reward must return a scalar for one transition")
                reward = float(reward_array)
                if not np.isfinite(reward):
                    raise ValueError("compute_reward returned a non-finite reward")
                virtual_success = reward == 0.0
                transition = GoalTransition(
                    observation=observation,
                    action=source.action,
                    action_parameters=source.action_parameters,
                    reward=reward,
                    next_observation=next_observation,
                    # Sparse goal completion is terminal. A time-limit
                    # truncation remains a truncation only when the relabeled
                    # goal was not achieved at this transition.
                    terminated=virtual_success,
                    truncated=source.truncated and not virtual_success,
                    info={**source.info, "is_success": virtual_success, "is_her": True},
                    phase=source.phase,
                )
                relabeled.append(RelabeledTransition(
                    transition=transition,
                    source_index=source_index,
                    future_index=future_index,
                    relabeled_goal=goal,
                ))
        return relabeled


class PhaseAwareFutureGoalRelabeler(FutureGoalRelabeler):
    """GSM reconstruction: future goals must come from the source phase."""

    def __init__(self, her_k: int = 4, seed: int = 0) -> None:
        super().__init__(her_k=her_k, seed=seed)
        self.last_cross_phase_filtered = 0
        self.total_cross_phase_filtered = 0

    @staticmethod
    def _validate_phases(episode: Sequence[GoalTransition]) -> None:
        phases = [transition.phase for transition in episode]
        if any(phase < 0 for phase in phases):
            raise ValueError("Episode phases must be non-negative")
        for previous, current in zip(phases, phases[1:]):
            if current < previous:
                raise ValueError("Episode phases must be monotonically non-decreasing")
            if current > previous + 1:
                raise ValueError("Episode phases cannot skip a stage")

    def _future_candidates(
        self,
        episode: Sequence[GoalTransition],
        source_index: int,
        goal_validator: GoalValidator | None,
    ) -> list[int]:
        valid_future = super()._future_candidates(
            episode, source_index, goal_validator)
        source_phase = episode[source_index].phase
        same_phase = [
            index for index in valid_future
            if episode[index].phase == source_phase
        ]
        filtered = len(valid_future) - len(same_phase)
        self.last_cross_phase_filtered += filtered
        self.total_cross_phase_filtered += filtered
        return same_phase

    def relabel_episode(
        self,
        episode: Sequence[GoalTransition],
        compute_reward: ComputeReward,
        *,
        goal_validator: GoalValidator | None = None,
    ) -> list[RelabeledTransition]:
        self.last_cross_phase_filtered = 0
        self._validate_phases(episode)
        return super().relabel_episode(
            episode, compute_reward, goal_validator=goal_validator)
