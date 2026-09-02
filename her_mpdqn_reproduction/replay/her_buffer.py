"""Replay-buffer integration for ordinary future-strategy HER."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Callable, Mapping

import numpy as np
import torch

from agents.common import TransitionBatch, goal_observation_to_vector

from .goal_relabeling import (
    ComputeReward,
    FutureGoalRelabeler,
    GoalTransition,
    GoalValidator,
    PhaseAwareFutureGoalRelabeler,
)
from .replay_buffer import ReplayBuffer, ReplayBufferConfig


class HERReplayBuffer:
    """Expand complete episodes into original plus hindsight transitions."""

    def __init__(
        self,
        config: ReplayBufferConfig,
        *,
        her_k: int = 4,
        seed: int | None = None,
        relabeler: FutureGoalRelabeler | None = None,
        state_encoder: Callable[[Any], np.ndarray] = goal_observation_to_vector,
    ) -> None:
        self.replay = ReplayBuffer(config)
        if relabeler is not None and relabeler.her_k != her_k:
            raise ValueError("Provided relabeler her_k does not match buffer her_k")
        self.relabeler = relabeler or FutureGoalRelabeler(
            her_k=her_k, seed=config.seed if seed is None else seed)
        self.her_k = int(her_k)
        self.state_encoder = state_encoder
        self.total_original_added = 0
        self.total_relabelled_added = 0
        self.episodes_added = 0
        self.total_cross_phase_filtered = 0

    def __len__(self) -> int:
        return len(self.replay)

    def _add_transition(self, transition: GoalTransition) -> int:
        return self.replay.add(
            state=self.state_encoder(transition.observation),
            action=transition.action,
            action_parameters=transition.action_parameters,
            reward=transition.reward,
            next_state=self.state_encoder(transition.next_observation),
            terminated=transition.terminated,
            truncated=transition.truncated,
        )

    def add_episode(
        self,
        episode: Sequence[GoalTransition],
        compute_reward: ComputeReward,
        *,
        goal_validator: GoalValidator | None = None,
    ) -> dict[str, int]:
        if not episode:
            raise ValueError("Cannot add an empty episode")
        episode = tuple(episode)
        for transition in episode:
            self._add_transition(transition)
        relabeled = self.relabeler.relabel_episode(
            episode, compute_reward, goal_validator=goal_validator)
        for item in relabeled:
            self._add_transition(item.transition)
        original_count = len(episode)
        relabel_count = len(relabeled)
        self.total_original_added += original_count
        self.total_relabelled_added += relabel_count
        self.episodes_added += 1
        cross_phase_filtered = int(
            getattr(self.relabeler, "last_cross_phase_filtered", 0))
        self.total_cross_phase_filtered += cross_phase_filtered
        counts = {
            "original_transition_count": original_count,
            "her_relabel_count": relabel_count,
            "stored_transition_count": original_count + relabel_count,
            "total_her_relabel_count": self.total_relabelled_added,
        }
        if isinstance(self.relabeler, PhaseAwareFutureGoalRelabeler):
            counts.update({
                "cross_phase_goal_count_filtered": cross_phase_filtered,
                "total_cross_phase_goal_count_filtered": self.total_cross_phase_filtered,
            })
        return counts

    def sample(
        self,
        batch_size: int,
        *,
        device: torch.device | str = "cpu",
        bootstrap_on_truncation: bool = True,
    ) -> TransitionBatch:
        return self.replay.sample(
            batch_size,
            device=device,
            bootstrap_on_truncation=bootstrap_on_truncation,
        )

    def state_dict(self) -> dict[str, Any]:
        """Return replay storage, HER counters, and both sampling RNG states."""
        return {
            "format_version": 1,
            "buffer_type": type(self).__name__,
            "her_k": self.her_k,
            "replay": self.replay.state_dict(),
            "relabeler_rng_state": self.relabeler.rng.bit_generator.state,
            "total_original_added": self.total_original_added,
            "total_relabelled_added": self.total_relabelled_added,
            "episodes_added": self.episodes_added,
            "total_cross_phase_filtered": self.total_cross_phase_filtered,
            "relabeler_total_cross_phase_filtered": int(
                getattr(self.relabeler, "total_cross_phase_filtered", 0)),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if int(state.get("format_version", -1)) != 1:
            raise ValueError("Unsupported HER replay format version")
        if state.get("buffer_type") != type(self).__name__:
            raise ValueError("HER replay checkpoint type does not match this buffer")
        if int(state["her_k"]) != self.her_k:
            raise ValueError("HER replay checkpoint her_k does not match this buffer")
        self.replay.load_state_dict(state["replay"])
        self.relabeler.rng.bit_generator.state = dict(state["relabeler_rng_state"])
        self.total_original_added = int(state["total_original_added"])
        self.total_relabelled_added = int(state["total_relabelled_added"])
        self.episodes_added = int(state["episodes_added"])
        self.total_cross_phase_filtered = int(state["total_cross_phase_filtered"])
        if hasattr(self.relabeler, "total_cross_phase_filtered"):
            self.relabeler.total_cross_phase_filtered = int(
                state.get("relabeler_total_cross_phase_filtered", 0))

    @property
    def metrics(self) -> dict[str, int]:
        metrics = {
            "episodes_added": self.episodes_added,
            "original_transition_count": self.total_original_added,
            "her_relabel_count": self.total_relabelled_added,
            "stored_transition_count": len(self.replay),
            "total_transitions_seen": self.replay.total_added,
        }
        if isinstance(self.relabeler, PhaseAwareFutureGoalRelabeler):
            metrics["cross_phase_goal_count_filtered"] = self.total_cross_phase_filtered
        return metrics


class PhaseAwareHERReplayBuffer(HERReplayBuffer):
    """HER replay using same-phase future goals for relay/multi-stage tasks."""

    def __init__(
        self,
        config: ReplayBufferConfig,
        *,
        her_k: int = 4,
        seed: int | None = None,
        state_encoder: Callable[[Any], np.ndarray] = goal_observation_to_vector,
    ) -> None:
        relabel_seed = config.seed if seed is None else seed
        super().__init__(
            config,
            her_k=her_k,
            seed=relabel_seed,
            relabeler=PhaseAwareFutureGoalRelabeler(
                her_k=her_k, seed=relabel_seed),
            state_encoder=state_encoder,
        )
