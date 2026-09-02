"""Unified trainer for P-DQN, MP-DQN, HER-PDQN, and HER-MPDQN."""

from __future__ import annotations

import argparse
from collections import deque
import copy
import json
from pathlib import Path
import random
import sys
from typing import Any, Mapping

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents import ActionSelection, MPDQNAgent, PDQNAgent, PDQNConfig  # noqa: E402
from agents.common import save_checkpoint  # noqa: E402
from envs import DirectNavigationEnv, MultiRelayNavigationEnv, RelayNavigationEnv  # noqa: E402
from envs.wrappers import GoalObservationEncoder  # noqa: E402
from replay import (  # noqa: E402
    GoalTransition,
    HERReplayBuffer,
    PhaseAwareHERReplayBuffer,
    ReplayBuffer,
    ReplayBufferConfig,
)


ALGORITHMS = ("pdqn", "mpdqn", "her_pdqn", "her_mpdqn")
ENVIRONMENTS = ("direct", "relay", "multi_relay")
TRAINING_CHECKPOINT_VERSION = 1


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Training config must be a YAML mapping")
    config["_config_path"] = str(path.resolve())
    return config


def deep_update(base: dict[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_update(dict(result[key]), value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def validate_config(config: Mapping[str, Any]) -> None:
    for section in ("experiment", "env", "agent", "replay", "training"):
        if section not in config or not isinstance(config[section], Mapping):
            raise ValueError(f"Missing config section: {section}")
    algorithm = str(config["experiment"].get("algorithm"))
    environment = str(config["experiment"].get("environment"))
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm {algorithm!r}")
    if environment not in ENVIRONMENTS:
        raise ValueError(f"Unknown environment {environment!r}")
    phase_aware = bool(config["replay"].get("phase_aware", False))
    uses_her = algorithm.startswith("her_")
    if phase_aware and not uses_her:
        raise ValueError("phase_aware replay requires a HER algorithm")
    if phase_aware and environment not in ("relay", "multi_relay"):
        raise ValueError("phase_aware replay is only valid for staged relay tasks")
    if algorithm == "her_mpdqn" and environment in ("relay", "multi_relay") and not phase_aware:
        raise ValueError("Relay HER-MPDQN must enable phase_aware replay")
    if int(config["training"].get("episodes", 0)) <= 0:
        raise ValueError("training.episodes must be positive")
    max_steps = config["training"].get("max_environment_steps")
    if max_steps is not None and int(max_steps) <= 0:
        raise ValueError("training.max_environment_steps must be positive when set")


def resolve_device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return name


def make_environment(config: Mapping[str, Any]):
    name = config["experiment"]["environment"]
    values = dict(config["env"])
    if name == "direct":
        return DirectNavigationEnv(**values)
    if name == "relay":
        return RelayNavigationEnv(**values)
    return MultiRelayNavigationEnv(**values)


def parameter_sizes_for_environment(env) -> tuple[int, ...]:
    """Map the environment's discrete actions to their scalar parameter sizes."""
    if env.action_space[0].n == 2:
        return (1, 1)
    if env.action_space[0].n == 3:
        return (1, 1, 0)
    raise ValueError(f"Unsupported number of hybrid actions: {env.action_space[0].n}")


def make_agent(config: Mapping[str, Any], state_dim: int, parameter_sizes: tuple[int, ...]):
    values = config["agent"]
    agent_config = PDQNConfig(
        state_dim=state_dim,
        parameter_sizes=parameter_sizes,
        hidden_sizes=tuple(int(size) for size in values["hidden_sizes"]),
        gamma=float(values["gamma"]),
        tau=float(values["tau"]),
        q_learning_rate=float(values["q_learning_rate"]),
        parameter_learning_rate=float(values["parameter_learning_rate"]),
        gradient_clip_norm=float(values.get("gradient_clip_norm", 10.0)),
        seed=int(config["experiment"]["seed"]),
        device=resolve_device(str(values.get("device", "auto"))),
    )
    algorithm = config["experiment"]["algorithm"]
    return (MPDQNAgent if "mpdqn" in algorithm else PDQNAgent)(agent_config)


def make_replay(
    config: Mapping[str, Any],
    state_dim: int,
    parameter_dim: int,
    num_actions: int,
    encoder: GoalObservationEncoder,
):
    values = config["replay"]
    replay_config = ReplayBufferConfig(
        capacity=int(values["capacity"]),
        state_dim=state_dim,
        parameter_dim=parameter_dim,
        num_actions=num_actions,
        seed=int(config["experiment"]["seed"]),
    )
    algorithm = config["experiment"]["algorithm"]
    if not algorithm.startswith("her_"):
        return ReplayBuffer(replay_config)
    buffer_class = PhaseAwareHERReplayBuffer if values.get("phase_aware", False) else HERReplayBuffer
    return buffer_class(
        replay_config,
        her_k=int(values.get("her_k", 4)),
        state_encoder=encoder,
    )


def linear_schedule(start: float, end: float, duration: int, step: int) -> float:
    if duration <= 0:
        return float(end)
    fraction = min(max(step / duration, 0.0), 1.0)
    return float(start + fraction * (end - start))


def random_selection(agent) -> ActionSelection:
    parameters = agent.rng.uniform(
        -1.0, 1.0, size=agent.spec.total_parameter_dim).astype(np.float32)
    action = int(agent.rng.integers(agent.spec.num_actions))
    return ActionSelection(
        discrete_action=action,
        selected_parameter=agent.spec.selected_parameter(parameters, action),
        all_parameters=parameters,
        q_values=np.zeros(agent.spec.num_actions, dtype=np.float32),
    )


def append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(record), sort_keys=True) + "\n")


def resume_signature(config: Mapping[str, Any]) -> dict[str, Any]:
    """Configuration fields that must not change across a resumed run."""
    return {
        "experiment": {
            "algorithm": config["experiment"]["algorithm"],
            "environment": config["experiment"]["environment"],
            "seed": int(config["experiment"]["seed"]),
        },
        "env": copy.deepcopy(dict(config["env"])),
        "agent": copy.deepcopy(dict(config["agent"])),
        "replay": copy.deepcopy(dict(config["replay"])),
    }


def save_training_checkpoint(
    path: Path, *, agent, replay, env, config: Mapping[str, Any],
    trainer_state: Mapping[str, Any],
) -> None:
    payload = agent.checkpoint()
    payload.update({
        "training_checkpoint_version": TRAINING_CHECKPOINT_VERSION,
        "resume_signature": resume_signature(config),
        "replay_state": replay.state_dict(),
        "trainer_state": dict(trainer_state),
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "environment": copy.deepcopy(env.np_random.bit_generator.state),
        },
    })
    save_checkpoint(path, payload)


def load_training_checkpoint(
    path: Path, *, agent, replay, env, config: Mapping[str, Any],
) -> dict[str, Any]:
    payload = torch.load(path, map_location=agent.device, weights_only=False)
    if int(payload.get("training_checkpoint_version", -1)) != TRAINING_CHECKPOINT_VERSION:
        raise ValueError(
            "Checkpoint contains agent weights only and cannot resume training; "
            "use a checkpoint produced by the updated trainer")
    if payload.get("resume_signature") != resume_signature(config):
        raise ValueError(
            "Resume checkpoint is incompatible with the current environment, agent, replay, or seed config")
    agent.load(path)
    replay.load_state_dict(payload["replay_state"])
    rng = payload["rng_state"]
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch_cpu"].cpu())
    if torch.cuda.is_available() and rng.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all([state.cpu() for state in rng["torch_cuda"]])
    env.np_random.bit_generator.state = copy.deepcopy(rng["environment"])
    state = dict(payload["trainer_state"])
    if int(state["total_updates"]) != agent.update_steps:
        raise ValueError("Trainer and agent update counters disagree in checkpoint")
    return state


def truncate_metrics_to_checkpoint(
    path: Path, *, completed_episodes: int, total_steps: int,
) -> None:
    """Remove stale records after the selected checkpoint before appending."""
    if not path.exists():
        return
    retained = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        record = json.loads(line)
        if (int(record["episode"]) <= completed_episodes
                and int(record["environment_steps"]) <= total_steps):
            retained.append(json.dumps(record, sort_keys=True) + "\n")
    path.write_text("".join(retained), encoding="utf-8")


def train(
    config: Mapping[str, Any], *, overrides: Mapping[str, Any] | None = None,
    resume_from: str | Path | None = None,
) -> dict[str, Any]:
    config = deep_update(dict(config), overrides or {})
    validate_config(config)
    seed = int(config["experiment"]["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = make_environment(config)
    obs, _ = env.reset(seed=seed)
    encoder = GoalObservationEncoder(env.observation_space)
    parameter_sizes = parameter_sizes_for_environment(env)
    agent = make_agent(config, encoder.output_dim, parameter_sizes)
    replay = make_replay(
        config,
        encoder.output_dim,
        agent.spec.total_parameter_dim,
        agent.spec.num_actions,
        encoder,
    )

    output_dir = Path(config["experiment"]["output_dir"])
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    resolved_config = copy.deepcopy(config)
    resolved_config.pop("_config_path", None)

    training = config["training"]
    exploration = config.get("exploration", {})
    total_steps = 0
    total_updates = 0
    total_successes = 0
    completed_episodes = 0
    update_credit = 0
    success_window: deque[float] = deque(maxlen=100)
    last_losses: dict[str, float] = {}
    best_success_rate = -1.0

    if resume_from is None:
        metrics_path.write_text("", encoding="utf-8")
        start_episode = 1
        reset_before_first_episode = False
    else:
        resume_path = Path(resume_from).resolve()
        if not resume_path.exists():
            raise FileNotFoundError(resume_path)
        restored = load_training_checkpoint(
            resume_path, agent=agent, replay=replay, env=env, config=config)
        total_steps = int(restored["total_steps"])
        total_updates = int(restored["total_updates"])
        total_successes = int(restored["total_successes"])
        completed_episodes = int(restored["completed_episodes"])
        update_credit = int(restored["update_credit"])
        success_window.extend(float(value) for value in restored["success_window"])
        last_losses = dict(restored["last_losses"])
        best_success_rate = float(restored["best_success_rate"])
        start_episode = completed_episodes + 1
        reset_before_first_episode = True
        source_metrics = resume_path.parent / "metrics.jsonl"
        if (not metrics_path.exists() and source_metrics.exists()
                and source_metrics.resolve() != metrics_path.resolve()):
            metrics_path.write_text(
                source_metrics.read_text(encoding="utf-8"), encoding="utf-8")
        truncate_metrics_to_checkpoint(
            metrics_path, completed_episodes=completed_episodes, total_steps=total_steps)

    # Write only after a requested resume has passed compatibility checks, so a
    # bad resume command cannot overwrite the original run configuration.
    with (output_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(resolved_config, handle, sort_keys=False)

    def checkpoint_state() -> dict[str, Any]:
        return {
            "total_steps": total_steps,
            "total_updates": total_updates,
            "total_successes": total_successes,
            "completed_episodes": completed_episodes,
            "update_credit": update_credit,
            "success_window": list(success_window),
            "last_losses": dict(last_losses),
            "best_success_rate": best_success_rate,
        }

    def save_full_checkpoint(path: Path) -> None:
        save_training_checkpoint(
            path, agent=agent, replay=replay, env=env, config=config,
            trainer_state=checkpoint_state())

    for episode_index in range(start_episode, int(training["episodes"]) + 1):
        max_environment_steps = training.get("max_environment_steps")
        if max_environment_steps is not None and total_steps >= int(max_environment_steps):
            break
        if reset_before_first_episode or episode_index > start_episode:
            obs, _ = env.reset()
        reset_before_first_episode = False
        steps_before_episode = total_steps
        episode_transitions: list[GoalTransition] = []
        episode_return = 0.0
        episode_losses: list[dict[str, float]] = []
        final_info: dict[str, Any] = {}

        while True:
            state = encoder(obs)
            epsilon = linear_schedule(
                float(exploration.get("epsilon_start", 1.0)),
                float(exploration.get("epsilon_end", 0.05)),
                int(exploration.get("decay_steps", 100_000)),
                total_steps,
            )
            noise = linear_schedule(
                float(exploration.get("parameter_noise_start", 0.5)),
                float(exploration.get("parameter_noise_end", 0.05)),
                int(exploration.get("decay_steps", 100_000)),
                total_steps,
            )
            if total_steps < int(training["learning_starts"]):
                selection = random_selection(agent)
            else:
                selection = agent.select_action(
                    state, epsilon=epsilon, parameter_noise_std=noise)
            source_phase = env.current_phase
            next_obs, reward, terminated, truncated, info = env.step(
                selection.environment_action())
            budget_reached = (
                max_environment_steps is not None
                and total_steps + 1 >= int(max_environment_steps))
            if budget_reached and not terminated:
                truncated = True
                info = dict(info)
                info["budget_truncated"] = True
            episode_transitions.append(GoalTransition(
                observation=obs,
                action=selection.discrete_action,
                action_parameters=selection.all_parameters,
                reward=reward,
                next_observation=next_obs,
                terminated=terminated,
                truncated=truncated,
                info=info,
                phase=source_phase,
            ))
            episode_return += reward
            total_steps += 1
            obs = next_obs
            final_info = info
            if terminated or truncated:
                break

        her_counts = {
            "her_relabel_count": 0,
            "cross_phase_goal_count_filtered": 0,
        }
        if isinstance(replay, ReplayBuffer):
            for transition in episode_transitions:
                replay.add(
                    state=encoder(transition.observation),
                    action=transition.action,
                    action_parameters=transition.action_parameters,
                    reward=transition.reward,
                    next_state=encoder(transition.next_observation),
                    terminated=transition.terminated,
                    truncated=transition.truncated,
                )
        else:
            her_counts.update(replay.add_episode(
                episode_transitions,
                env.compute_reward,
                goal_validator=env.observation_space["desired_goal"].contains,
            ))

        replay_size = len(replay)
        eligible_steps = max(
            0,
            total_steps - max(steps_before_episode, int(training["learning_starts"])),
        )
        update_credit += eligible_steps
        update_every = int(training.get("update_every", 1))
        if update_every <= 0:
            raise ValueError("training.update_every must be positive")
        if replay_size >= int(training["batch_size"]):
            update_events = update_credit // update_every
            update_credit %= update_every
            updates = update_events * int(training.get("gradient_steps", 1))
            for _ in range(updates):
                batch = replay.sample(int(training["batch_size"]), device=agent.device)
                losses = agent.update(batch)
                episode_losses.append(losses)
                last_losses = losses
                total_updates += 1

        success = float(bool(final_info.get("is_success", False)))
        completed_episodes += 1
        total_successes += int(success)
        success_window.append(success)
        record: dict[str, Any] = {
            "episode": episode_index,
            "environment_steps": total_steps,
            "updates": total_updates,
            "return": float(episode_return),
            "length": len(episode_transitions),
            "success": success,
            "success_rate_100": float(np.mean(success_window)),
            "success_rate_run": total_successes / episode_index,
            "out_of_bounds": float(bool(final_info.get("out_of_bounds", False))),
            "boundary_hit": float(any(
                bool(t.info.get("boundary_hit", False)) for t in episode_transitions)),
            "truncated": float(episode_transitions[-1].truncated),
            "relay_reached": float(any(
                int(t.info.get("phase", t.phase)) > 0 for t in episode_transitions)),
            "epsilon": epsilon,
            "parameter_noise_std": noise,
            "replay_size": replay_size,
            **her_counts,
        }
        if episode_losses:
            for name in ("q_loss", "parameter_actor_loss", "mean_q", "mean_target_q"):
                record[name] = float(np.mean([loss[name] for loss in episode_losses]))
        append_jsonl(metrics_path, record)

        current_rate = float(np.mean(success_window))
        if current_rate > best_success_rate:
            best_success_rate = current_rate
            save_full_checkpoint(output_dir / "best.pt")
        checkpoint_every = int(training.get("checkpoint_every", 100))
        if checkpoint_every > 0 and episode_index % checkpoint_every == 0:
            save_full_checkpoint(output_dir / f"checkpoint_{episode_index}.pt")

    save_full_checkpoint(output_dir / "last.pt")
    env.close()
    summary = {
        "algorithm": config["experiment"]["algorithm"],
        "environment": config["experiment"]["environment"],
        "episodes": completed_episodes,
        "environment_steps": total_steps,
        "updates": total_updates,
        "success_rate_run": total_successes / completed_episodes,
        "replay_size": len(replay),
        "output_dir": str(output_dir.resolve()),
        "resumed_from": str(Path(resume_from).resolve()) if resume_from else None,
        **{name: last_losses.get(name) for name in ("q_loss", "parameter_actor_loss")},
    }
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--episodes", type=int, help="Override training.episodes")
    parser.add_argument("--steps", type=int, help="Override training.max_environment_steps")
    parser.add_argument("--output-dir", help="Override experiment.output_dir")
    parser.add_argument("--resume", help="Resume from a full training checkpoint")
    args = parser.parse_args()
    overrides: dict[str, Any] = {}
    if args.episodes is not None:
        overrides.setdefault("training", {})["episodes"] = args.episodes
    if args.steps is not None:
        overrides.setdefault("training", {})["max_environment_steps"] = args.steps
    if args.output_dir is not None:
        overrides.setdefault("experiment", {})["output_dir"] = args.output_dir
    train(load_config(args.config), overrides=overrides, resume_from=args.resume)


if __name__ == "__main__":
    main()
