"""Deterministic checkpoint evaluation for all four parameterized baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.wrappers import GoalObservationEncoder  # noqa: E402
from scripts.train import (  # noqa: E402
    append_jsonl,
    load_config,
    make_agent,
    make_environment,
    parameter_sizes_for_environment,
    validate_config,
)


def checkpoint_config_path(checkpoint: str | Path) -> Path:
    path = Path(checkpoint).resolve().parent / "config.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"No config.yaml beside checkpoint; pass --config explicitly: {path}")
    return path


def load_evaluation_config(
    checkpoint: str | Path,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    return load_config(config_path or checkpoint_config_path(checkpoint))


def save_trajectory(path: Path, transitions: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not transitions:
        raise ValueError("Cannot save an empty trajectory")
    keys = transitions[0].keys()
    episode = {
        key: np.asarray([transition[key] for transition in transitions])
        for key in keys
    }
    np.savez_compressed(path, **episode)


def evaluate_checkpoint(
    config: Mapping[str, Any],
    checkpoint: str | Path,
    *,
    episodes: int = 100,
    seed: int = 10_000,
    output_dir: str | Path | None = None,
    save_trajectories: bool = False,
) -> dict[str, Any]:
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    config = dict(config)
    validate_config(config)
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    env = make_environment(config)
    obs, _ = env.reset(seed=seed)
    encoder = GoalObservationEncoder(env.observation_space)
    parameter_sizes = parameter_sizes_for_environment(env)
    agent = make_agent(config, encoder.output_dim, parameter_sizes)
    agent.load(checkpoint)
    agent.parameter_actor.eval()
    agent.q_network.eval()

    output_dir = Path(output_dir) if output_dir is not None else checkpoint.parent / "evaluation"
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes_path = output_dir / "episodes.jsonl"
    # Evaluation directories are immutable per invocation from the caller's
    # perspective; overwrite stale episode records rather than append duplicates.
    episodes_path.write_text("", encoding="utf-8")
    trajectory_dir = output_dir / "trajectories"

    records: list[dict[str, Any]] = []
    for episode_index in range(1, episodes + 1):
        if episode_index > 1:
            obs, _ = env.reset()
        episode_return = 0.0
        length = 0
        relay_reached = False
        boundary_hit = False
        phase_switch_step = -1
        final_info: dict[str, Any] = {}
        trajectory: list[dict[str, Any]] = []
        while True:
            source_phase = env.current_phase
            selection = agent.select_action(
                encoder(obs), epsilon=0.0, parameter_noise_std=0.0)
            next_obs, reward, terminated, truncated, info = env.step(
                selection.environment_action())
            length += 1
            episode_return += reward
            if env.current_phase > source_phase and phase_switch_step < 0:
                phase_switch_step = length
            relay_reached = relay_reached or env.current_phase > 0
            boundary_hit = boundary_hit or bool(info.get("boundary_hit", False))
            if save_trajectories:
                trajectory.append({
                    "obs": np.asarray(obs["observation"], np.float32),
                    "action_discrete": np.int32(selection.discrete_action),
                    "action_parameter": np.float32(
                        selection.selected_parameter[0]
                        if selection.selected_parameter.size else 0.0),
                    "all_action_parameters": np.asarray(selection.all_parameters, np.float32),
                    "reward": np.float32(reward),
                    "next_obs": np.asarray(next_obs["observation"], np.float32),
                    "goal": np.asarray(obs["desired_goal"], np.float32),
                    "next_goal": np.asarray(next_obs["desired_goal"], np.float32),
                    "achieved_goal": np.asarray(obs["achieved_goal"], np.float32),
                    "next_achieved_goal": np.asarray(next_obs["achieved_goal"], np.float32),
                    "phase": np.int32(source_phase),
                    "next_phase": np.int32(env.current_phase),
                    "terminated": np.bool_(terminated),
                    "truncated": np.bool_(truncated),
                })
            obs = next_obs
            final_info = info
            if terminated or truncated:
                break
        record = {
            "episode": episode_index,
            "return": float(episode_return),
            "length": length,
            "success": float(bool(final_info.get("is_success", False))),
            "out_of_bounds": float(bool(final_info.get("out_of_bounds", False))),
            "boundary_hit": float(boundary_hit),
            "truncated": float(bool(truncated)),
            "relay_reached": float(relay_reached),
            "phase_switch_step": phase_switch_step,
        }
        records.append(record)
        append_jsonl(episodes_path, record)
        if save_trajectories:
            save_trajectory(trajectory_dir / f"episode_{episode_index:05d}.npz", trajectory)

    env.close()
    switched = [record["phase_switch_step"] for record in records if record["phase_switch_step"] >= 0]
    summary = {
        "algorithm": config["experiment"]["algorithm"],
        "environment": config["experiment"]["environment"],
        "checkpoint": str(checkpoint),
        "evaluation_seed": seed,
        "episodes": episodes,
        "mean_return": float(np.mean([record["return"] for record in records])),
        "std_return": float(np.std([record["return"] for record in records])),
        "mean_length": float(np.mean([record["length"] for record in records])),
        "success_rate": float(np.mean([record["success"] for record in records])),
        "out_of_bounds_rate": float(np.mean([record["out_of_bounds"] for record in records])),
        "boundary_hit_rate": float(np.mean([record["boundary_hit"] for record in records])),
        "truncation_rate": float(np.mean([record["truncated"] for record in records])),
        "relay_reached_rate": float(np.mean([record["relay_reached"] for record in records])),
        "mean_phase_switch_step": float(np.mean(switched)) if switched else None,
        "trajectories_saved": episodes if save_trajectories else 0,
        "output_dir": str(output_dir.resolve()),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    with (output_dir / "evaluation_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump({
            "episodes": episodes,
            "seed": seed,
            "checkpoint": str(checkpoint),
            "source_config": config.get("_config_path"),
        }, handle, sort_keys=False)
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--output-dir")
    parser.add_argument("--save-trajectories", action="store_true")
    args = parser.parse_args()
    config = load_evaluation_config(args.checkpoint, args.config)
    evaluate_checkpoint(
        config,
        args.checkpoint,
        episodes=args.episodes,
        seed=args.seed,
        output_dir=args.output_dir,
        save_trajectories=args.save_trajectories,
    )


if __name__ == "__main__":
    main()
