"""Evaluate phase-aware HER-MPDQN as the ordered relay horizon grows."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate import evaluate_checkpoint  # noqa: E402
from scripts.run_all_baselines import metric_totals, parse_csv  # noqa: E402
from scripts.train import load_config, train  # noqa: E402


def run_horizons(*, relay_counts: list[int], seeds: list[int], steps: int,
                 eval_episodes: int, output_root: Path) -> dict:
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    records = []
    base = load_config(ROOT / "configs" / "multi_relay_her_mpdqn.yaml")
    for num_relays in relay_counts:
        for seed in seeds:
            run_dir = output_root / f"relays_{num_relays}" / f"seed_{seed}"
            training = train(base, overrides={
                "experiment": {"seed": seed, "output_dir": str(run_dir)},
                "env": {
                    "num_relays": num_relays,
                    "max_episode_steps": max(100, 100 * (num_relays + 1)),
                },
                "training": {
                    "max_environment_steps": steps,
                    "learning_starts": min(2_000, max(128, steps // 3)),
                },
            })
            evaluation = evaluate_checkpoint(
                load_config(run_dir / "config.yaml"), run_dir / "last.pt",
                episodes=eval_episodes, seed=200_000 + seed,
                output_dir=run_dir / "evaluation",
            )
            totals = metric_totals(run_dir / "metrics.jsonl")
            record = {
                "num_relays": num_relays,
                "num_phases": num_relays + 1,
                "seed": seed,
                "environment_steps": training["environment_steps"],
                "success_rate": evaluation["success_rate"],
                "mean_return": evaluation["mean_return"],
                "mean_length": evaluation["mean_length"],
                "relay_reached_rate": evaluation["relay_reached_rate"],
                "q_loss": totals.get("q_loss"),
                "parameter_actor_loss": totals.get("parameter_actor_loss"),
                "her_relabel_count": totals["her_relabel_count"],
                "run_dir": str(run_dir),
            }
            records.append(record)
            (output_root / "per_run.jsonl").write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
                encoding="utf-8")
    aggregate = []
    for num_relays in relay_counts:
        group = [row for row in records if row["num_relays"] == num_relays]
        result = {"num_relays": num_relays, "num_phases": num_relays + 1, "num_seeds": len(group)}
        for key in ("success_rate", "mean_return", "mean_length", "relay_reached_rate"):
            values = np.asarray([row[key] for row in group], dtype=float)
            result[f"{key}_mean"] = float(values.mean())
            result[f"{key}_std"] = float(values.std())
        aggregate.append(result)
    summary = {"steps": steps, "eval_episodes": eval_episodes, "runs": records, "aggregate": aggregate}
    (output_root / "horizon_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    with (output_root / "horizon_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate[0]))
        writer.writeheader()
        writer.writerows(aggregate)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--relay-counts", default="0,1,2,4,8")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--steps", type=int, default=5_000)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--output-root", default="outputs/horizon_3seed_5000")
    args = parser.parse_args()
    root = Path(args.output_root)
    if not root.is_absolute():
        root = ROOT / root
    summary = run_horizons(
        relay_counts=parse_csv(args.relay_counts, int),
        seeds=parse_csv(args.seeds, int), steps=args.steps,
        eval_episodes=args.eval_episodes, output_root=root,
    )
    print(json.dumps(summary["aggregate"], indent=2))


if __name__ == "__main__":
    main()
