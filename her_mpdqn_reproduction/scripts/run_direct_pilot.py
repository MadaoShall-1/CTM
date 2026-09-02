"""Run a single-seed Direct Navigation convergence pilot for four baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate import evaluate_checkpoint, load_evaluation_config  # noqa: E402
from scripts.train import load_config, train  # noqa: E402


ALGORITHMS = ("pdqn", "mpdqn", "her_pdqn", "her_mpdqn")


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def window_mean(records, key, start, stop):
    values = [record[key] for record in records[start:stop] if key in record]
    return float(np.mean(values)) if values else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10000,
                        help="Safety cap; fixed-step pilots normally stop first on --steps")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-root", default="outputs/pilot/direct_50ep")
    args = parser.parse_args()
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    results = []
    for algorithm in ALGORITHMS:
        output_dir = output_root / algorithm / f"seed_{args.seed}"
        if (output_dir / "metrics.jsonl").exists():
            raise FileExistsError(
                f"Pilot output already exists; choose a new --output-root: {output_dir}")
        print(f"\n=== pilot train: direct/{algorithm}, seed {args.seed} ===")
        config = load_config(ROOT / "configs" / f"direct_{algorithm}.yaml")
        training_summary = train(config, overrides={
            "experiment": {
                "seed": args.seed,
                "output_dir": str(output_dir),
            },
            "training": {
                "episodes": args.episodes,
                "max_environment_steps": args.steps,
                "checkpoint_every": 0,
            },
        })
        records = read_jsonl(output_dir / "metrics.jsonl")
        window = min(10, len(records))
        losses = {
            "late_q_loss": window_mean(records, "q_loss", -window, None),
            "late_parameter_actor_loss": window_mean(
                records, "parameter_actor_loss", -window, None),
        }
        print(f"\n=== pilot evaluate: direct/{algorithm}, seed {args.seed} ===")
        checkpoint = output_dir / "last.pt"
        evaluation = evaluate_checkpoint(
            load_evaluation_config(checkpoint),
            checkpoint,
            episodes=args.eval_episodes,
            seed=10_000 + args.seed,
            output_dir=output_dir / "evaluation_last",
            save_trajectories=False,
        )
        result = {
            "algorithm": algorithm,
            "seed": args.seed,
            "training_episodes": training_summary["episodes"],
            "training_steps": training_summary["environment_steps"],
            "training_updates": training_summary["updates"],
            "early_success_rate_10": window_mean(records, "success", 0, window),
            "late_success_rate_10": window_mean(records, "success", -window, None),
            "training_success_rate": training_summary["success_rate_run"],
            **losses,
            "evaluation_success_rate": evaluation["success_rate"],
            "evaluation_mean_return": evaluation["mean_return"],
            "evaluation_mean_length": evaluation["mean_length"],
            "evaluation_out_of_bounds_rate": evaluation["out_of_bounds_rate"],
        }
        if not all(
            value is None or np.isfinite(value)
            for key, value in result.items()
            if isinstance(value, (float, int)) and key not in ("seed",)
        ):
            raise RuntimeError(f"Non-finite pilot metric for {algorithm}")
        results.append(result)

    artifact = {
        "environment": "direct",
        "episodes": args.episodes,
        "max_environment_steps": args.steps,
        "evaluation_episodes": args.eval_episodes,
        "seed": args.seed,
        "results": results,
    }
    summary_path = output_root / "pilot_summary.json"
    summary_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print("\n" + json.dumps(artifact, indent=2))
    print(f"DIRECT_PILOT_OK: {summary_path}")


if __name__ == "__main__":
    main()
