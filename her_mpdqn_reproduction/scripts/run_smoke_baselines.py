"""Run short training checks for all four baselines without full experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train import load_config, train  # noqa: E402


ALGORITHMS = ("pdqn", "mpdqn", "her_pdqn", "her_mpdqn")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", choices=("direct", "relay", "both"), default="both")
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()
    environments = ("direct", "relay") if args.environment == "both" else (args.environment,)
    for environment in environments:
        for algorithm in ALGORITHMS:
            config_path = ROOT / "configs" / f"{environment}_{algorithm}.yaml"
            print(f"\n=== smoke: {environment}/{algorithm} ===")
            train(load_config(config_path), overrides={
                "experiment": {
                    "output_dir": f"outputs/smoke/{environment}_{algorithm}",
                },
                "env": {"max_episode_steps": 8},
                "agent": {"hidden_sizes": [16], "device": "cpu"},
                "replay": {"capacity": 512},
                "training": {
                    "episodes": args.episodes,
                    "learning_starts": 0,
                    "batch_size": 4,
                    "update_every": 2,
                    "gradient_steps": 1,
                    "checkpoint_every": 0,
                },
                "exploration": {"decay_steps": 16},
            })
    print("\nALL_BASELINE_SMOKE_TESTS_OK")


if __name__ == "__main__":
    main()
