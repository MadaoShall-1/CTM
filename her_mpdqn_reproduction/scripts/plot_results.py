"""Plot learning curves and aggregate evaluation results from run_all_baselines.py."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


def read_runs(root: Path) -> list[dict]:
    runs = []
    for path in root.rglob("metrics.jsonl"):
        config_path = path.parent / "config.yaml"
        if not config_path.exists():
            continue
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        metrics = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        if metrics:
            runs.append({"config": config, "metrics": metrics})
    return runs


def plot_learning_curves(root: Path, output: Path) -> list[Path]:
    runs = read_runs(root)
    created: list[Path] = []
    for environment in sorted({r["config"]["experiment"]["environment"] for r in runs}):
        env_runs = [r for r in runs if r["config"]["experiment"]["environment"] == environment]
        fig, ax = plt.subplots(figsize=(8, 5))
        for algorithm in sorted({r["config"]["experiment"]["algorithm"] for r in env_runs}):
            group = [r for r in env_runs if r["config"]["experiment"]["algorithm"] == algorithm]
            max_step = min(r["metrics"][-1]["environment_steps"] for r in group)
            grid = np.linspace(0, max_step, 101)
            curves = []
            for run in group:
                x = np.asarray([m["environment_steps"] for m in run["metrics"]], float)
                y = np.asarray([m["success_rate_100"] for m in run["metrics"]], float)
                curves.append(np.interp(grid, x, y, left=y[0], right=y[-1]))
            values = np.asarray(curves)
            mean, std = values.mean(0), values.std(0)
            ax.plot(grid, mean, label=algorithm.upper().replace("_", "-"))
            ax.fill_between(grid, mean - std, mean + std, alpha=0.18)
        ax.set(title=f"{environment.title()} navigation", xlabel="Environment steps", ylabel="Training success rate (last 100 episodes)", ylim=(-0.02, 1.02))
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        path = output / f"{environment}_learning_curve.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        created.append(path)
    return created


def plot_evaluation(root: Path, output: Path) -> Path | None:
    path = root / "aggregate_summary.json"
    if not path.exists():
        return None
    rows = json.loads(path.read_text(encoding="utf-8"))["aggregate"]
    environments = sorted({row["environment"] for row in rows})
    algorithms = ["pdqn", "mpdqn", "her_pdqn", "her_mpdqn"]
    x = np.arange(len(algorithms), dtype=float)
    width = 0.8 / max(1, len(environments))
    fig, ax = plt.subplots(figsize=(9, 5))
    for index, environment in enumerate(environments):
        mapping = {row["algorithm"]: row for row in rows if row["environment"] == environment}
        means = [mapping.get(a, {}).get("success_rate_mean", 0.0) for a in algorithms]
        stds = [mapping.get(a, {}).get("success_rate_std", 0.0) for a in algorithms]
        ax.bar(x + (index - (len(environments) - 1) / 2) * width, means, width, yerr=stds, capsize=3, label=environment.title())
    ax.set(xticks=x, xticklabels=[a.upper().replace("_", "-") for a in algorithms], ylabel="Evaluation success rate", ylim=(0, 1.05), title="Fixed-step baseline comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    target = output / "evaluation_success_rate.png"
    fig.savefig(target, dpi=180)
    plt.close(fig)
    return target


def plot_horizon(root: Path, output: Path) -> Path | None:
    path = root / "horizon_summary.json"
    if not path.exists():
        return None
    rows = json.loads(path.read_text(encoding="utf-8"))["aggregate"]
    relays = np.asarray([row["num_relays"] for row in rows])
    success = np.asarray([row["success_rate_mean"] for row in rows])
    success_std = np.asarray([row["success_rate_std"] for row in rows])
    reached = np.asarray([row["relay_reached_rate_mean"] for row in rows])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(relays, success, yerr=success_std, marker="o", capsize=4, label="Final success")
    ax.plot(relays, reached, marker="s", label="Reached at least one relay")
    ax.set(xticks=relays, xlabel="Number of relays", ylabel="Evaluation rate", ylim=(-0.02, 1.02), title="HER-MPDQN horizon scaling")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    target = output / "horizon_scaling.png"
    fig.savefig(target, dpi=180)
    plt.close(fig)
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    root = Path(args.input_root).resolve()
    output = Path(args.output_dir).resolve() if args.output_dir else root / "plots"
    output.mkdir(parents=True, exist_ok=True)
    created = plot_learning_curves(root, output)
    evaluation = plot_evaluation(root, output)
    if evaluation:
        created.append(evaluation)
    horizon = plot_horizon(root, output)
    if horizon:
        created.append(horizon)
    print(json.dumps({"plots": [str(path) for path in created]}, indent=2))


if __name__ == "__main__":
    main()
