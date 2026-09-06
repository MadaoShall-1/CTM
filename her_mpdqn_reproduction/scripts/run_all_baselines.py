"""Run comparable fixed-step, multi-seed baseline experiments and aggregate results."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate import evaluate_checkpoint  # noqa: E402
from scripts.train import ALGORITHMS, load_config, train  # noqa: E402


def parse_csv(value: str, cast=str) -> list:
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def aggregate(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for environment in sorted({r["environment"] for r in records}):
        for algorithm in ALGORITHMS:
            group = [r for r in records if r["environment"] == environment and r["algorithm"] == algorithm]
            if not group:
                continue
            row: dict[str, Any] = {
                "environment": environment,
                "algorithm": algorithm,
                "num_seeds": len(group),
                "seeds": [r["seed"] for r in group],
            }
            for key in (
                "success_rate", "mean_return", "mean_length", "out_of_bounds_rate", "boundary_hit_rate",
                "relay_reached_rate", "training_success_rate", "environment_steps",
                "q_loss", "parameter_actor_loss", "her_relabel_count",
                "best_eval_success_rate", "best_eval_mean_return",
            ):
                values = [float(r[key]) for r in group if r.get(key) is not None]
                row[f"{key}_mean"] = float(np.mean(values)) if values else None
                row[f"{key}_std"] = float(np.std(values)) if values else None
            rows.append(row)
    return rows


def metric_totals(metrics_path: Path) -> dict[str, float]:
    records = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line]
    if not records:
        return {"her_relabel_count": 0.0}
    return {
        "her_relabel_count": float(sum(r.get("her_relabel_count", 0) for r in records)),
        "q_loss": records[-1].get("q_loss"),
        "parameter_actor_loss": records[-1].get("parameter_actor_loss"),
        "training_success_rate": records[-1].get("success_rate_run"),
        "environment_steps": records[-1].get("environment_steps"),
    }


def latest_training_checkpoint(run_dir: Path) -> Path | None:
    candidates: list[Path] = []
    for path in run_dir.glob("checkpoint_*.pt"):
        try:
            int(path.stem.split("_")[-1])
            candidates.append(path)
        except ValueError:
            continue
    last = run_dir / "last.pt"
    if last.exists():
        candidates.append(last)
    return max(candidates, key=lambda path: path.stat().st_mtime_ns) if candidates else None


def run_one(job: dict[str, Any]) -> dict[str, Any]:
    """Execute one independent seed/run inside its own process."""
    import torch

    torch.set_num_threads(int(job.get("worker_threads", 1)))
    environment = job["environment"]
    algorithm = job["algorithm"]
    seed = int(job["seed"])
    steps = job.get("steps")
    steps = None if steps is None else int(steps)
    eval_episodes = int(job["eval_episodes"])
    reuse_existing = bool(job["reuse_existing"])
    run_dir = Path(job["run_dir"])
    config_path = Path(job["config_path"])
    checkpoint = run_dir / "last.pt"
    evaluation_path = run_dir / "evaluation" / "summary.json"
    if reuse_existing and checkpoint.exists() and evaluation_path.exists():
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    else:
        config = load_config(config_path)
        resume_from = latest_training_checkpoint(run_dir) if reuse_existing else None
        training_override = {} if steps is None else {"max_environment_steps": steps}
        train(config, overrides={
            "experiment": {"seed": seed, "output_dir": str(run_dir)},
            "training": training_override,
        }, resume_from=resume_from)
        evaluation = evaluate_checkpoint(
            load_config(run_dir / "config.yaml"), checkpoint,
            episodes=eval_episodes, seed=100_000 + seed,
            output_dir=run_dir / "evaluation",
        )
    best_evaluation = None
    best_checkpoint = run_dir / "best_eval.pt"
    best_evaluation_path = run_dir / "evaluation_best" / "summary.json"
    if best_checkpoint.exists():
        if reuse_existing and best_evaluation_path.exists():
            best_evaluation = json.loads(best_evaluation_path.read_text(encoding="utf-8"))
        else:
            best_evaluation = evaluate_checkpoint(
                load_config(run_dir / "config.yaml"), best_checkpoint,
                episodes=eval_episodes, seed=100_000 + seed,
                output_dir=run_dir / "evaluation_best",
            )
    totals = metric_totals(run_dir / "metrics.jsonl")
    return {
        "environment": environment,
        "algorithm": algorithm,
        "seed": seed,
        "success_rate": evaluation["success_rate"],
        "mean_return": evaluation["mean_return"],
        "mean_length": evaluation["mean_length"],
        "out_of_bounds_rate": evaluation["out_of_bounds_rate"],
        "boundary_hit_rate": evaluation.get("boundary_hit_rate", 0.0),
        "relay_reached_rate": evaluation["relay_reached_rate"],
        "training_success_rate": totals.get("training_success_rate"),
        "environment_steps": totals.get("environment_steps", steps),
        "q_loss": totals.get("q_loss"),
        "parameter_actor_loss": totals.get("parameter_actor_loss"),
        "her_relabel_count": totals["her_relabel_count"],
        "best_eval_success_rate": (
            best_evaluation["success_rate"] if best_evaluation is not None else None),
        "best_eval_mean_return": (
            best_evaluation["mean_return"] if best_evaluation is not None else None),
        "run_dir": str(run_dir),
    }


def run_suite(
    *, environments: list[str], algorithms: list[str], seeds: list[int], steps: int | None,
    eval_episodes: int, output_root: Path, reuse_existing: bool = False,
    workers: int = 1, worker_threads: int = 1,
) -> dict[str, Any]:
    if workers <= 0:
        raise ValueError("workers must be positive")
    if worker_threads <= 0:
        raise ValueError("worker_threads must be positive")
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    jobs: list[dict[str, Any]] = []
    for environment in environments:
        for algorithm in algorithms:
            config_path = ROOT / "configs" / f"{environment}_{algorithm}.yaml"
            if not config_path.exists():
                raise FileNotFoundError(config_path)
            for seed in seeds:
                run_dir = output_root / environment / algorithm / f"seed_{seed}"
                jobs.append({
                    "environment": environment, "algorithm": algorithm,
                    "seed": seed, "steps": steps, "eval_episodes": eval_episodes,
                    "reuse_existing": reuse_existing, "run_dir": str(run_dir),
                    "config_path": str(config_path), "worker_threads": worker_threads,
                })

    records: list[dict[str, Any]] = []

    def persist() -> None:
        ordered = sorted(records, key=lambda r: (
            environments.index(r["environment"]), algorithms.index(r["algorithm"]), r["seed"]))
        (output_root / "per_run.jsonl").write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in ordered),
            encoding="utf-8")

    if workers == 1:
        for job in jobs:
            records.append(run_one(job))
            persist()
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(run_one, job): job for job in jobs}
            for future in as_completed(futures):
                records.append(future.result())
                persist()

    aggregated = aggregate(records)
    result = {
        "fixed_environment_steps": steps,
        "evaluation_episodes": eval_episodes,
        "num_runs": len(records),
        "runs": records,
        "aggregate": aggregated,
    }
    (output_root / "aggregate_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    if aggregated:
        with (output_root / "aggregate_summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(aggregated[0]))
            writer.writeheader()
            writer.writerows(aggregated)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--environments", default="direct,relay")
    parser.add_argument("--algorithms", default=",".join(ALGORITHMS))
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument(
        "--steps", type=int,
        help="Optional environment-step cap; omit for paper episode counts")
    parser.add_argument("--eval-episodes", type=int, default=1000)
    parser.add_argument("--output-root", default="outputs/paper_reproduction_3seed")
    default_workers = min(8, max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--workers", type=int, default=default_workers,
                        help="Independent training processes to run concurrently")
    parser.add_argument("--worker-threads", type=int, default=1,
                        help="PyTorch CPU threads used by each worker")
    parser.add_argument(
        "--reuse-existing", action="store_true",
        help="Skip completed runs and resume incomplete runs from their latest checkpoint")
    args = parser.parse_args()
    result = run_suite(
        environments=parse_csv(args.environments),
        algorithms=parse_csv(args.algorithms),
        seeds=parse_csv(args.seeds, int),
        steps=args.steps,
        eval_episodes=args.eval_episodes,
        output_root=Path(args.output_root) if Path(args.output_root).is_absolute() else ROOT / args.output_root,
        reuse_existing=args.reuse_existing,
        workers=args.workers,
        worker_threads=args.worker_threads,
    )
    print(json.dumps({"num_runs": result["num_runs"], "aggregate": result["aggregate"]}, indent=2))


if __name__ == "__main__":
    main()
