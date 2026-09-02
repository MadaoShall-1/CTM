import json

from scripts.plot_results import plot_evaluation, plot_horizon, plot_learning_curves
from scripts.run_all_baselines import aggregate


def test_aggregate_reports_mean_std_and_required_metrics() -> None:
    records = []
    for seed, success in ((0, 0.25), (1, 0.75)):
        records.append({
            "environment": "direct", "algorithm": "pdqn", "seed": seed,
            "success_rate": success, "mean_return": -10.0, "mean_length": 10.0,
            "out_of_bounds_rate": 0.0, "relay_reached_rate": 0.0,
            "training_success_rate": success, "environment_steps": 100,
            "q_loss": 1.0, "parameter_actor_loss": -0.5,
            "her_relabel_count": 0,
        })
    row = aggregate(records)[0]
    assert row["success_rate_mean"] == 0.5
    assert row["success_rate_std"] == 0.25
    assert row["q_loss_mean"] == 1.0
    assert row["parameter_actor_loss_mean"] == -0.5
    assert row["her_relabel_count_mean"] == 0.0


def test_plotters_create_pngs_from_synthetic_run(tmp_path) -> None:
    run = tmp_path / "direct" / "pdqn" / "seed_0"
    run.mkdir(parents=True)
    (run / "config.yaml").write_text(
        "experiment:\n  environment: direct\n  algorithm: pdqn\n", encoding="utf-8")
    (run / "metrics.jsonl").write_text(
        json.dumps({"environment_steps": 1, "success_rate_100": 0.0}) + "\n" +
        json.dumps({"environment_steps": 2, "success_rate_100": 0.5}) + "\n",
        encoding="utf-8")
    aggregate_summary = {"aggregate": [{
        "environment": "direct", "algorithm": "pdqn",
        "success_rate_mean": 0.5, "success_rate_std": 0.1,
    }]}
    (tmp_path / "aggregate_summary.json").write_text(
        json.dumps(aggregate_summary), encoding="utf-8")
    output = tmp_path / "plots"
    output.mkdir()
    curves = plot_learning_curves(tmp_path, output)
    evaluation = plot_evaluation(tmp_path, output)
    assert curves and curves[0].stat().st_size > 0
    assert evaluation is not None and evaluation.stat().st_size > 0


def test_horizon_plotter(tmp_path) -> None:
    (tmp_path / "horizon_summary.json").write_text(json.dumps({"aggregate": [
        {"num_relays": 0, "success_rate_mean": 0.5, "success_rate_std": 0.1,
         "relay_reached_rate_mean": 0.0},
        {"num_relays": 1, "success_rate_mean": 0.1, "success_rate_std": 0.05,
         "relay_reached_rate_mean": 0.4},
    ]}), encoding="utf-8")
    output = tmp_path / "plots"
    output.mkdir()
    path = plot_horizon(tmp_path, output)
    assert path is not None and path.stat().st_size > 0
