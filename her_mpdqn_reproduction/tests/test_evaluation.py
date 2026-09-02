import json
from pathlib import Path

import numpy as np

from scripts.evaluate import evaluate_checkpoint, load_evaluation_config
from scripts.train import load_config, train


ROOT = Path(__file__).resolve().parents[1]


def train_tiny_checkpoint(output: Path) -> Path:
    config = load_config(ROOT / "configs" / "direct_mpdqn.yaml")
    train(config, overrides={
        "experiment": {"output_dir": str(output)},
        "env": {"max_episode_steps": 4},
        "agent": {"hidden_sizes": [8], "device": "cpu"},
        "replay": {"capacity": 128},
        "training": {
            "episodes": 2,
            "learning_starts": 0,
            "batch_size": 4,
            "update_every": 2,
            "checkpoint_every": 0,
        },
    })
    return output / "last.pt"


def test_checkpoint_evaluation_is_reproducible_and_saves_trajectories(tmp_path) -> None:
    checkpoint = train_tiny_checkpoint(tmp_path / "train")
    config = load_evaluation_config(checkpoint)
    first = evaluate_checkpoint(
        config, checkpoint, episodes=3, seed=77,
        output_dir=tmp_path / "eval_first", save_trajectories=True)
    second = evaluate_checkpoint(
        config, checkpoint, episodes=3, seed=77,
        output_dir=tmp_path / "eval_second", save_trajectories=False)
    for name in (
        "mean_return", "std_return", "mean_length", "success_rate",
        "out_of_bounds_rate", "truncation_rate"):
        assert first[name] == second[name]
    assert first["trajectories_saved"] == 3
    assert (tmp_path / "eval_first" / "summary.json").exists()
    records = [
        json.loads(line)
        for line in (tmp_path / "eval_first" / "episodes.jsonl").read_text().splitlines()
    ]
    assert len(records) == 3
    trajectory_path = tmp_path / "eval_first" / "trajectories" / "episode_00001.npz"
    with np.load(trajectory_path, allow_pickle=False) as trajectory:
        required = {
            "obs", "action_discrete", "action_parameter", "reward", "next_obs",
            "goal", "achieved_goal", "phase", "terminated", "truncated"}
        assert required.issubset(trajectory.files)
        assert trajectory["obs"].shape[0] == records[0]["length"]
        assert trajectory["action_discrete"].dtype == np.int32
        assert trajectory["terminated"].dtype == np.bool_
