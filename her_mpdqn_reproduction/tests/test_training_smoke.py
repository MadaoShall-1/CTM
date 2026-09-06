import json
from pathlib import Path

import pytest
import torch
import numpy as np

from scripts.train import (
    load_config,
    make_environment,
    make_observation_encoder,
    paper_update_count,
    train,
    validate_config,
)


ROOT = Path(__file__).resolve().parents[1]


def test_paper_observation_interfaces_do_not_leak_goal_or_phase() -> None:
    plain = load_config(ROOT / "configs" / "relay_mpdqn.yaml")
    her = load_config(ROOT / "configs" / "relay_her_mpdqn.yaml")
    env = make_environment(plain)
    obs, _ = env.reset(seed=3)
    plain_encoder = make_observation_encoder(plain, env)
    her_encoder = make_observation_encoder(her, env)
    assert plain_encoder.output_dim == 7
    assert her_encoder.output_dim == 9
    changed = {key: value.copy() for key, value in obs.items()}
    changed["desired_goal"] = np.asarray([0.0, 0.0], dtype=np.float32)
    np.testing.assert_allclose(plain_encoder(obs), plain_encoder(changed))
    assert not np.allclose(her_encoder(obs), her_encoder(changed))


def test_paper_update_schedules_match_published_multiplier_semantics() -> None:
    fixed = {"update_schedule": {"mode": "paper_fixed", "multiplier": 40}}
    assert paper_update_count(fixed, episode_length=80, eligible_steps=80) == (3200, 40)
    dynamic = {"update_schedule": {
        "mode": "paper_dynamic", "reference_episode_length": 100,
        "min_multiplier": 1, "max_multiplier": 10,
    }}
    assert paper_update_count(dynamic, episode_length=100, eligible_steps=100) == (100, 1)
    assert paper_update_count(dynamic, episode_length=20, eligible_steps=20) == (100, 5)


@pytest.mark.parametrize("algorithm", ["pdqn", "mpdqn", "her_pdqn", "her_mpdqn"])
def test_direct_baseline_short_training_writes_losses_and_checkpoints(tmp_path, algorithm) -> None:
    config = load_config(ROOT / "configs" / f"direct_{algorithm}.yaml")
    output = tmp_path / algorithm
    summary = train(config, overrides={
        "experiment": {"output_dir": str(output)},
        "env": {"max_episode_steps": 4},
        "agent": {"hidden_sizes": [8], "device": "cpu"},
        "replay": {"capacity": 128},
        "training": {
            "episodes": 2,
            "learning_starts": 0,
            "batch_size": 4,
            "update_every": 2,
            "gradient_steps": 1,
            "checkpoint_every": 0,
        },
        "exploration": {"decay_steps": 8},
    })
    assert summary["updates"] > 0
    assert summary["q_loss"] is not None
    assert summary["parameter_actor_loss"] is not None
    assert (output / "last.pt").exists()
    assert (output / "best.pt").exists()
    records = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    assert len(records) == 2
    assert "q_loss" in records[-1]
    assert "parameter_actor_loss" in records[-1]
    if algorithm.startswith("her_"):
        assert records[-1]["her_relabel_count"] > 0
    else:
        assert records[-1]["her_relabel_count"] == 0


def test_relay_her_mpdqn_requires_phase_aware_replay() -> None:
    config = load_config(ROOT / "configs" / "relay_her_mpdqn.yaml")
    config["replay"]["phase_aware"] = False
    with pytest.raises(ValueError, match="must enable phase_aware"):
        validate_config(config)


def test_training_stops_at_exact_environment_step_budget(tmp_path) -> None:
    config = load_config(ROOT / "configs" / "direct_mpdqn.yaml")
    summary = train(config, overrides={
        "experiment": {"output_dir": str(tmp_path / "fixed_steps")},
        "env": {"max_episode_steps": 4},
        "agent": {"hidden_sizes": [8], "device": "cpu"},
        "replay": {"capacity": 64},
        "training": {
            "episodes": 10,
            "max_environment_steps": 7,
            "learning_starts": 0,
            "batch_size": 4,
            "update_every": 1,
            "checkpoint_every": 0,
        },
    })
    assert summary["environment_steps"] == 7
    assert summary["episodes"] == 2
    records = [
        json.loads(line)
        for line in (tmp_path / "fixed_steps" / "metrics.jsonl").read_text().splitlines()
    ]
    assert records[-1]["environment_steps"] == 7
    assert records[-1]["length"] == 3
    assert records[-1]["truncated"] == 1.0


def test_periodic_deterministic_validation_writes_best_eval_checkpoint(tmp_path) -> None:
    config = load_config(ROOT / "configs" / "direct_mpdqn.yaml")
    output = tmp_path / "validated"
    train(config, overrides={
        "experiment": {"output_dir": str(output)},
        "env": {"max_episode_steps": 2},
        "agent": {"hidden_sizes": [8], "device": "cpu"},
        "replay": {"capacity": 32},
        "training": {
            "episodes": 2,
            "learning_starts": 10,
            "batch_size": 4,
            "update_schedule": None,
            "checkpoint_every": 0,
            "validation_every": 1,
            "validation_episodes": 2,
            "validation_seed": 91,
        },
    })
    assert (output / "best_eval.pt").exists()
    records = [
        json.loads(line)
        for line in (output / "metrics.jsonl").read_text().splitlines()
    ]
    assert all("validation_success_rate" in record for record in records)


@pytest.mark.parametrize("num_relays", [0, 1, 2, 4, 8])
def test_multi_relay_training_entrypoint_smoke(tmp_path, num_relays) -> None:
    config = load_config(ROOT / "configs" / "multi_relay_her_mpdqn.yaml")
    summary = train(config, overrides={
        "experiment": {"output_dir": str(tmp_path / f"relay_{num_relays}")},
        "env": {"num_relays": num_relays, "max_episode_steps": 2},
        "agent": {"hidden_sizes": [8], "device": "cpu"},
        "replay": {"capacity": 64, "her_k": 1},
        "training": {
            "episodes": 1,
            "max_environment_steps": 2,
            "learning_starts": 10,
            "batch_size": 4,
            "checkpoint_every": 0,
        },
    })
    assert summary["environment_steps"] == 2
    assert (tmp_path / f"relay_{num_relays}" / "last.pt").exists()


def _resume_test_overrides(output: Path, *, episodes: int, algorithm: str = "mpdqn") -> dict:
    return {
        "experiment": {"algorithm": algorithm, "output_dir": str(output)},
        "env": {"max_episode_steps": 4},
        "agent": {"hidden_sizes": [8], "device": "cpu"},
        "replay": {"capacity": 128, "her_k": 1},
        "training": {
            "episodes": episodes,
            "learning_starts": 0,
            "batch_size": 4,
            "update_every": 1,
            "gradient_steps": 1,
            "checkpoint_every": 2,
        },
        "exploration": {"decay_steps": 20},
    }


def test_resume_matches_uninterrupted_training_exactly(tmp_path) -> None:
    config = load_config(ROOT / "configs" / "direct_mpdqn.yaml")
    output = tmp_path / "exact_resume"
    train(config, overrides=_resume_test_overrides(output, episodes=4))
    uninterrupted = torch.load(output / "last.pt", map_location="cpu", weights_only=False)

    train(
        config,
        overrides=_resume_test_overrides(output, episodes=4),
        resume_from=output / "checkpoint_2.pt",
    )
    resumed = torch.load(output / "last.pt", map_location="cpu", weights_only=False)

    assert resumed["trainer_state"] == uninterrupted["trainer_state"]
    for network in ("parameter_actor", "q_network", "target_parameter_actor", "target_q_network"):
        for name, value in uninterrupted[network].items():
            torch.testing.assert_close(resumed[network][name], value, rtol=0, atol=0)
    for name in ("states", "actions", "action_parameters", "rewards", "next_states", "terminated", "truncated"):
        size = resumed["replay_state"]["size"]
        left = resumed["replay_state"][name][:size]
        right = uninterrupted["replay_state"][name][:size]
        assert (left == right).all()
    metrics = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    assert [record["episode"] for record in metrics] == [1, 2, 3, 4]


def test_her_resume_restores_replay_counters_and_rng(tmp_path) -> None:
    config = load_config(ROOT / "configs" / "direct_her_mpdqn.yaml")
    output = tmp_path / "her_resume"
    overrides = _resume_test_overrides(output, episodes=2, algorithm="her_mpdqn")
    train(config, overrides=overrides)
    checkpoint = torch.load(output / "checkpoint_2.pt", map_location="cpu", weights_only=False)
    replay = checkpoint["replay_state"]
    assert replay["buffer_type"] == "HERReplayBuffer"
    assert replay["episodes_added"] == 2
    assert replay["total_relabelled_added"] > 0
    assert "relabeler_rng_state" in replay
    assert "exploration_rng_state" in checkpoint
    assert "environment" in checkpoint["rng_state"]
