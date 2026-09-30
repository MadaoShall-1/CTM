"""Can a CT-WM world model tell better steps from worse ones? Reward prediction on ordinary steps.

The actor learns from imagined rewards. On non-terminal steps the goal_safe_v1 reward is
-(1-gamma) plus potential shaping, so the informative variation is tiny (~0.001-0.004).
This script compares the reward head's predictions with actual rewards on non-terminal steps:

- in distribution: the run's own last 100 training episodes;
- held out: validate_nrsm.py's 8 episodes (seeds 40000-40007), split into the controller and random halves.

Read-only. Usage (with the TensorFlow venv):
    python -B reward_signal_check.py <run_dir> <checkpoint.pkl> out.json
"""
import json
import sys
from pathlib import Path

import ablation_patches  # noqa: E402  (puts the DV2 folder on sys.path)
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

from validate_nrsm import collect  # noqa: E402


def check(agent, data):
    tensors = {k: tf.constant(v) for k, v in data.items()}
    _, prior, _, reward, _ = agent.world.forward(tensors, False)
    predictions = dict(posterior=reward.mean().numpy(),
                       prior=agent.world.reward(agent.world.core.get_feat(prior)).mean().numpy())
    mask = (data['valid'] > 0) & (data['is_first'] == 0) & (data['discount'] > 0)
    actual = data['reward'][mask]
    result = dict(steps=int(mask.sum()), actual_mean=float(actual.mean()), actual_std=float(actual.std()))
    for name, prediction in predictions.items():
        prediction = prediction[mask]
        result[f'{name}_rmse'] = float(np.sqrt(np.mean((prediction - actual) ** 2)))
        result[f'{name}_corr'] = float(np.corrcoef(prediction, actual)[0, 1])
    return result


def main(run_dir, checkpoint, output):
    tf.keras.mixed_precision.set_global_policy('float32')
    tf.keras.utils.set_random_seed(17)
    agent = ablation_patches.agent_for_run(run_dir)   # same architecture and ac_config as the checkpoint
    agent.load(checkpoint)
    files = sorted((Path(run_dir) / 'episodes').glob('*.npz'), key=lambda p: int(p.stem))[-100:]
    episodes = [np.load(f) for f in files]
    recent = {k: np.stack([e[k] for e in episodes]).astype(np.float32) for k in episodes[0].files}
    heldout, _ = collect(8, 40000)
    result = dict(checkpoint=Path(checkpoint).name,
                  in_distribution_last_100_training_episodes=check(agent, recent),
                  heldout_controller=check(agent, {k: v[[0, 2, 4, 6]] for k, v in heldout.items()}),
                  heldout_random=check(agent, {k: v[[1, 3, 5, 7]] for k, v in heldout.items()}))
    Path(output).write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main(*sys.argv[1:4])
