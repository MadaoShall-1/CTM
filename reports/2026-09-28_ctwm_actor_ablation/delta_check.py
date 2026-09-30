"""How well does a CT-WM world model predict one step of motion? Errors in metres (read-only).

For each row t >= 2 (so every method can be scored on the same steps), predict the position change
o_t - o_{t-1} from the one-step PRIOR state (built from the posterior at t-1 and action a_t; it has not
seen o_t):

- motion head (experiment-3 runs only): delta_head(prior features) * DELTA_SCALE;
- absolute decoder (every run): vector_head(prior features) - actual o_{t-1};
- constant velocity (no model): the previous actual change, o_{t-1} - o_{t-2}.

x/y are normalised as 2*pos/2000 - 1, so one normalised unit is 1000 m. Data: validate_nrsm.py's 8 held-out
episodes (seeds 40000-40007) and the run's own last 100 training episodes.

Usage (with the TensorFlow venv):  python -B delta_check.py <run_dir> <checkpoint.pkl> out.json
"""
import json
import sys
from pathlib import Path

import ablation_patches  # noqa: E402  (puts the DV2 folder on sys.path)
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

from validate_nrsm import collect  # noqa: E402


def score(agent, data):
    tensors = {k: tf.constant(v) for k, v in data.items()}
    _, prior, *_ = agent.world.forward(tensors, False)
    feat = agent.world.core.get_feat(prior)
    position = data['vector'][..., :2]
    actual = (position[:, 1:] - position[:, :-1]) * 1000.0                                 # rows 1..T
    mask = ablation_patches.delta_mask(tensors).numpy() > 0                                # rows 1..T
    common = mask[:, 1:] & mask[:, :-1]                                                    # rows 2..T
    predictions = {
        'absolute_decoder': (agent.world.vector(feat).mean().numpy()[:, 1:, :2] - position[:, :-1]) * 1000.0,
        'constant_velocity': np.concatenate([np.zeros_like(actual[:, :1]), actual[:, :-1]], 1),
    }
    if hasattr(agent.world, 'delta'):
        scale = np.asarray(ablation_patches.DELTA_SCALE[:2], np.float32)
        predictions['motion_head'] = agent.world.delta(feat[:, 1:]).mean().numpy()[..., :2] * scale * 1000.0
    target = actual[:, 1:][common]
    result = dict(steps=int(common.sum()), actual_step_median_m=float(np.median(np.hypot(*target.T))))
    for name, prediction in predictions.items():
        error = np.hypot(*(prediction[:, 1:][common] - target).T)
        result[name] = dict(median_m=float(np.median(error)), mean_m=float(error.mean()))
    return result


def main(run_dir, checkpoint, output):
    tf.keras.mixed_precision.set_global_policy('float32')
    tf.keras.utils.set_random_seed(17)
    agent = ablation_patches.agent_for_run(run_dir)
    agent.load(checkpoint)
    heldout, _ = collect(8, 40000)
    files = sorted((Path(run_dir) / 'episodes').glob('*.npz'), key=lambda p: int(p.stem))[-100:]
    episodes = [np.load(f) for f in files]
    recent = {k: np.stack([e[k] for e in episodes]).astype(np.float32) for k in episodes[0].files}
    result = dict(checkpoint=Path(checkpoint).name, heldout=score(agent, heldout),
                  in_distribution_last_100_training_episodes=score(agent, recent))
    Path(output).write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main(*sys.argv[1:4])
