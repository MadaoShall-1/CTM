"""Can the world model see the map edge coming? (Experiment 6.)

Two datasets (read-only):
- held out: the random-policy half of validate_nrsm.collect(400, 60000), 200 episodes, identical for every run;
- own: the 100 training episodes that ended last before the checkpoint was written.

From the one-step PRIOR state (what imagination uses), it reports:
- at out-of-bounds endings: the discount head's predicted probability that the episode ends (1 - mean / 0.99;
  stored discounts are 0.99 or 0), and the reward head's predicted reward as a share of the actual one;
- at steps within 20 m of an edge that do not end: the same ending probability (false alarms);
- on rows whose nearest edge is within 50 m: the error of the predicted distance to the nearest edge, in metres,
  from the vector head's x/y (every run) and from the edge head (experiment-6 runs). The same errors from the
  POSTERIOR state, which has already seen that row's observation, show how much is lost in the one-step prediction.

Usage (with the TensorFlow venv):  python -B edge_check.py <run_dir> <checkpoint.pkl> out.json
"""
import json
import sys
from pathlib import Path

import ablation_patches  # noqa: E402  (puts the DV2 folder on sys.path)
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

from validate_nrsm import collect  # noqa: E402

GAMMA = 0.99
HELDOUT = (400, 60000)
NEAR_M, ERROR_BAND_M = 20.0, 50.0


def nearest_edge_m(xy):
    """Distance from normalised x/y (clipped to [-1, 1]) to the nearest map edge, in metres."""
    return 1000 * (1 - np.max(np.abs(np.clip(xy, -1, 1)), axis=-1))


def own_episodes(run_dir, checkpoint):
    written_at = int(Path(checkpoint).stem.split('_')[1][3:])
    ended = {}
    with (Path(run_dir) / 'metrics.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            record = json.loads(line)
            if record['kind'] == 'episode' and record['env_steps'] <= written_at:
                ended[int(record['metrics']['scene'])] = record['env_steps']
    files = sorted((p for p in (Path(run_dir) / 'episodes').glob('*.npz') if int(p.stem) in ended),
                   key=lambda p: (ended[int(p.stem)], int(p.stem)))[-100:]
    episodes = [np.load(f) for f in files]
    return {k: np.stack([e[k] for e in episodes]).astype(np.float32) for k in episodes[0].files}


def check(agent, data):
    post, prior, *_ = agent.world.forward({k: tf.constant(v) for k, v in data.items()}, False)
    feat, post_feat = agent.world.core.get_feat(prior), agent.world.core.get_feat(post)
    p_end = 1 - agent.world.discount(feat).mean().numpy() / GAMMA
    predicted = agent.world.reward(feat).mean().numpy()
    actual_edge = nearest_edge_m(data['vector'][..., :2])
    vector_edge = nearest_edge_m(agent.world.vector(feat).mean().numpy()[..., :2])

    valid = (data['valid'] > 0) & (data['is_first'] == 0)
    step = np.broadcast_to(np.arange(data['valid'].shape[1]), valid.shape)
    # Endings before the 100-step horizon with a negative reward: out of bounds (a delivery ends with +1).
    oob = valid & (data['discount'] == 0) & (step < valid.shape[1] - 1) & (data['reward'] < 0)
    near = valid & (data['discount'] > 0) & (actual_edge < NEAR_M)
    band = valid & (actual_edge < ERROR_BAND_M)

    def error(prediction):
        e = np.abs(prediction[band] - actual_edge[band])
        return dict(median=float(np.median(e)), mean=float(e.mean()))

    result = dict(
        oob_endings=dict(steps=int(oob.sum()), end_prob_mean=float(p_end[oob].mean()),
                         end_prob_median=float(np.median(p_end[oob])),
                         reward_share=float(predicted[oob].mean() / data['reward'][oob].mean())) if oob.any() else None,
        near_edge_no_ending=dict(steps=int(near.sum()), end_prob_mean=float(p_end[near].mean())) if near.any() else None,
        nearest_edge_error_m=dict(rows=int(band.sum()), vector_head=error(vector_edge)),
        nearest_edge_error_posterior_m=dict(
            rows=int(band.sum()), vector_head=error(nearest_edge_m(agent.world.vector(post_feat).mean().numpy()[..., :2]))))
    if hasattr(agent.world, 'edge'):
        for key, features in (('nearest_edge_error_m', feat), ('nearest_edge_error_posterior_m', post_feat)):
            distances = ablation_patches.edge_prediction(agent.world.edge(features).mean()).numpy()
            result[key]['edge_head'] = error(1000 * distances.min(-1))
    return result


def main(run_dir, checkpoint, output):
    tf.keras.mixed_precision.set_global_policy('float32')
    tf.keras.utils.set_random_seed(17)
    agent = ablation_patches.agent_for_run(run_dir)   # same architecture and ac_config as the checkpoint
    agent.load(checkpoint)
    data, records = collect(*HELDOUT)
    random = [i for i, r in enumerate(records) if r['policy'] == 'random']
    heldout = {k: v[random] for k, v in data.items()}
    result = dict(checkpoint=Path(checkpoint).name, heldout_episodes=len(random),
                  heldout=check(agent, heldout), own=check(agent, own_episodes(run_dir, checkpoint)))
    Path(output).write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main(*sys.argv[1:4])
