"""Does the world model see episode endings coming? (Exploratory, after experiment 5.)

On the 100 training episodes that ended last before the checkpoint was written (read-only; for the final
checkpoint, the run's last 100), from the one-step PRIOR state (what imagination uses):
- the reward head's predicted reward (mean(), original units);
- the discount head's predicted probability that the episode ends at that step (1 - mean / 0.99; stored
  discounts are 0.99 or 0);
at ordinary steps, the step before an ending, timeout endings (row 100, the 100-step horizon) and earlier
endings (out of bounds, or a delivery). Also the least-squares slope of the predicted reward against the predicted
ending probability over all steps, which is about -0.69 when the two heads agree on what an ending costs.

For two-hot reward heads it also splits the predicted distribution into a tail (bins with |bin| > TAIL, i.e.
rewards more than ~0.074 from REWARD_MU: terminal-sized outcomes) and a body (the other bins, with their
conditional mean in original units). Ordinary rewards lie within |bin| <= ~1.6 and terminal rewards near
+/-6.6, so TAIL = 4 separates them cleanly.

Usage (with the TensorFlow venv):
    python -B ending_check.py <run_dir> <checkpoint.pkl> out.json   # one checkpoint
    python -B ending_check.py <run_dir> --all out.json              # every checkpoint after step 0, in order
"""
import json
import sys
from pathlib import Path

import ablation_patches  # noqa: E402  (puts the DV2 folder on sys.path)
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

TAIL = 4.0
GAMMA = 0.99


def check(agent, run_dir, checkpoint):
    agent.load(checkpoint)
    # Checkpoint names carry the environment step count (checkpoint_env<steps>_ac<updates>.pkl).
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
    data = {k: np.stack([e[k] for e in episodes]).astype(np.float32) for k in episodes[0].files}
    _, prior, *_ = agent.world.forward({k: tf.constant(v) for k, v in data.items()}, False)
    feat = agent.world.core.get_feat(prior)
    reward = agent.world.reward(feat)
    predicted = reward.mean().numpy()
    p_end = 1 - agent.world.discount(feat).mean().numpy() / GAMMA
    twohot = isinstance(agent.world.reward, ablation_patches.TwoHotRewardHead)
    if twohot:
        probs = tf.nn.softmax(reward.logits, -1).numpy().astype(np.float64)
        bins = np.array(ablation_patches.TWOHOT_BINS)
        values = ablation_patches.REWARD_MU + ablation_patches.REWARD_SIGMA * np.sign(bins) * np.expm1(np.abs(bins))
        tail = np.abs(bins) > TAIL
        p_tail = probs[..., tail].sum(-1)
        body = (probs[..., ~tail] * values[~tail]).sum(-1) / probs[..., ~tail].sum(-1)

    valid = (data['valid'] > 0) & (data['is_first'] == 0)
    ending = valid & (data['discount'] == 0)
    ordinary = valid & (data['discount'] > 0)
    # Episodes are padded to 101 rows; row t is step t, and a timeout ends at the last row.
    step = np.broadcast_to(np.arange(ending.shape[1]), ending.shape)
    timeout, early = ending & (step == ending.shape[1] - 1), ending & (step < ending.shape[1] - 1)
    before = np.zeros_like(ending)
    for b, t in zip(*np.nonzero(ending)):
        if t - 1 >= 1:
            before[b, t - 1] = True
    actual = data['reward']

    def summary(mask):
        if not mask.any():
            return None
        out = dict(steps=int(mask.sum()), actual_mean=float(actual[mask].mean()),
                   predicted_mean=float(predicted[mask].mean()), end_prob_mean=float(p_end[mask].mean()),
                   end_prob_median=float(np.median(p_end[mask])))
        if twohot:
            out.update(tail_mass_mean=float(p_tail[mask].mean()), tail_mass_median=float(np.median(p_tail[mask])))
        return out

    def fit(prediction, mask):
        error = prediction[mask] - actual[mask]
        return dict(rmse=float(np.sqrt(np.mean(error ** 2))),
                    corr=float(np.corrcoef(prediction[mask], actual[mask])[0, 1]))

    # Do the two heads agree on what an ending costs? Least-squares slope of the predicted reward against the
    # predicted ending probability over all steps: about -0.69 (ending ~-0.70 vs ordinary ~-0.007) if they do.
    slope, intercept = np.polyfit(p_end[valid], predicted[valid], 1)
    result = dict(checkpoint=Path(checkpoint).name, env_steps=written_at, episodes=len(files), twohot=twohot,
                  reward_vs_end_prob=dict(slope=float(slope), intercept=float(intercept), steps=int(valid.sum())),
                  ordinary=summary(ordinary), one_before_ending=summary(before),
                  timeout_ending=summary(timeout), early_ending=summary(early))
    result['ordinary'].update(actual_std=float(actual[ordinary].std()), full_prediction=fit(predicted, ordinary))
    if twohot:
        result.update(tail_threshold=TAIL)
        result['ordinary'].update(body_prediction=fit(body, ordinary),
                                  tail_vs_end_prob_corr=float(np.corrcoef(p_tail[ordinary], p_end[ordinary])[0, 1]))
    return result


def main(run_dir, checkpoint, output):
    tf.keras.mixed_precision.set_global_policy('float32')
    tf.keras.utils.set_random_seed(17)
    agent = ablation_patches.agent_for_run(run_dir)   # same architecture and ac_config as the checkpoints
    if checkpoint == '--all':
        paths = sorted(Path(run_dir).glob('checkpoint_env*.pkl'))
        result = [check(agent, run_dir, path) for path in paths if int(path.stem.split('_')[1][3:]) > 0]
    else:
        result = check(agent, run_dir, checkpoint)
    Path(output).write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main(*sys.argv[1:4])
