"""Does the reward head get episode-ending steps right? Actual vs predicted rewards near termination.

On the run's own last 100 training episodes (read-only), compares the actual reward with the reward head's
prediction (mean(), original units) from the one-step PRIOR state (what imagination uses) at:
- the terminal step (discount 0: out of bounds, timeout or success);
- the 1-3 steps before it;
- all other valid, non-first steps.

Usage (with the TensorFlow venv):  python -B terminal_reward_check.py <run_dir> <checkpoint.pkl> out.json
"""
import json
import sys
from pathlib import Path

import ablation_patches  # noqa: E402  (puts the DV2 folder on sys.path)
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402


def main(run_dir, checkpoint, output):
    tf.keras.mixed_precision.set_global_policy('float32')
    tf.keras.utils.set_random_seed(17)
    agent = ablation_patches.agent_for_run(run_dir)
    agent.load(checkpoint)
    files = sorted((Path(run_dir) / 'episodes').glob('*.npz'), key=lambda p: int(p.stem))[-100:]
    episodes = [np.load(f) for f in files]
    data = {k: np.stack([e[k] for e in episodes]).astype(np.float32) for k in episodes[0].files}
    tensors = {k: tf.constant(v) for k, v in data.items()}
    _, prior, *_ = agent.world.forward(tensors, False)
    predicted = agent.world.reward(agent.world.core.get_feat(prior)).mean().numpy()
    valid = (data['valid'] > 0) & (data['is_first'] == 0)
    terminal = valid & (data['discount'] == 0)
    before = {k: np.zeros_like(terminal) for k in (1, 2, 3)}
    for b, t in zip(*np.nonzero(terminal)):
        for k in before:
            if t - k >= 1:
                before[k][b, t - k] = True
    near = terminal | before[1] | before[2] | before[3]

    def summary(mask):
        return dict(steps=int(mask.sum()), actual_mean=float(data['reward'][mask].mean()),
                    predicted_mean=float(predicted[mask].mean()))

    result = dict(checkpoint=Path(checkpoint).name, episodes=len(files),
                  terminal=summary(terminal), **{f'{k}_before_terminal': summary(before[k]) for k in before},
                  other=summary(valid & ~near))
    Path(output).write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main(*sys.argv[1:4])
