"""What do the 64 DreamerV2 controller demonstrations teach? (Experiment 8.)

On the demonstrations of ablation_patches.controller_demonstrations() (seeds 30000-30063, each ending in a
delivery), for any run's checkpoint (read-only):
- delivery: at each demonstration's final row (the +1 delivery), the reward head's prediction from the one-step
  PRIOR state, as a share of the actual reward, and the discount head's predicted ending probability
  (1 - mean / 0.99);
- imitation: on every demonstrated step, whether the actor's deterministic choice (MOVE or TURN) matches the
  controller's, and the mean absolute error of its parameter for the chosen branch. The actor reads the posterior
  features of the state before the action, as in dreamer.align_behavior_supervision. The controller chooses MOVE
  on ~93% of steps, so an always-MOVE actor would agree 93% of the time; balanced agreement (the mean of the
  per-choice agreement on MOVE steps and on TURN steps) is 0.5 for any constant choice.

Usage (with the TensorFlow venv):  python -B demo_check.py <run_dir> <checkpoint.pkl> out.json
"""
import json
import sys
from pathlib import Path

import ablation_patches  # noqa: E402  (puts the DV2 folder on sys.path)
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

import models  # noqa: E402

GAMMA = 0.99


def main(run_dir, checkpoint, output):
    tf.keras.mixed_precision.set_global_policy('float32')
    tf.keras.utils.set_random_seed(17)
    agent = ablation_patches.agent_for_run(run_dir)   # same architecture and ac_config as the checkpoint
    agent.load(checkpoint)
    episodes = ablation_patches.controller_demonstrations()
    data = {k: np.stack([e[k] for e in episodes]) for k in episodes[0]}
    post, prior, *_ = agent.world.forward({k: tf.constant(v) for k, v in data.items()}, False)
    prior_feat = agent.world.core.get_feat(prior)
    predicted = agent.world.reward(prior_feat).mean().numpy()
    p_end = 1 - agent.world.discount(prior_feat).mean().numpy() / GAMMA
    last = data['valid'].sum(1).astype(int) - 1
    rows = np.arange(len(episodes))
    delivered = data['reward'][rows, last]

    # Imitation: state t (posterior) -> the action taken after it, action[t + 1].
    dist = agent.actor(agent.world.core.get_feat(post)[:, :-1])
    target = models.canonical_action(tf.constant(data['action'][:, 1:])).numpy()
    mask = (data['valid'][:, 1:] * (1 - data['is_first'][:, 1:])) > 0
    k = target.shape[-1] // 2
    chosen = np.argmax(dist.logits.numpy(), -1)
    demonstrated = np.argmax(target[..., :k], -1)
    parameter = np.tanh(dist.mean_tensor.numpy())
    selected = np.take_along_axis(parameter, demonstrated[..., None], -1)[..., 0]
    wanted = np.take_along_axis(target[..., k:], demonstrated[..., None], -1)[..., 0]
    result = dict(
        checkpoint=Path(checkpoint).name, demonstrations=len(episodes), steps=int(mask.sum()),
        delivery=dict(actual_mean=float(delivered.mean()),
                      predicted_mean=float(predicted[rows, last].mean()),
                      reward_share=float(predicted[rows, last].mean() / delivered.mean()),
                      end_prob_mean=float(p_end[rows, last].mean())),
        imitation=dict(discrete_agreement=float((chosen == demonstrated)[mask].mean()),
                       balanced_agreement=float(np.mean([(chosen == demonstrated)[mask & (demonstrated == c)].mean()
                                                         for c in (0, 1)])),
                       move_step_agreement=float((chosen == 0)[mask & (demonstrated == 0)].mean()),
                       turn_step_agreement=float((chosen == 1)[mask & (demonstrated == 1)].mean()),
                       controller_move_share=float((demonstrated == 0)[mask].mean()),
                       actor_move_share=float((chosen == 0)[mask].mean()),
                       parameter_mae=float(np.abs(selected - wanted)[mask].mean())))
    Path(output).write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main(*sys.argv[1:4])
