# DreamerV2 UAV migration

This build replaces the previous DreamerV1 Gaussian RSSM with the official
DreamerV2 algorithmic structure: categorical RSSM, straight-through discrete
latent samples, KL balancing, image/reward/discount heads, imagined actor-
critic learning, mixed dynamics/REINFORCE actor gradients, and a slow target
critic.

The UAV benchmark has a parameterized hybrid action space, which official
DreamerV2 does not natively support (it auto-detects purely Discrete or Box
actions). Therefore one explicit extension is necessary: `HybridActionDecoder`
uses a categorical action selection plus continuous per-action parameters.
This is not claimed to be part of upstream DreamerV2. The external
`uav_adapter.py` file is removed; the environment contract is implemented in
`envs.py` as `DreamerV2UAVEnv`.

Default task: `uav_relay`; time limit 100; action repeat 1; batch length 20.
There are no obstacles; map scale and MOVE/TURN/CATCH dynamics are unchanged.
The default `goal_safe_v1` profile changes layout sampling and reward/terminal
semantics as specified below. It is an engineering adaptation, not a claim of
strict paper reproduction. `--reward_mode legacy` retains the previous rules.
The batch length differs from upstream default 50 because paper UAV episodes
can terminate before 50 steps.

Run:

    python dreamer.py --task uav_relay --logdir ./outputs/dv2_relay

Task 1:

    python dreamer.py --task uav_direct --logdir ./outputs/dv2_direct

A small CPU smoke run can be launched with reduced model dimensions:

    python dreamer.py --task uav_relay \
      --logdir ./outputs/v2_fixed_smoke --steps 400 --prefill 200 \
      --eval_every 200 --eval_episodes 2 --batch_size 4 --batch_length 10 \
      --pretrain 2 --rssm_hidden 64 --rssm_deter 64 --rssm_stoch 8 \
      --rssm_discrete 8 --cnn_depth 8 --num_units 64 --vector_units 32

Each log directory contains `metrics.jsonl`, `run_metadata.jsonl`, replay
episodes, and `variables.pkl`. Reusing the same directory restores the
checkpoint instead of silently starting a new policy.

The UAV action contract is `move_turn_parameters_v2`: MOVE and TURN have one
effective parameter each; CATCH has none. Its parameter slot and all unselected
slots are zero in actor outputs, random prefill, environment decoding and replay
canonicalization. CATCH contributes categorical entropy only, never parameter
entropy or dynamics gradients. Parameter magnitude/saturation metrics exclude
CATCH; `parameter_action_fraction` reports the denominator coverage.

`pretrain` now counts world-model-only updates before the first actor/critic
update. Resumed optimizers skip this warmup. Exploration uses independent
`actor_discrete_ent=0.01` and `actor_parameter_ent=0.001` coefficients, replacing
the single `actor_ent` flag. These defaults require learning validation; finite
gradients and passing unit tests alone do not establish policy improvement.

`python controller_baseline.py --episodes 100 --output outputs/controller.json`
checks the task with a hand-coded exact-state feedback controller. It does not
train Dreamer or insert demonstrations into replay. The 100 evaluation seeds
10000 through 10099 achieved 100% pickup/delivery, no boundary failures, and
59.03 mean steps on the current scenario. This is a task sanity baseline,
not a sample-efficiency comparison with the learned agent.

Training corrections (2026-09-10):

- Imagination targets use successor rewards, discounts and values at matching
  indices, including the final imagined transition.
- Reset actions remain zero. Short replay episodes are padded with a `valid`
  mask used by reconstruction, reward, discount, KL and actor/critic objectives.
  Padding and true terminal states cannot seed actor/critic learning.
- The hybrid actor uses one action tanh, smoothly bounded Gaussian scale,
  transformed-action entropy, and a configurable 1% categorical uniform mix
  (`actor_unimix`). These are UAV actor design choices, not upstream claims.
- `replay_valid_fraction` reports padding, and `train/pickup` / `test/pickup`
  distinguish actual CATCH pickup from merely entering the relay radius.

Use a fresh output directory for validation of these corrections. A process
paused before these edits retains the old code and actor in memory; resuming
it does not apply source edits. Loading old actor weights also changes their meaning
because the parameter distribution has changed, so it is not an equivalent
continuation experiment.

## goal_safe_v1 reward and sampling

For the learner's discount gamma (default 0.99), the base reward is
`-(1-gamma)` on continuing steps, `+1` for successful delivery, and `-1`
for either boundary failure or timeout. All episode endings have discount 0;
timeout remains labeled `truncated` for diagnostics but is a real task failure
in this finite-horizon objective. Remaining time is present in the observation.
Boundary violations take precedence over simultaneous geometric success.

Every failed episode has discounted base return exactly -1, independently
of when it ends. Success at step T has return `-1 + 2*gamma**(T-1)`, which
is strictly better for the configured finite horizon, and improves with earlier
delivery. This removes the benefit of deliberate early exit.

Dense shaping adds `shaping_scale * (gamma * Phi(next) - Phi(current))`, with
Phi equal to negative normalized remaining route length and Phi=0 at every
terminal state, including timeout. Its discounted episode sum is exactly
`-shaping_scale * Phi(initial)`. Thus from the same starting state it preserves
the ranking of successful and failed trajectories. The normalization is the
map diagonal times the number of task phases; the default scale is 1.

Start/relay/final points must be separated by more than two goal radii.
The sum of straight-line route legs must fit 70% of the maximum-speed travel
budget for the horizon, reserving headroom for acceleration, turning and CATCH.
This is a geometric feasibility filter, not a reachability proof. Actual CATCH
within the relay radius is still required before delivery can succeed.

Use action_repeat=1. The environment shaping discount is wired directly to the
learner discount. Reusing replay after changing task, horizon, reward mode,
discount or shaping scale is rejected; use a new output directory.
