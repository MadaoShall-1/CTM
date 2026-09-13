# DreamerV2 UAV migration

This build replaces the previous DreamerV1 Gaussian RSSM with the official
DreamerV2 algorithmic structure: categorical RSSM, straight-through discrete
latent samples, KL balancing, vector/reward/discount heads, imagined actor-
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
There are no obstacles; map scale and MOVE/TURN dynamics are unchanged.
The CATCH action has been removed. Pickup occurs automatically at the
post-motion position within the inclusive 100 m relay radius, provided the
step is not an out-of-bounds failure. Cargo position, phase, and active goal
change on that same step, without an extra action or pickup reward. A segment
that passes through the disk but ends outside does not trigger pickup.
The default `goal_safe_v1` profile changes layout sampling and reward/terminal
semantics as specified below. It is an engineering adaptation, not a claim of
strict paper reproduction. `--reward_mode legacy` retains previous reward and
sampling rules, not the removed explicit-CATCH task.
The batch length differs from upstream default 50 because paper UAV episodes
can terminate before 50 steps.

Run:

    python dreamer.py --task uav_relay --logdir ./outputs/dv2_relay_autopickup_v3

Task 1:

    python dreamer.py --task uav_direct --logdir ./outputs/dv2_direct

A small CPU smoke run can be launched with reduced model dimensions:

    python dreamer.py --task uav_relay \
      --logdir ./outputs/v2_fixed_smoke --steps 400 --prefill 200 \
      --eval_every 200 --eval_episodes 2 --batch_size 4 --batch_length 10 \
      --demo_episodes 2 --actor_pretrain 2 --pretrain 2 \
      --rssm_hidden 64 --rssm_deter 64 --rssm_stoch 8 \
      --rssm_discrete 8 --cnn_depth 8 --num_units 64 --vector_units 32

Each log directory contains `metrics.jsonl`, `run_metadata.jsonl`, replay
episodes, and `variables.pkl`. Reusing the same directory restores the
checkpoint instead of silently starting a new policy.

Long-run engineering updates require the training contract
`bounded_replay_uniform_demo_bc_v1`. Earlier automatic-pickup runs are also
incompatible with this new training/checkpoint contract; retain them as
historical experiments and use a fresh directory. Demonstrations are now
permanent under `episodes/demonstrations`, and online BC as well as warmup
uses their separate uniform sampler. Image arrays are never loaded into the
training replay cache. The online capacity excludes pinned demonstrations.
See `LONG_RUN_HARDENING_20260910.md` for checkpoint/resume and stress tests.

The UAV action contract is `move_turn_autopickup_v3`. Direct, relay, and
multi-relay environments expose only MOVE and TURN; each has one effective
parameter. The flattened action has 4 channels instead of 6. Unselected
parameter slots remain zero in actor outputs, random prefill, environment
decoding and replay canonicalization. The Actor has no pickup mask or
pickup-specific cloning weight. All executed actions have a parameter.

Old replay and checkpoints must not be resumed or relabelled as v3. Start a
fresh output directory; the controller generates new two-action demonstrations.
Checkpoints now carry action/observation/task contracts, checked before loading
variables. Old experiments remain evaluable using their frozen source.

`pretrain` now counts world-model-only updates before the first actor/critic
update and defaults to 1000. Resume uses explicit model/Actor warmup counters
and finishes any remaining warmup, rather than skipping partially completed
warmup based on optimizer iterations. Exploration
uses independent `actor_discrete_ent=0.03` and `actor_parameter_ent=0.02`
coefficients plus a 5% categorical uniform mixture. Continuous imagination
gradients are scaled by 0.1.

`python controller_baseline.py --episodes 100 --output outputs/controller.json`
checks the task with a hand-coded exact-state feedback controller. A fresh
Dreamer run seeds replay with 64 tagged successful controller episodes. They
train the world model and an Actor behavior-cloning auxiliary
loss. The Actor receives 3000 demonstration-only warm-start updates before
imagined-return training begins; imagined returns remain the RL objective. The
historical explicit-CATCH controller's 100 evaluation seeds
10000 through 10099 achieved 100% pickup/delivery, no boundary failures, and
59.03 mean steps. This is a task sanity baseline for that former task,
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
  report the automatic transition to carrying the supply.
- The model consumes the exact 13-value navigation vector rather than spending
  capacity reconstructing a 64x64 raster. Heading uses sin/cos, both fixed goals
  and the active-goal offset are retained, and the phase is encoded as -1/+1.
- Half of world-model replay windows are anchored on a class-balanced choice
  among pickup, success and terminal events. Actor warm-start batches use
  uniform demonstration sampling to preserve the action prior. Cloning uses
  categorical cross-entropy and bounded parameter MSE; a squashed-Gaussian
  likelihood is deliberately avoided at controller targets near +/-1.
- The Actor consumes the exact current vector in real interaction and the
  decoded vector in imagination. The critic and dynamics remain latent-state
  based, while control does not discard exact state at deployment.
- Pickup is entirely an environment transition. No Actor action is forced
  inside the pickup radius. Imagined-return gradients remain scaled by 0.1.

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
budget for the horizon, reserving headroom for acceleration and turning.
This is a geometric feasibility filter, not a reachability proof. Automatic
pickup within the relay radius is still required before delivery can succeed.

Use action_repeat=1. The environment shaping discount is wired directly to the
learner discount. Reusing replay after changing task, horizon, reward mode,
discount or shaping scale is rejected; use a new output directory.
