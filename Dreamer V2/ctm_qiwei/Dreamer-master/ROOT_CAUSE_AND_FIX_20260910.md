# DreamerV2 UAV engineering root cause and fix

Historical report for the **explicit-CATCH, six-channel** task. The current
`move_turn_autopickup_v3` build removes CATCH and uses automatic pickup; the
metrics and CATCH-specific changes below describe the former frozen runs, not
validation of the new task. See README_MIGRATION.md for the current contract.

## Finding

The zero-pickup/zero-success result was not one isolated hyperparameter issue.
It was a causal chain across observation modeling, replay, and Actor training:

1. Heading was represented as one linear value, making physically adjacent
   angles near -pi/+pi appear maximally far apart. The model also spent most of
   its reconstruction capacity on a 12,288-value raster although the exact
   control state was already available.
2. Uniform replay almost never presented pickup or success transitions. Short
   terminal episodes were previously at risk of being excluded or incorrectly
   treated as full sequences.
3. Actor learning began after only 100 world-model updates. In the first probe,
   its continuous parameters went from 0.17 mean magnitude to about 0.99 and
   deterministic evaluation became about 95% MOVE within a few hundred
   updates. This is direct evidence of exploitation of an inaccurate model.
4. Raising pretraining to 1000 and strengthening entropy reduced parameter
   saturation to roughly 6-8%, but deterministic evaluation still had 0%
   pickup/success. Successful controller episodes trained only the world model;
   they supplied no direct TURN/CATCH target to the Actor. In addition, the
   Actor could only read the stochastic RSSM feature even though an exact
   compact control vector exists at deployment.
5. A first behavior-cloning implementation using squashed-Gaussian likelihood
   reproduced the saturation problem because controller targets at +/-1 map
   through `atanh`. It was rejected and replaced rather than retained.
6. Replay stores `action[t]` as the transition that produced `observation[t]`.
   The first cloning pass incorrectly learned `observation[t] -> action[t]`,
   pairing every target with its successor state. Its loss decreased while the
   deterministic policy collapsed to TURN. The corrected pair is
   `observation[t] -> action[t+1]`.

## Implemented correction

- A fixed 13-value topology-safe vector is now the sole model observation:
  position, speed, sin/cos heading, both fixed-goal offsets, remaining time,
  phase, and active-goal offset.
- The image encoder/decoder and pixel reconstruction loss were removed from the
  learner. Raster observations remain available for visual diagnostics.
- Replay preserves short episodes with an explicit `valid` mask and devotes
  half of samples to windows anchored on a class-balanced pickup, success, or
  terminal event.
- A fresh run adds 64 successful, tagged controller episodes to replay.
- The Actor receives an auxiliary demonstration loss on posterior states. The
  loss retains demonstrated action frequencies and regresses bounded executed
  parameters with MSE, avoiding both class-prior distortion and the +/-1
  likelihood singularity. Actor warm start uses a separate uniform sampler;
  rare-event priority remains limited to world-model training.
- Demonstration actions are shifted against replay observations so policy
  targets are aligned with their source states rather than successor states.
- Before imagined-return updates, the Actor receives 3000 demonstration-only
  warm-start updates. Its input is the exact current vector during real
  interaction and the world-model vector prediction during imagination.
- The same semantic action mask is applied in deployment and imagination:
  CATCH is forced inside the environment's 100-meter relay radius (0.05 after
  map normalization), and unavailable outside it or after pickup. Early
  imagined-return gradients are scaled by 0.1.
- World-model-only warmup is 1000 updates. The continuous imagination gradient
  is scaled to 0.1, entropy is stronger, and every discrete action retains at
  least 5% mixture exploration.
- RSSM inputs are canonicalized so parameters ignored by the environment cannot
  alter imagined dynamics.

## Acceptance criteria

Unit tests cover vector continuity, action canonicalization, rare-event replay,
demonstration tagging/class balance, padding masks, objective alignment, and
checkpoint integration. End-to-end acceptance additionally requires that a
fresh GPU run does not show rapid Actor parameter saturation and produces
non-zero pickup/success under deterministic fixed-seed evaluation. Passing
tests alone is not treated as evidence that the navigation policy learned.

## Validation result

The final frozen-source probe completed at 6,212 steps. Independent deterministic
evaluation over seeds 10000-10099 produced 62% pickup, 16% delivery success, 2%
out-of-bounds, and 13.7% selected-parameter saturation. The earlier validated
20,044-step checkpoint produced 0% pickup and 0% success on the same 100-seed
protocol. This establishes that the engineering deadlock is fixed; it does not
claim the policy is converged, since 82% of final episodes still timed out.

The inference-only world-model diagnostic confirms the remaining limitation at
this short horizon. Its MOVE(+1)-MOVE(-1) predicted speed difference is 3.30
versus the physical 8.0, and its TURN(+1)-TURN(-1) heading response is 4.35
degrees versus the physical 120 degrees (about 72% direction correctness).
Consequently the current policy improvement is primarily the repaired control
and demonstration path; longer multi-seed training is still required before
claiming that latent imagination itself is accurate.

Artifacts:

- `outputs/v2_rootfix_probe11_20260910/final_evaluation_100.json`
- `outputs/v2_rootfix_probe11_20260910/batch_status.json`
- `outputs/v2_rootfix_probe11_20260910/seed_0/metrics.jsonl`
- `outputs/v2_rootfix_probe11_20260910/world_model_diagnostic.json`
