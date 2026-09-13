# Two-action automatic pickup: implementation and validation

## Current contract

`move_turn_autopickup_v3` replaces the explicit-CATCH task. MOVE and TURN are
the only actions in direct, relay, and multi-relay environments. The Dreamer
action is `[select_MOVE, select_TURN, param_MOVE, param_TURN]`, width 4.

At each post-motion state, an in-bounds UAV at distance <= 100 m from the
relay automatically picks up the supply. Phase, cargo position, achieved goal,
and active goal update on that same step. There is no extra pickup action,
forced Actor decision, or pickup reward. Delivery still requires pickup first.
Boundary failures take precedence; a last-step pickup alone remains a timeout.
The radius test uses the step endpoint, not continuous segment intersection.

The shared action encoding, random prefill, Actor, RSSM, demonstrations,
metrics and diagnostic action probes now use two actions. CATCH-specific
masking and cloning weights were removed. Archived experiment data and frozen
sources were not changed.

## Compatibility

Start a fresh log directory. The new default is
`outputs/dreamerv2_uav_relay_autopickup_v3`.

Old replay provenance, six-channel actions, legacy checkpoints, and attempts
to override the action contract are rejected. New checkpoints include the
action/observation/task contracts and validate variable shapes before loading.
Old checkpoints can still be inspected with their own frozen source, but
must not be mixed into automatic-pickup training. `reward_mode=legacy` does
not restore the removed action.

## Validation

- 65 distinct regression tests cover environment semantics, replay/action
  alignment, finite training gradients, checkpoint save/load, migration guards
  and current/historical diagnostic decoding.
- The WSL TensorFlow GPU check passed matrix multiplication and convolution
  on the RTX 5070 Ti Laptop GPU.
- An exact-state controller completed pickup and delivery in 100/100 scenarios
  with seeds 50000-50099, zero out-of-bounds failures, and 61.36 mean steps.
  This is a hand-coded feasibility check, not learned-policy performance.
- A frozen-source, reduced-model GPU smoke run reached 387 stored transitions
  (87 demonstration, 100 random prefill, 200 online), saved a checkpoint, and
  resumed for another 100 online transitions to step 487. Metadata records
  `resumed_checkpoint=false` followed by `true`. Losses and gradients were
  finite. Online evaluation success remained zero, as expected to be possible
  in such a tiny engineering smoke test; no convergence claim is made.
- Inference-only diagnostics completed on two replay episodes, one held-out
  random episode and one held-out controller episode. Both action probes have
  six parameter settings (three MOVE, three TURN), with no CATCH. Checkpoint
  and optimizer state remained unchanged by the diagnostic.
- Python compilation and `git diff --check` passed.

Tests (from the project root):

```bash
bash run_wsl.sh -B -m unittest discover -s tests -v
bash run_wsl.sh -B -m unittest test_navigation_fixes test_replay_sampling -v
```

Artifacts are under `outputs/v3_auto_pickup_smoke_20260910/`:

- `source/`: frozen training source, checked against the live modules.
- `controller_100.json`: controller feasibility results.
- `checkpoint_step_387.pkl`: pre-resume checkpoint copy.
- `seed_0/variables.pkl`, `seed_0/metrics.jsonl`, `seed_0/run_metadata.jsonl`:
  final step-487 checkpoint, training metrics, and resume provenance.
- `diagnostic_smoke.json`: inference-only diagnostic and integrity checks.

Final smoke checkpoint SHA256:
`9021829c4592754feb2e6a8e6fa337e028a369ae3ed6e84b046a2545e1210b85`.

No full-scale training was launched. This change simplifies the task; it does
not establish that the world model is accurate or that the learned policy is
ready for a multi-million-step run.
