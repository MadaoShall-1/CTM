# Dreamer V2 long-run engineering hardening

## Scope and findings

This is engineering validation of the two-action automatic-pickup build, not
evidence of policy convergence. The original loader retained RGB images in
two independent full-replay caches. At 2,000,000 transitions, images alone
require 24.576 GB per cache; this workstation's WSL has about 7.5 GB RAM.
Old controller demonstrations were subject to eviction, and online BC reused
the rare-event sampler despite the warm-start sampler being uniform.

Other concrete risks were full replay/event rescans each batch, repeated
directory scans to count progress, direct non-atomic checkpoint/episode
writes, no NaN/Inf stop guard, partial warmup being skipped on resume, and
batch interruption leaving unfinished status or detached children.

## Implemented controls

- Training loads only vector/action/reward/discount/demo/event fields from
  NPZ, without decompressing images. Images remain on disk for diagnostics.
- `episodes/demonstrations/` is the authoritative, permanent teacher store.
  Its root copies preserve historical step accounting. World-model replay
  deduplicates these files and pins them outside the online capacity; BC reads
  only this small immutable store, uniformly, both during warmup and online.
- Online capacity is bounded (plus one boundary episode and pinned demos).
  Unchanged episodes are not reloaded or reindexed. Replay refresh is bounded
  by 10,000 yielded sequences or 10 seconds, whichever occurs first.
- Episode counters are maintained in-process after the initial disk scan;
  evaluation summaries stream JSONL instead of reading the whole history.
- Episodes and checkpoints are fsynced to same-directory temporary files and
  atomically replaced. `variables.prev.pkl` retains the preceding checkpoint.
  Temporary files are not visible to replay. Finalized corrupt files fail
  explicitly instead of causing repeated read loops.
- Non-finite loss/gradient norms stop optimizer updates; non-finite state
  cannot overwrite a good checkpoint. A default 2 GiB free-space guard stops
  writes before exhausting the disk.
- Checkpoints run independently of evaluation, every 1,000 completed steps
  (bounded overshoot by an episode). Model/Actor warmup counters are persisted
  every 100 warmup updates, so interrupted warmup resumes remaining work.
- Provenance is written before data collection. Torn JSONL tails are repaired
  without discarding complete records. Directory locks prevent concurrent
  training or batch runners writing the same run.
- Cleanup closes environment workers with bounded waits. Batch cancellation
  stops its owned process group, marks the run interrupted, and preserves logs.
  `run_batch.py --resume --outdir ...` verifies the frozen source and skips
  completed seeds; it uses original settings and rejects legacy batches.

The training contract is now `bounded_replay_uniform_demo_bc_v1`; the action
contract remains `move_turn_autopickup_v3`. Start a fresh hardened run rather
than mixing old training/replay/checkpoints. No historical results were erased.

## Reproducible validation commands

Run from the project root inside WSL with the configured Python environment.

```bash
bash run_wsl.sh -B -m unittest discover -s tests -v
bash run_wsl.sh -B -m unittest test_navigation_fixes test_replay_sampling -v
python stress_replay.py --output outputs/replay_stress.json
python stress_replay.py --temp-root outputs --output outputs/workspace_replay_stress.json
python long_run_smoke.py --outdir outputs/new_long_run_smoke
```

`stress_replay.py` uses only owned temporary synthetic data and removes it
after validation. Its default fills a 2-million-step cache, then rolls through
2.2 million disk transitions while checking bounded memory, pinned demos and
incremental event indexing. `long_run_smoke.py` uses the **default model,
batch size and imagination horizon**, but only two demonstration episodes,
two updates per warmup phase and a 480-step budget. It deliberately interrupts
its owned training child after a durable warmup checkpoint, then resumes the
same frozen batch and checks completion.

## Observed results

- **87 regression tests passed:** 73 in `tests/`, plus 14 navigation/replay
  tests. Fault-injection cases cover torn writes/logs, corrupt replay, disk
  guards, non-finite loss, partial warmup, duplicate-process locks and batch
  failure/interruption/resume. Compilation and `git diff --check` passed.
- Both the WSL-native and actual E-drive synthetic replay tests filled
  2,000,000 online transitions plus 300 pinned demonstration transitions,
  then added two rounds of 100,000 new online transitions. Cached arrays
  stayed at **169.6 MiB** in every round. This synthetic fixture stores the
  core training/event fields; production's redundant carrying flag and its
  configured demo reserve add a small amount to that figure.
- WSL-native test: **913.6 MiB peak process RSS**; initial load plus 1,000
  sampled windows took **8.21 s**, unchanged-cache rescan/sampling **0.37 s**,
  and each 100,000-transition replacement round **0.83 / 0.59 s**.
- E-drive test: **940.3 MiB peak process RSS**; the same rounds took
  **122.52 s**, **0.39 s**, and **6.25 / 6.04 s**, respectively. File creation
  and temporary cleanup are outside these sampling timings. Use a persistent
  WSL-native directory for active long-run replay when practical; E-drive
  runs remain supported but cold startup/recovery is substantially slower.
- The default-size GPU model (RSSM deter/hidden 400, 32x32 categorical state,
  batch 50x20, imagination horizon 15) completed a real interruption/resume
  check. Interruption occurred at step **187**, with warmup **model=2,
  actor=0**. The same frozen batch resumed, completed the remaining Actor
  warmup and training, and finished at step **487**, warmup **model=2,
  actor=2**, in two attempts. No model-size reduction was used; only demo
  count, warmup duration and total steps were bounded for this test.

Artifacts:

- `outputs/long_run_hardening_20260910/replay_stress.json`
- `outputs/long_run_hardening_20260910/replay_stress_workspace.json`
- `outputs/long_run_hardening_20260910/default_model_resume/smoke_result.json`
- `outputs/long_run_hardening_20260910/default_model_resume/batch_status.json`
  (contains frozen source hashes and exact child command).
- `outputs/long_run_hardening_20260910/default_model_resume/seed_0/console.log`
  and `run_metadata.jsonl` (actual interruption/resume provenance).

All owned synthetic stress directories were cleaned up. Historical runs and
their checkpoints were left intact. These are bounded engineering tests, not
a 2-million-step learned-policy training experiment.

## Remaining limitations

These tests cannot exclude hardware failures, power loss, driver faults or all
multi-hour memory/performance problems. Crash recovery is checkpoint-based,
not bitwise trajectory/RNG continuation. A kill can lose in-progress episodes
and updates since the last durable checkpoint; finalized replay is retained.
SIGKILL cannot run cleanup: if a batch runner is forcibly killed while its
training child survives, the run lock rejects duplicate training; inspect the
surviving process before resuming. Keep the Python/CUDA environment unchanged
through a run and its recovery.
Replay is capacity-bounded in RAM, not deleted from disk; the disk guard stops
a run rather than reclaiming user experiment data. A storage device failure
requires an independent backup. A prior checkpoint is available for manual
recovery if needed.

The existing BC scale and imagination weighting were not tuned here. Long-run
policy collapse, overreliance on demonstrations and world-model inaccuracy
still require multi-seed learning curves and independent evaluation. No full
multi-million-step training was started by this hardening task.
