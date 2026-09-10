# Dream to Control

**NOTE:** Check out the code for [DreamerV2](https://github.com/danijar/dreamerv2), which supports both Atari and DMControl environments.

Fast and simple implementation of the Dreamer agent in TensorFlow 2.

<img width="100%" src="https://imgur.com/x4NUHXl.gif">

If you find this code useful, please reference in your paper:

```
@article{hafner2019dreamer,
  title={Dream to Control: Learning Behaviors by Latent Imagination},
  author={Hafner, Danijar and Lillicrap, Timothy and Ba, Jimmy and Norouzi, Mohammad},
  journal={arXiv preprint arXiv:1912.01603},
  year={2019}
}
```

## Method

![Dreamer](https://imgur.com/JrXC4rh.png)

Dreamer learns a world model that predicts ahead in a compact feature space.
From imagined feature sequences, it learns a policy and state-value function.
The value gradients are backpropagated through the multi-step predictions to
efficiently learn a long-horizon policy.

- [Project website][website]
- [Research paper][paper]
- [Official implementation][code] (TensorFlow 1)

[website]: https://danijar.com/dreamer
[paper]: https://arxiv.org/pdf/1912.01603.pdf
[code]: https://github.com/google-research/dreamer

## Instructions for this workstation

The supported GPU path is Ubuntu on WSL2. Native Windows TensorFlow can run
the code on CPU, but does not provide current NVIDIA GPU support.

Create the WSL environment once:

```bash
python3 -m venv /home/madao/.venvs/ctm-dreamer
/home/madao/.venvs/ctm-dreamer/bin/python -m pip install --upgrade pip
/home/madao/.venvs/ctm-dreamer/bin/python -m pip install -r requirements.txt
```

Check the WSL driver, TensorFlow GPU visibility, and actual GPU matrix
multiplication and convolution:

```bash
bash run_wsl.sh --check-gpu --require-gpu
```

Train from WSL with a GPU check before training. `CTM_REQUIRE_GPU=1` stops
the launch if the check fails; without it, CPU fallback is allowed:

```bash
CTM_REQUIRE_GPU=1 bash run_wsl.sh -u dreamer.py \
  --logdir ./outputs/dreamerv2_uav_relay \
  --task uav_relay
```

Run the engineering regression tests from the `Dreamer-master` directory:

```bash
bash run_wsl.sh -m unittest discover -s tests -v
```

The UAV tasks are available as `uav_direct` and `uav_relay`; the relay task
is the default. No obstacles are added. The default reward/sampling profile
is `goal_safe_v1`; `--reward_mode legacy` selects the previous rules.

```bash
python dreamer.py \
  --logdir ./outputs/dreamerv2_uav_relay \
  --task uav_relay \
  --action_repeat 1 \
  --time_limit 100
```

The adapter represents a hybrid action with `K` continuous selection channels
followed by `K` parameter channels. The selected discrete action is the argmax
of the first group, and only its matching parameter is passed to the benchmark.
It renders the structured navigation state into the 64x64 RGB input. A
normalized vector containing state, goals, and phase is also encoded and
reconstructed by the world model, rather than merely being retained in replay.

UAV runs append episode and optimization records to `metrics.jsonl`, including
return, length, success, relay, out-of-bounds, truncation, MOVE/TURN/CATCH
fractions, selected-parameter magnitude and saturation, losses, gradient
norms, and categorical entropy. The resolved
configuration and runtime provenance are appended to `run_metadata.jsonl`.
`base_return`, `shaping_return`, and `discounted_return` separate the objective
from shaping; compare success/pickup rates first. Raw undiscounted return alone
does not represent the discounted training objective.

For a frozen-source batch of three independent seeds on one GPU:

```bash
python run_batch.py --outdir ./outputs/goal_safe_batch --seeds 0 1 2 --steps 100000
```

The runner launches one seed at a time, requires a passing GPU check, records
PIDs and status in `batch_status.json`, and stops the queue if a run fails.
See `README_MIGRATION.md` for the reward definition and horizon semantics.

The [2026-09-10 validation report](outputs/v2_corrected_validation_20260910/VALIDATION.md)
records a completed 20,044-step single-seed run and a fixed 100-episode
checkpoint evaluation. All 22 engineering tests passed, but the learned policy
achieved 0% pickup and delivery; this is not evidence of task learning.
The hand-coded exact-state controller is a feasibility check, not a learned
policy or a sample-efficiency comparison.

Evaluate a locally retained frozen-source run without updating its weights:

```bash
CTM_REQUIRE_GPU=1 bash run_wsl.sh -u evaluate_checkpoint.py \
  --run ./outputs/goal_safe_batch/seed_0 \
  --episodes 100 --seed-start 10000 \
  --output ./outputs/goal_safe_batch/final_evaluation_100.json
```

Git includes selected lightweight validation records only. Checkpoints, replay,
frozen source copies, and runtime logs remain local; a fresh clone does not
contain the trained model needed to rerun its checkpoint evaluation.

The launcher locates the virtual environment's site-packages dynamically
and exposes its pip-installed CUDA libraries plus `/usr/lib/wsl/lib` before
TensorFlow imports. Override `CTM_VENV_PATH` to use another venv. The RTX 5070
Ti uses PTX JIT with the current TensorFlow wheel, so the first GPU operation
can take substantially longer than later runs.

Host-oriented defaults are `float32`, batch size 32, one environment process,
dataset prefetch 2, and a 100k-transition in-memory replay cache. All remain
overridable through command-line flags.

Generate plots:

```
python3 plotting.py --indir ./logdir --outdir ./plots --xaxis step --yaxis test/return --bins 3e4
```

Graphs and GIFs:

```
tensorboard --logdir ./logdir
```
