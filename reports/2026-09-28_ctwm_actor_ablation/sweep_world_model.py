"""World-model sweep: score every checkpoint of a CT-WM run on validate_nrsm.py's held-out episodes.

Same data (seeds 40000-40007) and metrics as ../2026-09-28_first_run_mac/eval_ctwm_world_model.py,
but the agent is rebuilt from the run's manifest and ablation sidecar, so ablation checkpoints load
into the right architecture. Read-only.

Usage (with the TensorFlow venv):  python -B sweep_world_model.py <run_dir> out.json
"""
import json
import sys
from pathlib import Path

import ablation_patches  # noqa: E402  (puts the DV2 folder on sys.path)
import tensorflow as tf  # noqa: E402

from validate_nrsm import collect, evaluate  # noqa: E402


def main(run_dir, output):
    tf.keras.mixed_precision.set_global_policy('float32')
    tf.keras.utils.set_random_seed(17)
    agent = ablation_patches.agent_for_run(run_dir)
    heldout, records = collect(8, 40000)
    heldout = {k: tf.constant(v) for k, v in heldout.items()}
    results = []
    for checkpoint in sorted(Path(run_dir).glob('checkpoint_env*.pkl')):
        state = agent.load(checkpoint)
        results.append(dict(checkpoint=checkpoint.name, counts=state['counts'],
                            heldout_seeds=[r['seed'] for r in records],
                            metrics=evaluate(agent.world, heldout)))
        print(checkpoint.name, round(results[-1]['metrics']['open_loop_15_vector_rmse'], 3), flush=True)
    Path(output).write_text(json.dumps(results, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main(*sys.argv[1:3])
