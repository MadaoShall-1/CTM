"""Score a CT-WM online checkpoint's world model on validate_nrsm's held-out episodes.

Uses the same data (8 episodes, seeds 40000-40007, half controller / half random) and the
same metrics as validate_nrsm.py, so numbers are directly comparable with the offline check.
Read-only: loads the checkpoint, never trains or writes to the run directory.

Usage (with the TensorFlow venv; paths may be relative to the current directory):
    python -B eval_ctwm_world_model.py <run>/checkpoint_envXXXXXXX_acXXXXXX.pkl out.json
"""
import json
import sys
from pathlib import Path

DV2 = Path(__file__).resolve().parents[2] / 'Dreamer V2' / 'ctm_qiwei' / 'Dreamer-master'
sys.path.insert(0, str(DV2))

import tensorflow as tf  # noqa: E402

from nrsm_online_agent import OnlineAgent  # noqa: E402
from validate_nrsm import collect, evaluate  # noqa: E402


def main(checkpoint, output):
    tf.keras.mixed_precision.set_global_policy('float32')
    tf.keras.utils.set_random_seed(17)
    agent = OnlineAgent()
    training_state = agent.load(checkpoint)
    heldout, records = collect(8, 40000)
    heldout = {k: tf.constant(v) for k, v in heldout.items()}
    result = dict(checkpoint=Path(checkpoint).name, counts=training_state['counts'],
                  heldout_seeds=[r['seed'] for r in records],
                  metrics=evaluate(agent.world, heldout))
    Path(output).write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main(*sys.argv[1:3])
