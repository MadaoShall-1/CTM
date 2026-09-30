"""CT-WM ablations: the unmodified train_nrsm_online.py with selected runtime changes.

- --imagination-scale S      replaces the default ACConfig.imagination_scale (0.1).
- --learned-reward-std MIN   replaces the fixed unit-variance reward head with a learned-std Gaussian
                             (see ablation_patches.py).
- --delta-head W             adds W * (normalised one-step motion-change NLL) to the world-model loss
                             (see ablation_patches.py).
- --normalised-reward        the reward head predicts symlog((r - mu) / sigma) (see ablation_patches.py).
- --twohot-reward            the reward head is a two-hot categorical over normalised symlog bins, with its
                             mean taken in original units (see ablation_patches.py).
- --edge-head W              adds W * (NLL of the distances to the four map edges, symlog-scaled) to the
                             world-model loss (see ablation_patches.py).
- --stoch N --classes N      the compact world model's stochastic state becomes N categorical variables of N
                             classes (compact default: 8 x 8); every other size stays compact.
- --demonstrations           DreamerV2's 64 controller demonstrations in world-model training and its 5x
                             behaviour-cloning actor term (see ablation_patches.py).

All other arguments go to the trainer unchanged. The trainer records the resolved ac_config in the run's
manifest.json; this wrapper also writes <output>.ablation.json (the changes and this file's SHA-256) next to
the run directory. With only --imagination-scale, behaviour is identical to this file's first revision
(git history), which the imagination_scale=1.0 runs used.

Usage (from "Dreamer V2/ctm_qiwei/Dreamer-master", with the TensorFlow venv):
    python -B -u <path>/run_ablation.py --imagination-scale 1.0 [--learned-reward-std 0.01] \
        --output outputs/<new> --steps 137680 --seed 17
"""
import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import ablation_patches  # noqa: F401  (puts the DV2 folder on sys.path)
from ablation_patches import (BC_SCALE, CLASSES_KEY, DELTA_KEY, DEMO_EPISODES, DEMO_KEY, DEMO_SEED_START,
                              DEMO_SHARE, DV2, EDGE_KEY, REWARD_NORM_KEY, REWARD_STD_KEY, REWARD_TWOHOT_KEY,
                              STOCH_KEY)

import nrsm_online_agent  # noqa: E402
import train_nrsm_online  # noqa: E402


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--imagination-scale', type=float)
    parser.add_argument('--learned-reward-std', type=float, metavar='MIN_STD')
    parser.add_argument('--delta-head', type=float, metavar='WEIGHT')
    parser.add_argument('--normalised-reward', action='store_true')
    parser.add_argument('--twohot-reward', action='store_true')
    parser.add_argument('--edge-head', type=float, metavar='WEIGHT')
    parser.add_argument('--stoch', type=int)
    parser.add_argument('--classes', type=int)
    parser.add_argument('--demonstrations', action='store_true')
    known, trainer_args = parser.parse_known_args()
    default = nrsm_online_agent.ACConfig()
    config = default
    changed = {}
    if known.imagination_scale is not None:
        config = nrsm_online_agent.ACConfig(imagination_scale=known.imagination_scale)
        changed['ACConfig.imagination_scale'] = [default.imagination_scale, config.imagination_scale]
    if known.learned_reward_std is not None:
        if known.learned_reward_std <= 0:
            parser.error('--learned-reward-std must be positive')
        changed[REWARD_STD_KEY] = [None, known.learned_reward_std]
    if known.delta_head is not None:
        if known.delta_head <= 0:
            parser.error('--delta-head must be positive')
        changed[DELTA_KEY] = [None, known.delta_head]
    if known.normalised_reward:
        if known.learned_reward_std is not None:
            parser.error('--normalised-reward and --learned-reward-std both replace the reward head')
        changed[REWARD_NORM_KEY] = [False, True]
    if known.twohot_reward:
        if known.learned_reward_std is not None or known.normalised_reward:
            parser.error('--twohot-reward cannot be combined with another reward-head change')
        changed[REWARD_TWOHOT_KEY] = [False, True]
    if known.edge_head is not None:
        if known.edge_head <= 0:
            parser.error('--edge-head must be positive')
        changed[EDGE_KEY] = [None, known.edge_head]
    if known.demonstrations:
        changed[DEMO_KEY] = [None, dict(episodes=DEMO_EPISODES, seed_start=DEMO_SEED_START, bc_scale=BC_SCALE,
                                        world_model_share=round(DEMO_SHARE, 4))]
    if (known.learned_reward_std is not None or known.delta_head is not None or known.normalised_reward
            or known.twohot_reward or known.edge_head is not None or known.demonstrations):
        # The demonstrations' own random stream is seeded by the trainer's --seed (default 17).
        seed = int(trainer_args[trainer_args.index('--seed') + 1]) if '--seed' in trainer_args else 17
        ablation_patches.apply(known.learned_reward_std, known.delta_head, known.normalised_reward,
                               known.twohot_reward, known.edge_head, seed if known.demonstrations else None)
    core = None
    if known.stoch is not None or known.classes is not None:
        if min(x for x in (known.stoch, known.classes) if x is not None) < 1:
            parser.error('--stoch and --classes must be positive')
        compact, core = nrsm_online_agent.compact_config(), ablation_patches.core_config(known.stoch, known.classes)
        for key, name in ((STOCH_KEY, 'stoch'), (CLASSES_KEY, 'classes')):
            if getattr(core, name) != getattr(compact, name):
                changed[key] = [getattr(compact, name), getattr(core, name)]
    if not changed:
        parser.error('Specify at least one ablation')
    base = nrsm_online_agent.OnlineAgent

    def agent_factory(core_config=None, ac_config=None):
        # Explicit configs (checkpoint-restore verification) pass through unchanged.
        return base(core_config or core, ac_config or config)

    train_nrsm_online.OnlineAgent = agent_factory
    output = Path(trainer_args[trainer_args.index('--output') + 1])
    sidecar = output.with_name(output.name + '.ablation.json')
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps(dict(
        wrapper=Path(__file__).name,
        wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        patches_sha256=hashlib.sha256(Path(ablation_patches.__file__).read_bytes()).hexdigest(),
        changed=changed, ac_config=asdict(config), trainer_args=trainer_args), indent=2), encoding='utf-8')
    sys.argv = [str(DV2 / 'train_nrsm_online.py')] + trainer_args
    train_nrsm_online.main()


if __name__ == '__main__':
    main()
