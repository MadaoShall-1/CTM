"""Per-episode action statistics of a train_nrsm_online.py run (read-only, NumPy only).

Usage:  python action_stats.py "<DV2>/outputs/<run>" ctwm_online_1h_action_stats.csv
Episodes are the saved <scene>.npz files, one per completed training episode. Row 0 is the reset row.
"""
import csv
import sys
from pathlib import Path

import numpy as np


def main(run_dir, output):
    rows = []
    for path in sorted(Path(run_dir, 'episodes').glob('*.npz'), key=lambda p: int(p.stem)):
        episode = np.load(path)
        valid = episode['valid'] > 0
        # Row 0 is the reset observation with no preceding action: drop it from both, so every
        # statistic covers the same post-action steps.
        action = episode['action'][valid][1:]
        vector = episode['vector'][valid][1:]
        move, turn = action[:, 0] > .5, action[:, 1] > .5
        rows.append(dict(
            scene=int(path.stem), steps=len(action), move_share=float(move.mean()),
            mean_move_param=float(action[move, 2].mean()) if move.any() else '',
            mean_abs_turn_param=float(np.abs(action[turn, 3]).mean()) if turn.any() else '',
            mean_speed_mps=float(((vector[:, 2] + 1) / 2 * 40).mean()),
            final_distance_to_active_goal_m=float(np.hypot(*vector[-1, 11:13]) * 2000)))
    with open(output, 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'{len(rows)} episodes -> {output}')


if __name__ == '__main__':
    main(*sys.argv[1:3])
