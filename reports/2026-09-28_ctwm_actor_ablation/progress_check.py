"""Do training episodes end closer to the active goal than they start? (Exploratory, after experiment 5; NumPy only.)

For the last quarter of each run's policy (non-prefill) training episodes, reads the saved episodes (read-only)
and reports the distance to the active goal at the first and last valid rows (vector dims 11-12, x 2000 m),
their difference ("progress", positive = ended closer), and the same split by how the episode ended
(timeout or out of bounds). Standard errors are across episodes.

Usage (any Python with numpy):
    python progress_check.py out.json <prefix>=<run_dir> [<prefix>=<run_dir> ...]
"""
import json
import sys
from pathlib import Path

import numpy as np


def summary(rows):
    if not rows:
        return None
    start, end = np.array([r[0] for r in rows]), np.array([r[1] for r in rows])
    progress = start - end
    return dict(episodes=len(rows), start_m=float(start.mean()), end_m=float(end.mean()),
                progress_m=float(progress.mean()), progress_se_m=float(progress.std(ddof=1) / np.sqrt(len(rows))))


def check(run_dir):
    run_dir = Path(run_dir)
    outcome = {}
    with (run_dir / 'metrics.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            record = json.loads(line)
            if record['kind'] == 'episode' and record.get('phase') == 'policy':
                outcome[int(record['metrics']['scene'])] = record['metrics']
    rows = []
    for path in sorted(run_dir.glob('episodes/*.npz'), key=lambda p: int(p.stem)):
        metrics = outcome.get(int(path.stem))
        if metrics is None:
            continue
        episode = np.load(path)
        vector = episode['vector'][episode['valid'] > 0]
        distance = np.hypot(vector[:, 11], vector[:, 12]) * 2000
        rows.append((float(distance[0]), float(distance[-1]), metrics))
    last_quarter = rows[3 * (len(rows) // 4):]   # same split as the README's behaviour tables
    split = {name: [r for r in last_quarter if r[2][name]] for name in ('timeout', 'oob')}
    return dict(policy_episodes=len(rows), last_quarter=summary(last_quarter),
                timeout_share=len(split['timeout']) / len(last_quarter),
                oob_share=len(split['oob']) / len(last_quarter),
                timeout=summary(split['timeout']), oob=summary(split['oob']))


def main(output, *runs):
    result = {}
    for item in runs:
        prefix, run_dir = item.split('=', 1)
        result[prefix] = check(run_dir)
        q, t = result[prefix]['last_quarter'], result[prefix]['timeout']
        print(f"{prefix:36s} progress {q['progress_m']:+5.0f} ± {q['progress_se_m']:.0f} m | "
              f"timeouts ({100 * result[prefix]['timeout_share']:.0f}%) {t['progress_m']:+5.0f} ± {t['progress_se_m']:.0f} m")
    Path(output).write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main(*sys.argv[1:])
