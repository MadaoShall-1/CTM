"""Condense a train_nrsm_online.py output directory into small, committable files.

Usage (from this folder):
    python extract_ctwm_run.py "<DV2>/outputs/<run>" ctwm_online_1h [output_dir]

Writes <prefix>_evaluations.json, <prefix>_train_log.csv, <prefix>_episodes.csv,
and copies result.json / manifest.json with local absolute paths removed.
output_dir defaults to this folder. Standard library only.
"""
import csv
import json
import re
import sys
from pathlib import Path

TRAIN_KEYS = ('model_loss', 'vector_nll', 'reward_nll', 'discount_bce', 'kl', 'actor_loss',
              'critic_loss', 'imagined_return', 'imagined_discount', 'model_gradient_norm',
              'actor_gradient_norm', 'critic_gradient_norm', 'delta_nll', 'edge_nll', 'actor_bc_loss')
EPISODE_KEYS = ('scene', 'success', 'pickup', 'timeout', 'oob', 'length', 'return_')


def scrub(text):
    """Drop machine-specific absolute prefixes so the files are portable."""
    return re.sub(r'/(?:Users|home)/[^"\s]*?/(?=outputs/|Dreamer V2/|her_mpdqn_reproduction/)', '', text)


def main(run_dir, prefix, output_dir=None):
    run_dir = Path(run_dir)
    out = Path(output_dir) if output_dir else Path(__file__).resolve().parent
    evaluations, train_rows, episode_rows = [], [], []
    with (run_dir / 'metrics.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            record = json.loads(line)
            counters = {k: record.get(k) for k in ('env_steps', 'model_updates', 'ac_updates')}
            counters['elapsed_seconds'] = round(record.get('elapsed_seconds', 0.0), 1)
            metrics = record.get('metrics', {})
            if record['kind'] == 'evaluation':
                evaluations.append(dict(
                    split=record['split'], **counters,
                    **{k: metrics.get(k) for k in ('count', 'success', 'pickup', 'timeout', 'oob',
                                                    'length', 'return_')},
                    scenes=[{k: e[k] for k in EPISODE_KEYS} for e in metrics.get('episodes', [])]))
            elif record['kind'] == 'train' and metrics:
                train_rows.append(dict(phase=record.get('phase'), **counters,
                                       **{k: metrics.get(k) for k in TRAIN_KEYS}))
            elif record['kind'] == 'episode':
                episode_rows.append(dict(phase=record.get('phase'), **counters,
                                         **{k: metrics.get(k) for k in EPISODE_KEYS}))
    (out / f'{prefix}_evaluations.json').write_text(json.dumps(evaluations, indent=1), encoding='utf-8')
    for name, rows in (('train_log', train_rows), ('episodes', episode_rows)):
        with (out / f'{prefix}_{name}.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    for name in ('result.json', 'manifest.json'):
        if (run_dir / name).exists():
            (out / f'{prefix}_{name}').write_text(
                scrub((run_dir / name).read_text(encoding='utf-8')), encoding='utf-8')
    print(f'{len(evaluations)} evaluations, {len(train_rows)} train logs, {len(episode_rows)} episodes')


if __name__ == '__main__':
    main(*sys.argv[1:4])
