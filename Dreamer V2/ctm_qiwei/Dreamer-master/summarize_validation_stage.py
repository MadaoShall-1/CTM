"""Read-only Stage 1 integrity/metric audit; writes only a compact evidence copy."""
import argparse
import collections
import datetime
import json
import math
import pathlib
import pickle
import shutil
import statistics

import numpy as np

from run_support import read_jsonl, atomic_write
from run_validation_stage import digest


def avg(rows, key):
  values = [row[key] for row in rows if key in row]
  return statistics.mean(values) if values else None


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--root', type=pathlib.Path, required=True)
  parser.add_argument('--output', type=pathlib.Path, required=True)
  args = parser.parse_args()
  root = args.root.resolve()
  state = json.loads((root / 'stage_status.json').read_text())
  batch = json.loads((root / 'batch_status.json').read_text())
  manifest = json.loads((root / 'checkpoint_manifest.json').read_text())
  assert state['status'] == batch['status'] == 'completed'
  for name, expected in batch['source_sha256'].items():
    assert digest(root / 'source' / name) == expected, name
  for name, info in manifest.items():
    assert (root / name).stat().st_size == info['bytes'], name
    assert digest(root / name) == info['sha256'], name
  for job in state['jobs']:
    assert job['status'] == 'completed'
    assert digest(job['output']) == job['output_sha256']
    assert digest(root / f"seed_{job['seed']}" / 'variables.pkl') == job['checkpoint_sha256']
  started = datetime.datetime.fromisoformat(state['started_at'])
  trained = datetime.datetime.fromisoformat(batch['finished_at'])
  finished = datetime.datetime.fromisoformat(state['finished_at'])
  summary = dict(integrity=dict(source_files_verified=len(batch['source_sha256']),
      checkpoints_verified=len(manifest), checkpoint_bytes=sum(x['bytes'] for x in manifest.values()),
      evaluation_files_verified=len(state['jobs'])),
      training_hours=(trained - started).total_seconds() / 3600,
      evaluation_minutes=(finished - trained).total_seconds() / 60,
      started_at=state['started_at'], finished_at=state['finished_at'], runs=[])
  for entry in batch['runs']:
    run = root / f"seed_{entry['seed']}"
    with (run / 'variables.pkl').open('rb') as stream:
      checkpoint = pickle.load(stream)
    assert all(np.isfinite(v).all() for v in checkpoint['variables'])
    assert checkpoint['step'] == entry['final_step'] >= 100000
    assert checkpoint['warmup'] == dict(model=1000, actor=3000)
    demos = list((run / 'episodes' / 'demonstrations').glob('*.npz'))
    assert len(demos) == 64
    for demo in demos:
      assert digest(demo) == digest(run / 'episodes' / demo.name)
      with np.load(demo) as episode:
        assert episode['is_success'][-1] and episode['action'].shape[1] == 4
    evaluation = json.loads((run / 'final_evaluation.json').read_text())
    wm = json.loads((run / 'final_world_model.json').read_text())
    assert wm['checkpoint_sha256'] == digest(run / 'variables.pkl')
    assert wm['invariants']['parameters_unchanged'] and wm['invariants']['checkpoint_unchanged']
    assert wm['invariants']['optimizer_iterations_before'] == wm['invariants']['optimizer_iterations_after']
    for name, value in wm['frozen_source_sha256'].items():
      assert value == batch['source_sha256'][name]
    curves, losses = collections.defaultdict(list), []
    nonfinite = []
    for index, row in enumerate(read_jsonl(run / 'metrics.jsonl')):
      for key, value in row.items():
        if isinstance(value, (int, float)) and not math.isfinite(value):
          nonfinite.append([index, key])
      if 'model_loss' in row:
        losses.append(row)
      for phase in ('train', 'test'):
        if f'{phase}/success' in row and row['step'] > 5000:
          curves[(phase, min(3, int(row['step'] // 25000)))].append(row)
    assert not nonfinite
    records = evaluation.pop('records')
    compact = dict(seed=entry['seed'], attempts=entry['attempts'], exit_code=entry['exit_code'],
        final_step=checkpoint['step'], warmup=checkpoint['warmup'], pinned_demos=len(demos),
        milestone_steps=[int(x.stem.split('_')[1]) for x in sorted((run / 'checkpoints').glob('*.pkl'))],
        evaluation=evaluation, all_turn_episodes=sum(x['action_counts'][0] == 0 for x in records),
        pickup_without_delivery=sum(x['pickup'] and not x['success'] for x in records),
        metric_rows_finite=True, loss_rows=len(losses), curves={}, losses={}, world_model={},
        invariants=wm['invariants'])
    for (phase, quarter), rows in sorted(curves.items()):
      compact['curves'][f'{phase}_quarter_{quarter + 1}'] = dict(episodes=len(rows),
          **{key: avg(rows, f'{phase}/{key}') for key in ('success', 'pickup', 'out_of_bounds', 'truncated')})
    for key in ('model_loss', 'vector_loss', 'reward_loss', 'discount_loss', 'actor_bc_loss',
                'actor_loss', 'model_grad_norm', 'actor_grad_norm', 'critic_grad_norm'):
      values = [row[key] for row in losses if key in row]
      compact['losses'][key] = dict(first20=avg(losses[:20], key), last20=avg(losses[-20:], key),
                                  maximum=max(values))
    for dataset, info in wm['datasets'].items():
      predictions = {}
      for name, p in info['predictions'].items():
        predictions[name] = dict(position_mae=p['model']['position_mae_m'],
            baseline_position_mae=p['persistence']['position_mae_m'],
            heading_mae=p['model']['heading_mae_deg'], baseline_heading_mae=p['persistence']['heading_mae_deg'],
            reward_mae=p['all']['reward_mae'], baseline_reward_mae=p['all']['constant_reward_baseline_mae'],
            nonterminal_false_terminal=p['nonterminal']['predicted_terminal_fraction'],
            position_mae_early=p.get('time_1_20', {}).get('position_mae_m'),
            position_mae_late=p.get('time_21_100', {}).get('position_mae_m'))
      compact['world_model'][dataset] = predictions
    compact['action_probe'] = {k: v for k, v in wm['action_probes']['motion'].items()
                               if k in ('contexts', 'move_plus_minus_speed_difference_mean',
                                 'turn_plus_minus_heading_difference_deg_mean',
                                 'move_speed_direction_correct_fraction', 'turn_heading_direction_correct_fraction')}
    summary['runs'].append(compact)
  summary['aggregate'] = {key: statistics.mean(row['evaluation'][key] for row in summary['runs'])
                         for key in ('success_rate', 'pickup_rate', 'out_of_bounds_rate', 'timeout_rate',
                                     'parameter_saturation')}
  summary['aggregate']['success_seed_sample_sd'] = statistics.stdev(
      row['evaluation']['success_rate'] for row in summary['runs'])
  args.output.parent.mkdir(parents=True, exist_ok=True)
  atomic_write(args.output, lambda stream: stream.write(json.dumps(summary, indent=2, allow_nan=False).encode()))
  for name in ('stage_status.json', 'batch_status.json', 'checkpoint_manifest.json'):
    shutil.copy2(root / name, args.output.parent / name)
  for seed in (0, 1, 2):
    destination = args.output.parent / f'seed_{seed}'
    destination.mkdir(exist_ok=True)
    for name in ('final_evaluation.json', 'final_world_model.json', 'metrics.jsonl', 'run_metadata.jsonl'):
      shutil.copy2(root / f'seed_{seed}' / name, destination / name)
  print(json.dumps({k: v for k, v in summary.items() if k != 'runs'}, indent=2))
  for row in summary['runs']:
    print(json.dumps({k: v for k, v in row.items() if k not in ('world_model', 'evaluation', 'invariants', 'losses')}, indent=2))
    print(json.dumps(dict(seed=row['seed'], losses=row['losses'], replay=row['world_model']['replay']), indent=2))


if __name__ == '__main__':
  main()
