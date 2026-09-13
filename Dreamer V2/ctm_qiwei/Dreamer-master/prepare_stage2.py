"""Prepare isolated continuation from a completed frozen Stage 1 (no training)."""
import argparse
import copy
import datetime
import hashlib
import json
import pathlib
import pickle
import shutil

import numpy as np

from run_support import RunLock, atomic_write, require_free_space


def digest(path):
  checksum = hashlib.sha256()
  with pathlib.Path(path).open('rb') as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b''):
      checksum.update(block)
  return checksum.hexdigest()


def write_json(path, value):
  atomic_write(path, lambda stream: stream.write(json.dumps(value, indent=2, allow_nan=False).encode()))


def prepare(previous, root, steps=300000):
  previous, root = pathlib.Path(previous).resolve(), pathlib.Path(root).resolve()
  if root == previous or root.is_relative_to(previous) or previous.is_relative_to(root):
    raise ValueError('Continuation must be in a separate directory')
  if root.exists():
    raise FileExistsError('Destination exists; refusing to overwrite or duplicate a stage')
  with RunLock(previous / '.stage.lock'), RunLock(previous / '.batch.lock'):
    old_state = json.loads((previous / 'stage_status.json').read_text())
    old_batch = json.loads((previous / 'batch_status.json').read_text())
    manifest = json.loads((previous / 'checkpoint_manifest.json').read_text())
    if old_state['status'] != 'completed' or old_batch['status'] != 'completed':
      raise ValueError('Previous stage is not completed')
    if [entry['seed'] for entry in old_batch['runs']] != [0, 1, 2]:
      raise ValueError('Expected the original three seeds')
    for name, checksum in old_batch['source_sha256'].items():
      if digest(previous / 'source' / name) != checksum:
        raise ValueError(f'Frozen source changed: {name}')
    parents = []
    for entry in old_batch['runs']:
      run = previous / f"seed_{entry['seed']}"
      with RunLock(run / '.run.lock'):
        checkpoint = run / 'variables.pkl'
        checksum = digest(checkpoint)
        if checksum != manifest[str(checkpoint.relative_to(previous))]['sha256']:
          raise ValueError('Parent checkpoint hash mismatch')
        with checkpoint.open('rb') as stream:
          payload = pickle.load(stream)
        if steps <= payload['step'] or payload['step'] != entry['final_step']:
          raise ValueError('Continuation budget must exceed all parent checkpoints')
        if payload['warmup'] != dict(model=1000, actor=3000):
          raise ValueError('Parent warmup is incomplete')
        if not all(np.isfinite(v).all() for v in payload['variables']):
          raise ValueError('Parent checkpoint is not finite')
        episode_paths = list((run / 'episodes').glob('*.npz'))
        if sum(int(p.stem.rsplit('-', 1)[1]) - 1 for p in episode_paths) != payload['step']:
          raise ValueError('Replay step count differs from checkpoint')
        if len(list((run / 'episodes' / 'demonstrations').glob('*.npz'))) != 64:
          raise ValueError('Incomplete permanent demonstrations')
        parents.append(dict(seed=entry['seed'], step=payload['step'], checkpoint_sha256=checksum))
    # Stage 2 is bounded: about 5.5 GiB of retained checkpoints plus small replay.
    require_free_space(previous, 12 * 2**30)
    root.mkdir(parents=True)
    shutil.copytree(previous / 'source', root / 'source')
    batch = copy.deepcopy(old_batch)
    batch.update(status='prepared', started_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    for key in ('finished_at', 'pid', 'error'):
      batch.pop(key, None)
    jobs = []
    for entry, parent in zip(batch['runs'], parents):
      seed = entry['seed']
      old_run, run = previous / f'seed_{seed}', root / f'seed_{seed}'
      with RunLock(old_run / '.run.lock'):
        run.mkdir()
        shutil.copytree(old_run / 'episodes', run / 'episodes')
        for name in ('variables.pkl', 'metrics.jsonl', 'run_metadata.jsonl'):
          shutil.copy2(old_run / name, run / name)
        if digest(run / 'variables.pkl') != parent['checkpoint_sha256']:
          raise ValueError('Copied checkpoint hash mismatch')
        for path in (old_run / 'episodes').rglob('*.npz'):
          if digest(path) != digest(run / path.relative_to(old_run)):
            raise ValueError('Copied replay hash mismatch')
        (run / 'checkpoints').mkdir()
        shutil.copy2(run / 'variables.pkl', run / 'checkpoints' / f"step_{parent['step']:09d}.pkl")
      command = list(entry['command'])
      command[command.index('--steps') + 1] = str(steps)
      command[command.index('--logdir') + 1] = str(run)
      entry.clear()
      entry.update(seed=seed, status='pending', outdir=str(run), command=command,
                   parent_checkpoint=parent, attempts=0)
      for name, script, options in (
          ('evaluation', 'evaluate_checkpoint.py', ['--episodes', '100', '--seed-start', '60000']),
          ('holdout', 'evaluate_checkpoint.py', ['--episodes', '100', '--seed-start', '70000']),
          ('world_model', 'diagnose_world_model.py', ['--replay-limit', '32', '--heldout-episodes', '8',
                                                   '--batch-size', '4', '--mc-samples', '16'])):
        output = run / f'final_{name}.json'
        jobs.append(dict(seed=seed, name=name, status='pending', output=str(output),
            command=['bash', 'run_wsl.sh', '-B', script, '--run', str(run), '--output', str(output), *options]))
    provenance = dict(previous_stage=str(previous), target_steps=steps, parent_checkpoints=parents,
        preparer_sha256=digest(__file__),
        changed_training_arguments=['steps', 'logdir'],
        note='Same frozen training source and hyperparameters; copied optimizer, warmup and replay. '
             'RNG/environment states restart: continuation is not bitwise identical to an uninterrupted run.')
    shutil.copy2(__file__, root / 'prepare_stage2.py')
    write_json(root / 'continuation_provenance.json', provenance)
    write_json(root / 'batch_status.json', batch)
    write_json(root / 'stage_status.json', dict(status='prepared',
        started_at=batch['started_at'], protocol=f'autopickup_stage2_3x{steps}_v1', jobs=jobs,
        previous_stage=str(previous), target_steps=steps,
        notes='Same code/parameters. 60000-60099 is reused paired comparison; 70000-70099 is new holdout. '
              'No automatic third stage. Original frozen runner may print Stage 1; this state identifies Stage 2.'))
    return provenance


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--previous', type=pathlib.Path, required=True)
  parser.add_argument('--outdir', type=pathlib.Path, required=True)
  parser.add_argument('--steps', type=int, default=300000)
  args = parser.parse_args()
  print(json.dumps(prepare(args.previous, args.outdir, args.steps), indent=2))
