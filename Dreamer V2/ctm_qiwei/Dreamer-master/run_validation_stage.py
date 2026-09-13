"""Fixed Stage 1: three 100k-step seeds, followed by independent evaluation.

No LLM calls. Stop on errors; --resume keeps the frozen protocol and source.
"""
import argparse
import datetime
import hashlib
import json
import os
import pathlib
import signal
import subprocess

import run_batch
from run_support import RunLock


def digest(path):
  checksum = hashlib.sha256()
  with pathlib.Path(path).open('rb') as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b''):
      checksum.update(block)
  return checksum.hexdigest()


def verify_source(root, batch):
  for name, expected in batch['source_sha256'].items():
    if digest(root / 'source' / name) != expected:
      raise ValueError(f'Frozen source changed: {name}')


def evaluation_jobs(root):
  jobs = []
  for seed in (0, 1, 2):
    run = root / f'seed_{seed}'
    for name, script, options in (
        ('evaluation', 'evaluate_checkpoint.py', ['--episodes', '100', '--seed-start', '60000']),
        ('world_model', 'diagnose_world_model.py', ['--replay-limit', '32',
          '--heldout-episodes', '8', '--batch-size', '4', '--mc-samples', '16'])):
      output = run / f'final_{name}.json'
      jobs.append(dict(seed=seed, name=name, status='pending', output=str(output),
          command=['bash', 'run_wsl.sh', '-B', script, '--run', str(run),
                   '--output', str(output), *options]))
  return jobs


def evaluate(root, state):
  for job in state['jobs']:
    run = root / f"seed_{job['seed']}"
    checkpoint_hash = digest(run / 'variables.pkl')
    if job['status'] == 'completed':
      if (digest(job['output']) != job['output_sha256'] or
          checkpoint_hash != job['checkpoint_sha256']):
        raise ValueError('Completed evaluation or checkpoint changed')
      continue
    child = None
    try:
      with (run / f"final_{job['name']}.log").open('a') as console:
        child = subprocess.Popen(job['command'], cwd=root / 'source',
            env=dict(os.environ, CTM_REQUIRE_GPU='1', PYTHONDONTWRITEBYTECODE='1'),
            stdout=console, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            start_new_session=True)
        job.update(status='running', pid=child.pid)
        run_batch.write_json(root / 'stage_status.json', state)
        if child.wait() != 0:
          raise RuntimeError(f"Evaluation failed: seed {job['seed']} {job['name']}")
      json.loads(pathlib.Path(job['output']).read_text())
      if digest(run / 'variables.pkl') != checkpoint_hash:
        raise ValueError('Evaluation modified checkpoint')
      job.update(status='completed', checkpoint_sha256=checkpoint_hash,
                 output_sha256=digest(job['output']))
      run_batch.write_json(root / 'stage_status.json', state)
    except BaseException:
      run_batch.stop_child(child)
      job['status'] = 'failed'
      raise


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--outdir', type=pathlib.Path, required=True)
  parser.add_argument('--resume', action='store_true')
  args = parser.parse_args()
  root = args.outdir.resolve()
  root.mkdir(parents=True, exist_ok=True)
  def interrupted(*_):
    raise KeyboardInterrupt('Stage interrupted')
  signal.signal(signal.SIGTERM, interrupted)
  with RunLock(root / '.stage.lock'), RunLock(root / '.batch.lock'):
    status_path = root / 'stage_status.json'
    if args.resume:
      state = json.loads(status_path.read_text())
      batch = json.loads((root / 'batch_status.json').read_text())
      verify_source(root, batch)
      if state['status'] == 'completed':
        print('Stage already completed; nothing to do', flush=True)
        return
    else:
      if status_path.exists():
        raise ValueError('Stage exists; use --resume')
      batch = run_batch.prepare_batch(root, argparse.Namespace(
          seeds=[0, 1, 2], steps=100000, eval_every=5000, eval_episodes=5))
      state = dict(status='training', started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
          protocol='autopickup_stage1_3x100k_v1', jobs=evaluation_jobs(root),
          notes='100k budget includes prefill and controller demonstration transitions; '
                'automatic pickup plus exact-vector actor and demo BC, not unmodified DreamerV2.')
    state.update(pid=os.getpid(), status='training')
    state.pop('error', None)
    run_batch.write_json(status_path, state)
    try:
      run_batch.run_entries(root, batch)
      state['status'] = 'evaluating'
      run_batch.write_json(status_path, state)
      verify_source(root, batch)
      evaluate(root, state)
      manifest = {}
      for seed in (0, 1, 2):
        run = root / f'seed_{seed}'
        files = [run / 'variables.pkl', run / 'variables.prev.pkl',
                 *sorted((run / 'checkpoints').glob('*.pkl'))]
        for path in files:
          manifest[str(path.relative_to(root))] = dict(bytes=path.stat().st_size, sha256=digest(path))
      run_batch.write_json(root / 'checkpoint_manifest.json', manifest)
      state.update(status='completed', finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
      run_batch.write_json(status_path, state)
      print('Stage 1 training and evaluation completed; ready for analysis.', flush=True)
    except BaseException as error:
      state.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=str(error))
      run_batch.write_json(status_path, state)
      raise


if __name__ == '__main__':
  main()
