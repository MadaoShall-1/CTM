"""Run independent UAV seeds sequentially on one GPU using frozen source."""
import argparse
import datetime
import hashlib
import json
import os
import pathlib
import shutil
import signal
import subprocess
import sys
import math

from run_support import RunLock, atomic_write, read_jsonl


def write_json(path, value):
  data = json.dumps(value, indent=2, allow_nan=False).encode('utf-8')
  atomic_write(path, lambda stream: stream.write(data))


def final_evaluation(path):
  last_step, final = -1, []
  for record in read_jsonl(path):
    if 'test/success' not in record:
      continue
    if record['step'] > last_step:
      last_step, final = record['step'], []
    if record['step'] == last_step:
      final.append(record)
  if not final:
    raise ValueError('Run finished without any complete evaluation records')
  summary = {key: sum(row[key] for row in final) / len(final) for key in (
      'test/success', 'test/pickup', 'test/out_of_bounds', 'test/return', 'test/discounted_return')}
  if not all(math.isfinite(value) for value in summary.values()):
    raise ValueError('Run produced non-finite evaluation metrics')
  return last_step, summary


def stop_child(child):
  if child is None or child.poll() is not None:
    return
  try:
    if os.name == 'posix':
      os.killpg(child.pid, signal.SIGTERM)
    else:
      child.terminate()
  except ProcessLookupError:
    child.wait(timeout=10)
    return
  try:
    child.wait(timeout=10)
  except subprocess.TimeoutExpired:
    if os.name == 'posix':
      os.killpg(child.pid, signal.SIGKILL)
    else:
      child.kill()
    child.wait(timeout=10)


def run_entries(root, status):
  """Resume only unfinished seeds and keep failures/cancellation visible."""
  if status.get('runner_contract') != 'long_run_safe_v1':
    raise ValueError('Legacy batch requires its old runner; create a fresh hardened batch')
  source = root / 'source'
  for name, expected in status['source_sha256'].items():
    if hashlib.sha256((source / name).read_bytes()).hexdigest() != expected:
      raise ValueError(f'Frozen source changed: {name}; refusing mixed-code resume')
  status.update(status='running', pid=os.getpid())
  write_json(root / 'batch_status.json', status)
  for entry in status['runs']:
    if entry['status'] == 'completed':
      continue
    run = pathlib.Path(entry['outdir'])
    run.mkdir(exist_ok=True)
    child = None
    try:
      # Do not compete with an orphaned but still healthy training process.
      with RunLock(run / '.run.lock'):
        pass
      entry['status'] = 'running'
      entry['attempts'] = entry.get('attempts', 0) + 1
      environment = dict(os.environ, CTM_REQUIRE_GPU='1', PYTHONDONTWRITEBYTECODE='1')
      with (run / 'console.log').open('a') as console:
        child = subprocess.Popen(entry['command'], cwd=source, env=environment,
                                 stdout=console, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, start_new_session=True)
        entry['pid'] = child.pid
        write_json(root / 'batch_status.json', status)
        print(f"Seed {entry['seed']} started: pid={child.pid}, {run}", flush=True)
        code = child.wait()
      entry['exit_code'] = code
      if code != 0:
        raise RuntimeError(f"Seed {entry['seed']} failed with exit code {code}")
      entry['final_step'], entry['final_evaluation'] = final_evaluation(run / 'metrics.jsonl')
      entry['status'] = 'completed'
      entry.pop('error', None)
      write_json(root / 'batch_status.json', status)
    except BaseException as error:
      stop_child(child)
      entry['status'] = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
      entry['error'] = str(error)
      status['status'] = entry['status']
      write_json(root / 'batch_status.json', status)
      raise
  status['status'] = 'completed'
  status['finished_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
  write_json(root / 'batch_status.json', status)
  print('Batch completed', flush=True)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--outdir', type=pathlib.Path, required=True)
  parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
  parser.add_argument('--steps', type=int, default=100000)
  parser.add_argument('--eval-every', type=int, default=5000)
  parser.add_argument('--eval-episodes', type=int, default=5)
  parser.add_argument('--resume', action='store_true', help='Resume the frozen batch with its original settings')
  args = parser.parse_args()
  if len(set(args.seeds)) != len(args.seeds) or args.steps <= 5000:
    parser.error('Use distinct seeds and a budget greater than the 5000-step prefill')
  if args.eval_every <= 0 or args.eval_episodes <= 0:
    parser.error('Evaluation interval and episode count must be positive')
  root = args.outdir.resolve()
  root.mkdir(parents=True, exist_ok=True)
  flags = {item.split('=', 1)[0] for item in sys.argv[1:]}
  if args.resume and flags.intersection({'--seeds', '--steps', '--eval-every', '--eval-episodes'}):
    parser.error('--resume uses the original saved commands; do not override run settings')
  previous_handler = signal.getsignal(signal.SIGTERM)
  def interrupted(*_):
    raise KeyboardInterrupt('Batch interrupted')
  signal.signal(signal.SIGTERM, interrupted)
  try:
    with RunLock(root / '.batch.lock'):
      status = (json.loads((root / 'batch_status.json').read_text()) if args.resume
                else prepare_batch(root, args))
      run_entries(root, status)
  finally:
    signal.signal(signal.SIGTERM, previous_handler)


def prepare_batch(root, args):
  if (root / 'batch_status.json').exists():
    raise ValueError('Batch already exists; choose a fresh outdir')
  source = root / 'source'
  source.mkdir()
  repository = pathlib.Path(__file__).resolve().parent
  names = ['dreamer.py', 'envs.py', 'models.py', 'tools.py', 'wrappers.py', 'uav_actions.py',
           'controller_baseline.py', 'run_support.py', 'run_wsl.sh', 'gpu_check.py', 'requirements.txt',
           'run_batch.py', 'run_validation_stage.py', 'evaluate_checkpoint.py', 'diagnose_world_model.py']
  hashes = {}
  for name in names:
    shutil.copy2(repository / name, source / name)
    hashes[name] = hashlib.sha256((source / name).read_bytes()).hexdigest()
  status = dict(status='running', pid=os.getpid(), runner_contract='long_run_safe_v1',
                started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                source_sha256=hashes, runs=[])
  for seed in args.seeds:
    run = root / f'seed_{seed}'
    command = ['bash', 'run_wsl.sh', '-u', 'dreamer.py', '--task', 'uav_relay',
               '--reward_mode', 'goal_safe_v1', '--shaping_scale', '1.0',
               '--discount', '.99', '--seed', str(seed), '--steps', str(args.steps),
               '--prefill', '5000', '--eval_every', str(args.eval_every),
               '--eval_episodes', str(args.eval_episodes),
               '--log_every', '500', '--logdir', str(run)]
    status['runs'].append(dict(seed=seed, status='pending', outdir=str(run), command=command))
  write_json(root / 'batch_status.json', status)
  return status


if __name__ == '__main__':
  main()
