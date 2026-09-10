"""Run independent UAV seeds sequentially on one GPU using frozen source."""
import argparse
import datetime
import hashlib
import json
import os
import pathlib
import shutil
import subprocess


def write_json(path, value):
  temporary = path.with_suffix('.tmp')
  temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
  temporary.replace(path)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--outdir', type=pathlib.Path, required=True)
  parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
  parser.add_argument('--steps', type=int, default=100000)
  parser.add_argument('--eval-every', type=int, default=5000)
  parser.add_argument('--eval-episodes', type=int, default=5)
  args = parser.parse_args()
  if len(set(args.seeds)) != len(args.seeds) or args.steps <= 5000:
    parser.error('Use distinct seeds and a budget greater than the 5000-step prefill')
  if args.eval_every <= 0 or args.eval_episodes <= 0:
    parser.error('Evaluation interval and episode count must be positive')
  root = args.outdir.resolve()
  root.mkdir(parents=True, exist_ok=True)
  if (root / 'batch_status.json').exists():
    raise ValueError('Batch already exists; choose a fresh outdir')
  source = root / 'source'
  source.mkdir()
  repository = pathlib.Path(__file__).resolve().parent
  names = ['dreamer.py', 'envs.py', 'models.py', 'tools.py', 'wrappers.py', 'uav_actions.py',
           'run_wsl.sh', 'gpu_check.py', 'requirements.txt']
  hashes = {}
  for name in names:
    shutil.copy2(repository / name, source / name)
    hashes[name] = hashlib.sha256((source / name).read_bytes()).hexdigest()
  status = dict(status='running', pid=os.getpid(),
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
  for entry in status['runs']:
    run = pathlib.Path(entry['outdir'])
    run.mkdir()
    entry['status'] = 'running'
    environment = dict(os.environ, CTM_REQUIRE_GPU='1', PYTHONDONTWRITEBYTECODE='1')
    with (run / 'console.log').open('w') as console:
      child = subprocess.Popen(entry['command'], cwd=source, env=environment,
                               stdout=console, stderr=subprocess.STDOUT,
                               stdin=subprocess.DEVNULL, start_new_session=True)
      entry['pid'] = child.pid
      write_json(root / 'batch_status.json', status)
      print(f"Seed {entry['seed']} started: pid={child.pid}, {run}", flush=True)
      code = child.wait()
    entry['exit_code'] = code
    entry['status'] = 'completed' if code == 0 else 'failed'
    if code != 0:
      status['status'] = 'failed'
      write_json(root / 'batch_status.json', status)
      raise SystemExit(code)
    records = [json.loads(line) for line in (run / 'metrics.jsonl').read_text().splitlines()]
    evaluations = [record for record in records if 'test/success' in record]
    last_step = max(record['step'] for record in evaluations)
    final = [record for record in evaluations if record['step'] == last_step]
    entry['final_evaluation'] = {key: sum(row[key] for row in final) / len(final)
        for key in ['test/success', 'test/pickup', 'test/out_of_bounds',
                    'test/return', 'test/discounted_return']}
    entry['final_step'] = last_step
    write_json(root / 'batch_status.json', status)
  status['status'] = 'completed'
  status['finished_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
  write_json(root / 'batch_status.json', status)
  print('Batch completed', flush=True)


if __name__ == '__main__':
  main()
