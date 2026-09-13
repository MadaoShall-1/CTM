"""Let seed 1 finish; deny seed 2's pre-launch lock without editing frozen code.

The old runner reports lock contention as a failure. After it exits, preserve
that original state and reclassify ONLY this expected boundary as user-paused.
This helper never signals a trainer or changes checkpoints/training commands.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def read_json(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def process(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        command = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
        return dict(state=fields[0], ppid=int(fields[1]), start=fields[19],
                    command=[part.decode() for part in command if part])
    except FileNotFoundError:
        return None


def same_live_process(pid, identity):
    current = process(pid)
    return bool(current and current['start'] == identity['start']
                and current['state'] not in ('Z', 'X'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--parent-pid', required=True, type=int)
    parser.add_argument('--poll-seconds', type=float, default=5)
    args = parser.parse_args()
    if not 0.02 <= args.poll_seconds <= 30:
        raise ValueError('Use a bounded poll interval')
    root = args.root.resolve(strict=True)
    batch = read_json(root / 'batch_status.json')
    stage = read_json(root / 'stage_status.json')
    identity = process(args.parent_pid)
    expected_runner = str(root / 'source' / 'run_validation_stage.py')
    if (not identity or expected_runner not in identity['command']
            or str(root) not in identity['command']
            or batch.get('pid') != args.parent_pid or stage.get('pid') != args.parent_pid):
        raise ValueError('Parent process identity does not match this stage')
    for name, expected in batch['source_sha256'].items():
        if digest(root / 'source' / name) != expected:
            raise ValueError(f'Frozen source mismatch: {name}')
    sys.path.insert(0, str(root / 'source'))
    from run_support import RunLock
    from run_batch import write_json

    runs = {entry['seed']: entry for entry in batch['runs']}
    child = process(runs[1].get('pid', -1))
    if (runs[0]['status'] != 'completed' or runs[1]['status'] != 'running'
            or not child or child['ppid'] != args.parent_pid
            or runs[2]['status'] != 'pending' or runs[2].get('attempts', 0) != 0
            or runs[2].get('pid') is not None):
        raise ValueError('Expected seed 1 running and seed 2 never started')
    state_path = root / 'pause_after_seed1.json'
    if state_path.exists():
        raise FileExistsError('Pause request already exists; inspect it before retrying')
    with RunLock(root / '.pause_after_seed1.lock'), RunLock(root / 'seed_2' / '.run.lock'):
        # Parent cannot pass seed 2's lock while this context is alive.
        checkpoint2_hash = digest(root / 'seed_2' / 'variables.pkl')
        request = dict(status='armed', requested_at=now(), guard_pid=os.getpid(),
                       parent_pid=args.parent_pid, parent_start=identity['start'],
                       after_seed=1, blocked_seed=2,
                       seed2_checkpoint_sha256=checkpoint2_hash,
                       script_sha256=digest(Path(__file__).resolve()),
                       reason='User requested pause after seed 1 completes; do not start seed 2.')
        write_json(state_path, request)
        print(json.dumps(request), flush=True)
        try:
            while same_live_process(args.parent_pid, identity):
                time.sleep(args.poll_seconds)
            # The exited runner can no longer overwrite these state files.
            with RunLock(root / '.stage.lock'), RunLock(root / '.batch.lock'):
                batch = read_json(root / 'batch_status.json')
                stage = read_json(root / 'stage_status.json')
                runs = {entry['seed']: entry for entry in batch['runs']}
                expected_error = f'Another process is already using {root / "seed_2"}'
                if (batch.get('pid') != args.parent_pid or stage.get('pid') != args.parent_pid
                        or batch['status'] != 'failed' or stage['status'] != 'failed'
                        or stage.get('error') != expected_error
                        or runs[1]['status'] != 'completed' or runs[1].get('exit_code') != 0
                        or runs[1].get('final_step', 0) < 300000
                        or runs[2]['status'] != 'failed' or runs[2].get('error') != expected_error
                        or runs[2].get('attempts', 0) != 0 or runs[2].get('pid') is not None
                        or any(job['status'] != 'pending' for job in stage['jobs'])):
                    raise RuntimeError('Runner stopped for an unexpected reason; original state preserved')
                if digest(root / 'seed_2' / 'variables.pkl') != checkpoint2_hash:
                    raise RuntimeError('Seed 2 checkpoint changed unexpectedly')
                import pickle
                import numpy as np
                checkpoint = root / 'seed_1' / 'variables.pkl'
                with checkpoint.open('rb') as stream:
                    payload = pickle.load(stream)
                if (payload['step'] != runs[1]['final_step']
                        or payload['warmup'] != dict(model=1000, actor=3000)
                        or not all(np.isfinite(value).all() for value in payload['variables'])):
                    raise RuntimeError('Final seed 1 checkpoint verification failed')
                checkpoint_hash = digest(checkpoint)
                snapshot = root / 'seed_1' / 'checkpoints' / f"step_{payload['step']:09d}.pkl"
                if digest(snapshot) != checkpoint_hash:
                    raise RuntimeError('Final retained snapshot does not match checkpoint')
                for name, expected in batch['source_sha256'].items():
                    if digest(root / 'source' / name) != expected:
                        raise RuntimeError('Frozen source changed during pause')
                # Keep the old runner's exact lock-denial outcome for auditability.
                write_json(root / 'pause_after_seed1_runner_exit.json', dict(stage=stage, batch=batch))
                stamp = now()
                runs[2]['status'] = 'pending'
                runs[2].pop('error', None)
                for state in (batch, stage):
                    state.update(status='interrupted', pause_reason=request['reason'], paused_at=stamp,
                                 pause_after_seed=1)
                    state.pop('error', None)
                write_json(root / 'batch_status.json', batch)
                write_json(root / 'stage_status.json', stage)
                request.update(status='paused', paused_at=stamp,
                               seed1_final_step=payload['step'], seed1_checkpoint_sha256=checkpoint_hash,
                               seed1_checkpoint_finite=True, seed2_not_started=True)
                write_json(state_path, request)
                print(json.dumps(request), flush=True)
        except BaseException as error:
            request.update(status='needs_attention', error=str(error), observed_at=now())
            write_json(state_path, request)
            raise


if __name__ == '__main__':
    main()
