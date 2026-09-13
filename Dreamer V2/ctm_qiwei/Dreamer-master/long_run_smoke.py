"""Bounded GPU test of real interruption and frozen-batch resume, not full training."""
import argparse
import json
import os
import pathlib
import pickle
import signal
import threading
import time

import run_batch
from run_support import RunLock, atomic_write, read_jsonl


def main(root):
  root = root.resolve()
  if root.exists():
    raise ValueError('Use a new smoke directory; no existing run will be changed')
  root.mkdir(parents=True)
  with RunLock(root / '.batch.lock'):
    args = argparse.Namespace(seeds=[0], steps=480, eval_every=100, eval_episodes=1)
    status = run_batch.prepare_batch(root, args)
    command = status['runs'][0]['command']
    command[command.index('--prefill') + 1] = '100'
    command += ['--demo_episodes', '2', '--pretrain', '2', '--actor_pretrain', '2',
                '--checkpoint_every', '100', '--pretrain_checkpoint_every', '1']
    run_batch.write_json(root / 'batch_status.json', status)
    stop, injected = threading.Event(), threading.Event()
    checkpoint = root / 'seed_0' / 'variables.pkl'
    def interrupt_after_durable_checkpoint():
      deadline = time.monotonic() + 240
      while not stop.wait(.5):
        if checkpoint.exists():
          with checkpoint.open('rb') as stream:
            payload = pickle.load(stream)
          # Interrupt in Actor warmup, after a durable model warmup checkpoint.
          if payload['warmup']['model'] >= 2:
            injected.set()
            print(f"Injecting interrupt at step {payload['step']}, warmup={payload['warmup']}", flush=True)
            os.kill(os.getpid(), signal.SIGINT)
            return
        if time.monotonic() > deadline:
          print('Smoke deadline reached; stopping the owned run', flush=True)
          os.kill(os.getpid(), signal.SIGINT)
          return
    watcher = threading.Thread(target=interrupt_after_durable_checkpoint, daemon=True)
    watcher.start()
    try:
      run_batch.run_entries(root, status)
    except KeyboardInterrupt:
      if not injected.is_set():
        raise RuntimeError('Smoke interruption point was not reached within the budget')
    finally:
      stop.set()
      watcher.join(timeout=2)
    assert injected.is_set() and status['status'] == 'interrupted', status
    with checkpoint.open('rb') as stream:
      interrupted_checkpoint = pickle.load(stream)
    run_batch.run_entries(root, status)
    with checkpoint.open('rb') as stream:
      final_checkpoint = pickle.load(stream)
    metadata = list(read_jsonl(root / 'seed_0' / 'run_metadata.jsonl'))
    assert any(row['resumed_checkpoint'] and row['phase'] == 'running' for row in metadata)
    assert final_checkpoint['step'] >= 480
    assert final_checkpoint['warmup'] == dict(model=2, actor=2)
    assert status['runs'][0]['attempts'] == 2
    summary = dict(passed=True, model='default RSSM 400/32x32, batch 50x20, imagination 15',
        interrupted_step=interrupted_checkpoint['step'], interrupted_warmup=interrupted_checkpoint['warmup'],
        final_step=final_checkpoint['step'], final_warmup=final_checkpoint['warmup'],
        resumed=True, attempts=2)
    atomic_write(root / 'smoke_result.json', lambda stream: stream.write(json.dumps(summary, indent=2).encode()))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--outdir', type=pathlib.Path, required=True)
  main(parser.parse_args().outdir)
