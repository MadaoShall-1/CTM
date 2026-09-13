"""Exercise the actual frozen-runner lock boundary with tiny CPU-only fixtures."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest


PROJECT = Path(__file__).resolve().parents[1]


class PauseBoundaryTest(unittest.TestCase):
    def exercise(self, child_exit):
        with tempfile.TemporaryDirectory(prefix='ctm-pause-test-') as temporary:
            root = Path(temporary)
            source = root / 'source'
            source.mkdir()
            hashes = {}
            for name in ('run_batch.py', 'run_support.py', 'run_validation_stage.py'):
                shutil.copyfile(PROJECT / name, source / name)
                hashes[name] = hashlib.sha256((source / name).read_bytes()).hexdigest()
            for seed in range(3):
                (root / f'seed_{seed}').mkdir()
            (root / 'seed_2' / 'variables.pkl').write_bytes(b'unchanged seed 2 fixture')
            fixture = root / 'fixture.py'
            fixture.write_text('''import json, pathlib, pickle, shutil, sys, time
import numpy as np
root = pathlib.Path(sys.argv[1])
sys.path.insert(0, str(root / 'source'))
from run_support import RunLock
with RunLock(root / 'seed_1' / '.run.lock'):
    deadline = time.monotonic() + 20
    while not (root / 'pause_after_seed1.json').exists():
        if time.monotonic() > deadline:
            raise RuntimeError('guard did not arm')
        time.sleep(.02)
    if int(sys.argv[2]):
        sys.exit(int(sys.argv[2]))
    run = root / 'seed_1'
    with (run / 'variables.pkl').open('wb') as stream:
        pickle.dump(dict(step=300005, warmup=dict(model=1000, actor=3000),
                         variables=[np.array([1.0])]), stream)
    (run / 'checkpoints').mkdir()
    shutil.copyfile(run / 'variables.pkl', run / 'checkpoints' / 'step_000300005.pkl')
    (run / 'metrics.jsonl').write_text(json.dumps(dict(step=300005, **{
        'test/success': 1., 'test/pickup': 1., 'test/out_of_bounds': 0.,
        'test/return': 1., 'test/discounted_return': 1.})) + '\\n')
''')
            entries = [dict(seed=seed, status='completed' if seed == 0 else 'pending',
                            attempts=0, outdir=str(root / f'seed_{seed}'),
                            command=[sys.executable, '-B', str(fixture), str(root), str(child_exit)])
                       for seed in range(3)]
            (root / 'batch_status.json').write_text(json.dumps(dict(
                status='prepared', runner_contract='long_run_safe_v1', source_sha256=hashes, runs=entries)))
            (root / 'stage_status.json').write_text(json.dumps(dict(status='prepared', jobs=[])))
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
            parent = subprocess.Popen([sys.executable, '-B', str(source / 'run_validation_stage.py'),
                                       '--outdir', str(root), '--resume'], env=env,
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                deadline = time.monotonic() + 10
                while True:
                    batch = json.loads((root / 'batch_status.json').read_text())
                    if batch['runs'][1].get('pid'):
                        break
                    if time.monotonic() > deadline:
                        self.fail('fixture runner did not start')
                    time.sleep(.02)
                guard = subprocess.run([sys.executable, '-B', str(PROJECT / 'pause_after_seed1.py'),
                                        '--root', str(root), '--parent-pid', str(parent.pid),
                                        '--poll-seconds', '.02'], env=env,
                                       capture_output=True, text=True, timeout=25)
                parent.wait(timeout=5)
                request = json.loads((root / 'pause_after_seed1.json').read_text())
                batch = json.loads((root / 'batch_status.json').read_text())
                self.assertEqual(batch['runs'][2]['attempts'], 0)
                self.assertNotIn('pid', batch['runs'][2])
                self.assertEqual((root / 'seed_2' / 'variables.pkl').read_bytes(), b'unchanged seed 2 fixture')
                if child_exit == 0:
                    self.assertEqual(guard.returncode, 0, guard.stderr)
                    self.assertEqual(request['status'], 'paused')
                    self.assertTrue(request['seed1_checkpoint_finite'])
                    self.assertEqual(batch['runs'][1]['status'], 'completed')
                    self.assertEqual(batch['runs'][2]['status'], 'pending')
                    self.assertEqual(batch['status'], 'interrupted')
                    self.assertTrue((root / 'pause_after_seed1_runner_exit.json').exists())
                else:
                    self.assertNotEqual(guard.returncode, 0)
                    self.assertEqual(request['status'], 'needs_attention')
                    self.assertEqual(batch['status'], 'failed')
                    self.assertEqual(batch['runs'][1]['exit_code'], child_exit)
                    self.assertFalse((root / 'pause_after_seed1_runner_exit.json').exists())
            finally:
                if parent.poll() is None:
                    parent.terminate()
                    parent.wait(timeout=15)

    def test_pauses_only_after_success_and_blocks_seed2(self):
        self.exercise(0)

    def test_preserves_real_training_failure(self):
        self.exercise(3)


if __name__ == '__main__':
    unittest.main()
