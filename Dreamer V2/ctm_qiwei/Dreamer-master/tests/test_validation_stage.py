import json
import pathlib
import tempfile
import unittest
from unittest import mock

import run_support
import run_validation_stage as stage


class ValidationStageTest(unittest.TestCase):
  def setUp(self):
    directory = tempfile.TemporaryDirectory()
    self.addCleanup(directory.cleanup)
    self.root = pathlib.Path(directory.name)

  def test_snapshot_is_independent_and_never_overwritten(self):
    checkpoint = self.root / 'variables.pkl'
    checkpoint.write_bytes(b'complete checkpoint')
    saved = run_support.snapshot_checkpoint(checkpoint, 10007)
    checkpoint.write_bytes(b'later checkpoint')
    self.assertEqual(saved.read_bytes(), b'complete checkpoint')
    with self.assertRaises(FileExistsError):
      run_support.snapshot_checkpoint(checkpoint, 10007)

  def test_snapshot_disk_guard_before_write(self):
    checkpoint = self.root / 'variables.pkl'
    checkpoint.write_bytes(b'complete')
    with mock.patch('run_support.require_free_space', side_effect=RuntimeError('disk')):
      with self.assertRaisesRegex(RuntimeError, 'disk'):
        run_support.snapshot_checkpoint(checkpoint, 10)
    self.assertFalse((self.root / 'checkpoints').exists())

  def test_fixed_disjoint_holdout_and_bounded_diagnostics(self):
    jobs = stage.evaluation_jobs(self.root)
    self.assertEqual(len(jobs), 6)
    for job in jobs:
      command = job['command']
      if job['name'] == 'evaluation':
        self.assertEqual(command[command.index('--episodes') + 1], '100')
        self.assertEqual(command[command.index('--seed-start') + 1], '60000')
      else:
        self.assertEqual(command[command.index('--replay-limit') + 1], '32')

  def test_evaluation_failure_stops_queue(self):
    (self.root / 'seed_0').mkdir()
    (self.root / 'seed_0' / 'variables.pkl').write_bytes(b'checkpoint')
    state = dict(jobs=stage.evaluation_jobs(self.root))
    child = mock.Mock(pid=1000, wait=mock.Mock(return_value=1), poll=mock.Mock(return_value=1))
    with mock.patch('run_validation_stage.subprocess.Popen', return_value=child) as start:
      with self.assertRaisesRegex(RuntimeError, 'Evaluation failed'):
        stage.evaluate(self.root, state)
      self.assertEqual(start.call_count, 1)
    self.assertEqual(state['jobs'][0]['status'], 'failed')

  def test_resume_verifies_completed_evaluation_without_rerun(self):
    run = self.root / 'seed_0'
    run.mkdir()
    (run / 'variables.pkl').write_bytes(b'checkpoint')
    output = run / 'result.json'
    output.write_text(json.dumps({'success_rate': .5}))
    state = dict(jobs=[dict(seed=0, status='completed', output=str(output),
        output_sha256=stage.digest(output), checkpoint_sha256=stage.digest(run / 'variables.pkl'))])
    with mock.patch('run_validation_stage.subprocess.Popen') as start:
      stage.evaluate(self.root, state)
      start.assert_not_called()
      output.write_text('{}')
      with self.assertRaisesRegex(ValueError, 'changed'):
        stage.evaluate(self.root, state)


if __name__ == '__main__':
  unittest.main()
