import json
import pathlib
import pickle
import tempfile
import unittest
from unittest import mock

import prepare_stage2 as preparation


class ContinuationTest(unittest.TestCase):
  def setUp(self):
    temporary = tempfile.TemporaryDirectory()
    self.addCleanup(temporary.cleanup)
    self.parent = pathlib.Path(temporary.name) / 'stage1'
    self.target = self.parent.parent / 'stage2'
    (self.parent / 'source').mkdir(parents=True)
    (self.parent / 'source' / 'dreamer.py').write_text('# frozen')
    batch = dict(status='completed', runner_contract='long_run_safe_v1', source_sha256={
        'dreamer.py': preparation.digest(self.parent / 'source' / 'dreamer.py')}, runs=[])
    manifest = {}
    for seed in range(3):
      run = self.parent / f'seed_{seed}'
      (run / 'episodes' / 'demonstrations').mkdir(parents=True)
      (run / 'episodes' / 'online-101.npz').write_bytes(b'immutable replay')
      for demo in range(64):
        (run / 'episodes' / 'demonstrations' / f'demo-{demo}.npz').write_bytes(b'demo')
      payload = dict(step=100, warmup=dict(model=1000, actor=3000), variables=[1.0])
      (run / 'variables.pkl').write_bytes(pickle.dumps(payload))
      (run / 'metrics.jsonl').write_text('{}\n')
      (run / 'run_metadata.jsonl').write_text('{}\n')
      manifest[f'seed_{seed}/variables.pkl'] = dict(sha256=preparation.digest(run / 'variables.pkl'))
      batch['runs'].append(dict(seed=seed, final_step=100, status='completed', attempts=1,
          command=['bash', 'run_wsl.sh', '-u', 'dreamer.py', '--seed', str(seed),
                   '--steps', '100', '--logdir', str(run)]))
    preparation.write_json(self.parent / 'batch_status.json', batch)
    preparation.write_json(self.parent / 'checkpoint_manifest.json', manifest)
    preparation.write_json(self.parent / 'stage_status.json', dict(status='completed'))

  def test_continuation_preserves_parent_and_only_changes_budget_and_output(self):
    before = preparation.digest(self.parent / 'batch_status.json')
    with mock.patch('prepare_stage2.require_free_space'):
      preparation.prepare(self.parent, self.target, 300)
    self.assertEqual(before, preparation.digest(self.parent / 'batch_status.json'))
    state = json.loads((self.target / 'stage_status.json').read_text())
    batch = json.loads((self.target / 'batch_status.json').read_text())
    self.assertEqual(len(state['jobs']), 9)
    self.assertEqual([j['name'] for j in state['jobs'][:3]], ['evaluation', 'holdout', 'world_model'])
    for entry in batch['runs']:
      self.assertEqual(entry['status'], 'pending')
      self.assertEqual(entry['command'][entry['command'].index('--steps') + 1], '300')
      self.assertTrue((self.target / f"seed_{entry['seed']}" / 'checkpoints' / 'step_000000100.pkl').exists())

  def test_completed_budget_not_replayed(self):
    with self.assertRaisesRegex(ValueError, 'budget'):
      preparation.prepare(self.parent, self.target, 100)
    self.assertFalse(self.target.exists())

  def test_corrupt_parent_rejected_before_copy(self):
    (self.parent / 'seed_0' / 'variables.pkl').write_bytes(b'broken')
    with self.assertRaisesRegex(ValueError, 'hash'):
      preparation.prepare(self.parent, self.target)
    self.assertFalse(self.target.exists())

  def test_existing_or_nested_destination_rejected(self):
    with self.assertRaisesRegex(ValueError, 'separate'):
      preparation.prepare(self.parent, self.parent / 'stage2')
    self.target.mkdir()
    with self.assertRaises(FileExistsError):
      preparation.prepare(self.parent, self.target)


if __name__ == '__main__':
  unittest.main()
