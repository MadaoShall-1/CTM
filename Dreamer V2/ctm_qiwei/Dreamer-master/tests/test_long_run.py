"""Capacity, persistence, numerical and interruption regressions."""
import argparse
import itertools
import json
import pathlib
import pickle
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import tensorflow as tf

import dreamer
import run_batch
import run_support
import tools


def episode(marker=0, demonstration=False, length=11):
  return dict(vector=np.full((length, 13), marker, np.float32),
      action=np.tile([1, 0, .1, 0], (length, 1)).astype(np.float32),
      reward=np.zeros(length, np.float32), discount=np.r_[np.ones(length - 1), 0].astype(np.float32),
      demonstration=np.r_[0, np.full(length - 1, demonstration)].astype(np.float32),
      is_success=np.r_[np.zeros(length - 1), demonstration].astype(np.float32),
      phase=np.r_[np.zeros(length // 2), np.full(length - length // 2, demonstration)].astype(np.float32))


class ReplayLongRunTest(unittest.TestCase):
  def setUp(self):
    temporary = tempfile.TemporaryDirectory()
    self.addCleanup(temporary.cleanup)
    self.root = pathlib.Path(temporary.name)

  def write(self, name, marker=0, **kwargs):
    ep = episode(marker, **kwargs)
    path = self.root / f'{name}-11.npz'
    np.savez_compressed(path, **ep)
    return path

  def test_images_not_decompressed_or_cached(self):
    ep = episode()
    ep['image'] = np.ones((11, 64, 64, 3), np.uint8)
    tools.save_episodes(self.root, [ep])
    with np.load(next(self.root.glob('*.npz'))) as archive:
      archive_type = type(archive)
    original = archive_type.__getitem__
    def selective(archive, key):
      self.assertNotEqual(key, 'image')
      return original(archive, key)
    statistics = {}
    with mock.patch.object(archive_type, '__getitem__', selective):
      data = next(tools.load_episodes(self.root, 1, length=5,
          keys=('vector', 'action', 'reward', 'discount', 'demonstration'), statistics=statistics))
    self.assertNotIn('image', data)
    self.assertLess(statistics['cached_bytes'], 2000)

  def test_pinned_demos_survive_online_capacity_and_are_not_duplicated(self):
    demo = self.write('a', 7, demonstration=True)
    pinned = self.root / 'demonstrations'
    pinned.mkdir()
    (pinned / demo.name).write_bytes(demo.read_bytes())
    self.write('b', 1)
    stats = {}
    sampler = tools.load_episodes(self.root, 1, length=5, capacity=10,
        pinned_directory=pinned, priority_fraction=.5, statistics=stats)
    next(sampler)
    for i in range(8):
      self.write(f'c{i}', i + 10)
      next(sampler)
      self.assertEqual(stats['cached_episodes'], 2)
      self.assertEqual(stats['cached_transitions'], 20)
      self.assertEqual(stats['pinned_episodes'], 1)
    self.assertTrue(any(np.all(next(sampler)['vector'] == 7) for _ in range(40)))
    self.assertEqual(stats['loaded_episodes'], 10)
    self.assertEqual(stats['event_index_builds'], 10)

  def test_unchanged_files_not_reloaded_or_reindexed(self):
    self.write('a')
    stats = {}
    sampler = tools.load_episodes(self.root, 1, length=5, priority_fraction=1, statistics=stats)
    for _ in range(8):
      next(sampler)
    self.assertEqual(stats['scans'], 8)
    self.assertEqual(stats['loaded_episodes'], 1)
    self.assertEqual(stats['event_index_builds'], 1)

  def test_uniform_demo_dataset_ignores_online_capacity(self):
    for i in range(3):
      self.write(str(i), i, demonstration=True)
    config = argparse.Namespace(**dreamer.define_config())
    config.replay_capacity, config.batch_size, config.batch_length = 1, 1, 5
    config.dataset_prefetch = 1
    dataset = iter(dreamer.load_dataset(self.root, config, priority_fraction=0,
                                        demonstrations_only=True))
    seen = {float(next(dataset)['vector'][0, 0, 0]) for _ in range(60)}
    self.assertEqual(seen, {0, 1, 2})

  def test_corrupt_finalized_episode_stops_instead_of_repeating_forever(self):
    (self.root / 'bad-11.npz').write_bytes(b'partial ZIP')
    with self.assertRaisesRegex(RuntimeError, 'finalized replay'):
      next(tools.load_episodes(self.root, 1))

  def test_zero_rescan_time_rejected_instead_of_busy_loop(self):
    self.write('a')
    with self.assertRaisesRegex(ValueError, 'rescan_seconds'):
      next(tools.load_episodes(self.root, 1, rescan_seconds=0))

  def test_slow_scan_still_yields_before_rescanning(self):
    self.write('a')
    sampler = tools.load_episodes(self.root, 100, length=5, rescan_seconds=1e-12)
    for _ in range(3):
      self.assertEqual(next(sampler)['vector'].shape, (5, 13))


class PersistenceTest(unittest.TestCase):
  def setUp(self):
    temporary = tempfile.TemporaryDirectory()
    self.addCleanup(temporary.cleanup)
    self.root = pathlib.Path(temporary.name)

  def test_failed_write_preserves_previous_file_and_hides_temporary(self):
    path = self.root / 'state.pkl'
    path.write_bytes(b'previous')
    def fail(stream):
      stream.write(b'partial')
      raise OSError('injected disk failure')
    with self.assertRaises(OSError):
      run_support.atomic_write(path, fail)
    self.assertEqual(path.read_bytes(), b'previous')
    self.assertEqual(list(self.root.iterdir()), [path])

  def test_failed_episode_write_not_visible_to_replay(self):
    with mock.patch.object(np, 'savez_compressed', side_effect=OSError('injected')):
      with self.assertRaises(OSError):
        tools.save_episodes(self.root, [episode()])
    self.assertFalse(list(self.root.iterdir()))

  def test_disk_guard_checks_before_writing(self):
    with mock.patch('run_support.shutil.disk_usage', return_value=argparse.Namespace(free=1)):
      with self.assertRaisesRegex(RuntimeError, 'disk space'):
        tools.save_episodes(self.root, [episode()], minimum_free_bytes=2)
    self.assertFalse(list(self.root.iterdir()))

  def test_torn_log_tail_recovered_without_losing_complete_records(self):
    path = self.root / 'metrics.jsonl'
    path.write_bytes(b'{"step":1}\n{"step":')
    self.assertEqual(list(run_support.read_jsonl(path)), [dict(step=1)])
    run_support.append_jsonl(path, dict(step=2))
    self.assertEqual(list(run_support.read_jsonl(path)), [dict(step=1), dict(step=2)])
    path.write_bytes(b'{"step":1}')
    run_support.append_jsonl(path, dict(step=2))
    self.assertEqual(len(list(run_support.read_jsonl(path))), 2)

  def test_corrupt_complete_log_line_is_not_silently_ignored(self):
    path = self.root / 'metrics.jsonl'
    path.write_bytes(b'bad\n{"step":2}\n')
    with self.assertRaises(json.JSONDecodeError):
      list(run_support.read_jsonl(path))

  def test_run_lock_is_exclusive_across_processes_and_reusable(self):
    path = self.root / '.run.lock'
    command = [sys.executable, '-c',
        'import sys; from run_support import RunLock; '
        'lock=RunLock(sys.argv[1]); lock.__enter__(); lock.__exit__()', str(path)]
    with run_support.RunLock(path):
      child = subprocess.run(command, capture_output=True, timeout=10)
      self.assertNotEqual(child.returncode, 0)
      self.assertIn(b'already using', child.stderr)
    child = subprocess.run(command, capture_output=True, timeout=10)
    self.assertEqual(child.returncode, 0, child.stderr)

  def test_checkpoint_backup_progress_and_nonfinite_rejection(self):
    config = argparse.Namespace(**dreamer.define_config())
    config.minimum_free_gb = 0
    agent = argparse.Namespace(c=config, variables=[tf.Variable([1.])], warmup=dict(model=2, actor=3))
    path = self.root / 'variables.pkl'
    dreamer.DreamerV2.save(agent, path)
    previous = path.read_bytes()
    agent.variables[0].assign([2.])
    dreamer.DreamerV2.save(agent, path)
    self.assertEqual(path.with_suffix('.prev.pkl').read_bytes(), previous)
    agent.warmup = dict(model=0, actor=0)
    dreamer.DreamerV2.load(agent, path)
    self.assertEqual(agent.warmup, dict(model=2, actor=3))
    previous = path.read_bytes()
    agent.variables[0].assign([np.nan])
    with self.assertRaisesRegex(ValueError, 'non-finite'):
      dreamer.DreamerV2.save(agent, path)
    self.assertEqual(path.read_bytes(), previous)


class TrainingSafetyTest(unittest.TestCase):
  def test_nonfinite_loss_stops_before_optimizer_update(self):
    module = tools.Module()
    module.weight = tf.Variable(1.)
    optimizer = tools.Adam('test', [module], .1, clip=10)
    optimizer.initialize()
    @tf.function
    def update():
      with tf.GradientTape() as tape:
        loss = module.weight * tf.constant(float('nan'))
      return optimizer(tape, loss)
    with self.assertRaises(tf.errors.InvalidArgumentError):
      update()
    self.assertEqual(float(module.weight), 1.)
    self.assertEqual(int(optimizer._opt.iterations), 0)

  def test_partial_model_or_actor_warmup_resumes_remaining_updates(self):
    for progress, expected in ((dict(model=1, actor=0), (2, 4)),
                               (dict(model=3, actor=2), (0, 2))):
      agent = argparse.Namespace(
          c=argparse.Namespace(pretrain=3, actor_pretrain=4, pretrain_checkpoint_every=2,
                               logdir=pathlib.Path('.')),
          warmup=progress, dataset=itertools.repeat({}), behavior_dataset=itertools.repeat({}),
          train=mock.Mock(), train_behavior=mock.Mock(), save=mock.Mock())
      dreamer.DreamerV2.pretrain(agent)
      self.assertEqual((agent.train.call_count, agent.train_behavior.call_count), expected)
      self.assertEqual(agent.warmup, dict(model=3, actor=4))

  def test_online_call_uses_independent_behavior_batch(self):
    online, demonstration = object(), object()
    agent = argparse.Namespace(step=tf.Variable(0, dtype=tf.int64), float=tf.float32,
        should_train=lambda _: True, should_pretrain=lambda: False, should_log=lambda _: False,
        c=argparse.Namespace(train_steps=1, action_repeat=1), dataset=iter([online]),
        behavior_dataset=iter([demonstration]), train=mock.Mock(),
        policy=mock.Mock(return_value=(None, None)))
    dreamer.DreamerV2.__call__(agent, {}, np.array([False]))
    agent.train.assert_called_once_with(online, behavior_data=demonstration)


class BatchSafetyTest(unittest.TestCase):
  def setUp(self):
    temporary = tempfile.TemporaryDirectory()
    self.addCleanup(temporary.cleanup)
    self.root = pathlib.Path(temporary.name)
    (self.root / 'source').mkdir()
    self.status = dict(runner_contract='long_run_safe_v1', source_sha256={}, runs=[dict(seed=i, status='pending',
        outdir=str(self.root / f'seed_{i}'), command=['unused']) for i in range(2)])

  def test_failure_stops_queue_and_records_failure(self):
    child = mock.Mock(pid=999, wait=mock.Mock(return_value=1), poll=mock.Mock(return_value=1))
    with mock.patch('run_batch.subprocess.Popen', return_value=child) as start:
      with self.assertRaises(RuntimeError):
        run_batch.run_entries(self.root, self.status)
    self.assertEqual(start.call_count, 1)
    self.assertEqual(self.status['status'], 'failed')
    self.assertEqual(self.status['runs'][1]['status'], 'pending')

  def test_interrupt_stops_owned_child_and_records_resume_state(self):
    child = mock.Mock(pid=999, wait=mock.Mock(side_effect=KeyboardInterrupt))
    with mock.patch('run_batch.subprocess.Popen', return_value=child), \
         mock.patch('run_batch.stop_child') as stop:
      with self.assertRaises(KeyboardInterrupt):
        run_batch.run_entries(self.root, self.status)
    stop.assert_called_once_with(child)
    self.assertEqual(self.status['status'], 'interrupted')

  def test_resume_skips_completed_seed_and_streams_latest_evaluation(self):
    self.status['runs'][0]['status'] = 'completed'
    run = self.root / 'seed_1'
    run.mkdir()
    for step in (10, 20, 20):
      run_support.append_jsonl(run / 'metrics.jsonl', dict(step=step, **{
          key: 1. for key in ('test/success', 'test/pickup', 'test/out_of_bounds',
                              'test/return', 'test/discounted_return')}))
    with (run / 'metrics.jsonl').open('ab') as stream:
      stream.write(b'{"step":')
    child = mock.Mock(pid=999, wait=mock.Mock(return_value=0))
    with mock.patch('run_batch.subprocess.Popen', return_value=child) as start:
      run_batch.run_entries(self.root, self.status)
    self.assertEqual(start.call_count, 1)
    self.assertEqual(self.status['status'], 'completed')
    self.assertEqual(self.status['runs'][1]['final_step'], 20)

  def test_changed_frozen_source_rejected_before_start(self):
    (self.root / 'source' / 'code.py').write_bytes(b'changed')
    self.status['source_sha256'] = {'code.py': 'old hash'}
    with mock.patch('run_batch.subprocess.Popen') as start:
      with self.assertRaisesRegex(ValueError, 'Frozen source'):
        run_batch.run_entries(self.root, self.status)
    start.assert_not_called()

  def test_legacy_batch_cannot_silently_resume_old_training_code(self):
    self.status.pop('runner_contract')
    with mock.patch('run_batch.subprocess.Popen') as start:
      with self.assertRaisesRegex(ValueError, 'Legacy batch'):
        run_batch.run_entries(self.root, self.status)
    start.assert_not_called()


if __name__ == '__main__':
  unittest.main()
