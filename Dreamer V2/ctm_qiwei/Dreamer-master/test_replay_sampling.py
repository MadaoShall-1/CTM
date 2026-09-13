"""Replay filtering regressions. Run: python test_replay_sampling.py"""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import tools


class ReplaySamplingTests(unittest.TestCase):
  def setUp(self):
    self.tmp = tempfile.TemporaryDirectory()
    self.addCleanup(self.tmp.cleanup)
    self.directory = Path(self.tmp.name)

  def episode(self, name, length, marker=1):
    np.savez_compressed(self.directory / f'{name}-{length}.npz',
        vector=np.full((length, 13), marker, np.float32),
        action=np.zeros((length, 4), np.float32),
        phase=np.zeros(length, np.float32),
        is_success=np.zeros(length, np.float32),
        discount=np.r_[np.ones(max(0, length - 1)), 0].astype(np.float32),
        reward=np.arange(length, dtype=np.float32))

  def test_short_episode_is_padded_without_repeated_messages(self):
    self.episode('short', 9, 9)
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
      sampler = tools.load_episodes(self.directory, 5, length=10)
      for _ in range(5):
        sample = next(sampler)
        self.assertEqual(sample['vector'].shape, (10, 13))
        np.testing.assert_array_equal(sample['valid'], [1] * 9 + [0])
        self.assertEqual(float(sample['discount'][8]), 0.)
    self.assertNotIn('Replay filter:', output.getvalue())
    self.assertNotIn('Skipped short episode', output.getvalue())

  def test_all_short_is_still_trainable(self):
    self.episode('short', 9)
    sample = next(tools.load_episodes(self.directory, 1, length=10))
    np.testing.assert_array_equal(sample['valid'], [1] * 9 + [0])

  def test_empty_directory_fails(self):
    with self.assertRaisesRegex(RuntimeError, 'No episodes found'):
      next(tools.load_episodes(self.directory, 1, length=10))

  def test_exact_length_and_full_episode_modes(self):
    self.episode('exact', 10)
    for balance in (False, True):
      sample = next(tools.load_episodes(self.directory, 1, 10, balance))
      np.testing.assert_array_equal(sample['reward'], np.arange(10))
    self.assertEqual(len(next(tools.load_episodes(self.directory, 1))['reward']), 10)

  def test_rescan_occurs_after_yielded_sequence_count(self):
    self.episode('a', 10)
    original_glob = Path.glob
    calls = []
    def counted_glob(path, pattern):
      calls.append(pattern)
      return original_glob(path, pattern)
    with mock.patch.object(Path, 'glob', counted_glob):
      sampler = tools.load_episodes(self.directory, 4, 10)
      for _ in range(4):
        next(sampler)
      self.assertEqual(len(calls), 1)
      self.episode('b', 10, 2)
      next(sampler)
      self.assertEqual(len(calls), 2)
      self.assertTrue(any(np.all(next(sampler)['vector'] == 2) for _ in range(30)))

  def test_capacity_evicts_old_cached_episodes(self):
    self.episode('a', 10, 1)
    sampler = tools.load_episodes(self.directory, 1, 10, capacity=9)
    self.assertTrue(np.all(next(sampler)['vector'] == 1))
    self.episode('b', 10, 2)
    self.assertTrue(np.all(next(sampler)['vector'] == 2))

  def test_rescans_do_not_emit_short_episode_noise(self):
    self.episode('long', 10)
    self.episode('short', 9)
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
      sampler = tools.load_episodes(self.directory, 1, 10)
      next(sampler)
      self.episode('another_long', 20)
      next(sampler)
      self.episode('another_short', 8)
      next(sampler)
      next(sampler)
    self.assertEqual(output.getvalue(), '')

  def test_zero_rescan_fails_instead_of_spinning(self):
    with self.assertRaisesRegex(ValueError, 'rescan must be at least 1'):
      next(tools.load_episodes(self.directory, 0, 10))


if __name__ == '__main__':
  unittest.main(verbosity=2)
