import unittest

import numpy as np

from diagnose_world_model import decode_recorded_action, errors, sampling_weights


class DiagnosticMetricTest(unittest.TestCase):
  def test_recorded_action_decode_supports_current_and_historical_widths(self):
    self.assertEqual(decode_recorded_action([0, 1, .9, -.6], 2), (1, -.6))
    self.assertEqual(decode_recorded_action([0, 0, 1, .9, .8, 0], 3), (2, 0))

  def test_recorded_action_decode_rejects_mixed_contracts(self):
    with self.assertRaises(ValueError):
      decode_recorded_action([0, 0, 1, 0, 0, 0], 2)

  def test_physical_units_and_angle_wrap(self):
    target = np.zeros((2, 13), np.float32)
    target[:, 7] = -1
    predicted = target.copy()
    predicted[:, 0] = .5
    predicted[:, 2] = .2
    predicted[:, 3] = 2
    metric = errors(predicted, target)
    self.assertAlmostEqual(metric['position_mae_m'], 500)
    self.assertAlmostEqual(metric['speed_mae'], 4, places=5)
    self.assertLess(metric['heading_mae_deg'], 1e-5)
    self.assertIsNone(metric['phase1_recall'])

  def test_phase_recall_does_not_confuse_majority_accuracy(self):
    target = np.zeros((100, 13), np.float32)
    target[:, 7] = -1
    target[-1, 7] = 1
    predicted = target.copy()
    predicted[:, 7] = -1
    metric = errors(predicted, target)
    self.assertEqual(metric['phase_accuracy'], .99)
    self.assertEqual(metric['phase1_recall'], 0)

  def test_uniform_and_balanced_inclusion(self):
    uniform = sampling_weights(101, 20)
    balanced = sampling_weights(101, 20, balance=True)
    self.assertAlmostEqual(uniform.sum(), 20)
    self.assertAlmostEqual(balanced.sum(), 20)
    self.assertAlmostEqual(uniform[-1], 1 / 82)
    self.assertAlmostEqual(balanced[-1], 20 / 101)
    np.testing.assert_equal(sampling_weights(5, 20), np.ones(5))


if __name__ == '__main__':
  unittest.main()
