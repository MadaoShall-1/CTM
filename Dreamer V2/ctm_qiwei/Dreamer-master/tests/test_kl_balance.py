"""Check KL gradient routing: scalar loss alone cannot detect swapped weights."""
import unittest

import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd

import models
import tools


class KLBalanceGradientTest(unittest.TestCase):

  def setUp(self):
    self.rssm = models.RSSM(stoch=1, deter=2, hidden=2, discrete=3)
    self.post = tf.constant([[[[1., -.5, .2]], [[-.3, .7, 1.2]]]])
    self.prior = tf.constant([[[[.1, .8, -.3]], [[.6, -.2, .1]]]])

  def measure(self, forward=False, balance=.8, valid=None, free=0., compiled=False):
    def evaluate(post_logits, prior_logits):
      with tf.GradientTape(persistent=True) as tape:
        tape.watch([post_logits, prior_logits])
        post, prior = dict(logits=post_logits), dict(logits=prior_logits)
        lhs, rhs = (prior, post) if forward else (post, prior)
        divergence = tfd.kl_divergence(
            self.rssm.get_dist(lhs), self.rssm.get_dist(rhs))
        reference = (tf.reduce_mean(divergence) if valid is None
                     else tools.masked_mean(divergence, valid))
        loss, logged_kl = self.rssm.kl_loss(
            post, prior, balance=balance, forward=forward, valid=valid, free=free)
      gradients = tape.gradient(loss, [post_logits, prior_logits])
      reference_gradients = tape.gradient(reference, [post_logits, prior_logits])
      return loss, logged_kl, reference, gradients, reference_gradients
    function = tf.function(evaluate) if compiled else evaluate
    return function(self.post, self.prior)

  def assert_weights(self, forward, balance, **kwargs):
    loss, _, reference, gradients, reference_gradients = self.measure(
        forward=forward, balance=balance, **kwargs)
    np.testing.assert_allclose(loss, reference, rtol=1e-6, atol=1e-7)
    for gradient, reference_gradient, weight in zip(
        gradients, reference_gradients, (1. - balance, balance)):
      self.assertIsNotNone(gradient)
      self.assertGreater(float(tf.linalg.norm(reference_gradient)), 0.)
      np.testing.assert_allclose(
          gradient, weight * reference_gradient, rtol=2e-6, atol=1e-7)
    return gradients

  def test_default_backward_kl_trains_prior_with_point_eight(self):
    self.assert_weights(forward=False, balance=.8)

  def test_forward_kl_keeps_prior_weight(self):
    self.assert_weights(forward=True, balance=.8)

  def test_boundary_weights_route_to_only_one_side(self):
    for forward in (False, True):
      for balance in (0., 1.):
        with self.subTest(forward=forward, balance=balance):
          self.assert_weights(forward, balance)

  def test_half_balance_preserves_existing_half_gradient_semantics(self):
    for forward in (False, True):
      with self.subTest(forward=forward):
        self.assert_weights(forward, .5)

  def test_padding_has_zero_gradient_and_correct_valid_gradient(self):
    valid = tf.constant([[1., 0.]])
    for forward in (False, True):
      with self.subTest(forward=forward):
        gradients = self.assert_weights(forward, .8, valid=valid)
        for gradient in gradients:
          np.testing.assert_array_equal(gradient[:, 1], 0.)

  def test_free_floor_blocks_both_gradients(self):
    for forward in (False, True):
      with self.subTest(forward=forward):
        loss, _, reference, gradients, _ = self.measure(forward=forward, free=10.)
        self.assertLess(float(reference), 10.)
        self.assertEqual(float(loss), 10.)
        for gradient in gradients:
          np.testing.assert_array_equal(gradient, 0.)

  def test_all_padding_is_finite_and_zero(self):
    loss, logged, reference, gradients, _ = self.measure(valid=tf.zeros([1, 2]))
    np.testing.assert_array_equal([loss, logged, reference], [0., 0., 0.])
    for gradient in gradients:
      np.testing.assert_array_equal(gradient, 0.)

  def test_compiled_gradients_match_eager_routing(self):
    for forward in (False, True):
      with self.subTest(forward=forward):
        self.assert_weights(forward, .8, compiled=True)


if __name__ == '__main__':
  unittest.main()
