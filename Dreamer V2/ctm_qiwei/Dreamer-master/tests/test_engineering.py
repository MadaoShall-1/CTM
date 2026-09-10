import unittest
import pathlib
import tempfile
import argparse

import numpy as np
import tensorflow as tf

import dreamer
import models
import tools
import uav_actions
from controller_baseline import evaluate
from envs import DreamerV2UAVEnv, UAVState


class EnvironmentContractTest(unittest.TestCase):

  def test_original_relay_scene_exposes_structured_vector(self):
    env = DreamerV2UAVEnv('relay', seed=7)
    obs = env.reset()
    action = np.asarray([0, 0, 1, 0, 0, 0], np.float32)
    next_obs, _, _, _ = env.step(action)
    self.assertEqual(obs['vector'].shape, env.observation_space['vector'].shape)
    self.assertFalse(np.allclose(obs['vector'], next_obs['vector']))
    self.assertNotIn('collision', next_obs)

  def test_obstacle_variant_is_not_part_of_the_scene(self):
    with self.assertRaises(ValueError):
      DreamerV2UAVEnv('relay_obstacles')

  def test_out_of_bounds_keeps_original_sparse_reward(self):
    env = DreamerV2UAVEnv('direct', seed=3, reward_mode='legacy')
    env.reset()
    env.env.state = UAVState(1999.0, 1000.0, 40.0, 0.0)
    env.env.goal = np.asarray([0.0, 0.0], np.float32)
    action = np.asarray([1, 0, 0, 0], np.float32)
    _, reward, done, info = env.step(action)
    self.assertTrue(done)
    self.assertTrue(info['out_of_bounds'])
    self.assertEqual(reward, -1.0)


class SafeRewardTest(unittest.TestCase):

  def test_layouts_are_separated_and_within_geometric_budget(self):
    for task in ('direct', 'relay'):
      for seed in range(20):
        env = DreamerV2UAVEnv(task, seed=seed)
        env.reset()
        self.assertTrue(env._valid_layout())

  def test_failure_return_does_not_reward_early_exit(self):
    for seed in range(10):
      scores = []
      for action in ([1, 0, 0, .964, 0, 0], [0, 0, 1, 0, 0, 0]):
        env = DreamerV2UAVEnv('relay', seed=seed)
        env.reset()
        potential = env._potential()
        discounted, base = 0., 0.
        for step in range(100):
          obs, reward, done, info = env.step(np.array(action, np.float32))
          discounted += .99 ** step * reward
          base += .99 ** step * info['base_reward']
          if done:
            break
        self.assertTrue(done)
        self.assertEqual(float(info['discount']), 0.)
        self.assertFalse(bool(obs['is_success']))
        self.assertAlmostEqual(base, -1., places=6)
        self.assertAlmostEqual(discounted, -1. - potential, places=6)
        scores.append(discounted)
      self.assertAlmostEqual(scores[0], scores[1], places=6)

  def test_success_reward_and_boundary_priority(self):
    env = DreamerV2UAVEnv('direct', seed=2, shaping_scale=0)
    env.reset()
    env.env.state = UAVState(1000, 1000, 0, 0)
    env.env.goal = np.array([1000, 1000], np.float32)
    obs, reward, done, info = env.step(np.array([1, 0, 0, 0], np.float32))
    self.assertTrue(done)
    self.assertEqual(reward, 1.)
    self.assertEqual(float(obs['is_success']), 1.)
    self.assertEqual(float(info['discount']), 0.)
    env.reset()
    env.env.state = UAVState(1999, 1000, 40, 0)
    env.env.goal = np.array([2000, 1000], np.float32)
    obs, reward, done, info = env.step(np.array([1, 0, 0, 0], np.float32))
    self.assertTrue(done)
    self.assertEqual(reward, -1.)
    self.assertEqual(float(obs['is_success']), 0.)

  def test_catch_pickup_and_delivery_still_required(self):
    env = DreamerV2UAVEnv('relay', seed=0, shaping_scale=0)
    env.reset()
    env.env.reset(options=dict(start=[500, 500], relay_goal=[500, 500],
                               final_goal=[1500, 500], speed=0, heading=0))
    obs, reward, done, _ = env.step(np.array([1, 0, 0, 0, 0, 0], np.float32))
    self.assertFalse(done)
    self.assertEqual(float(obs['carrying_supply']), 0.)
    obs, reward, done, _ = env.step(np.array([0, 0, 1, 0, 0, 0], np.float32))
    self.assertFalse(done)
    self.assertEqual(float(obs['carrying_supply']), 1.)
    env.env.state = UAVState(1490, 500, 0, 0)
    obs, reward, done, _ = env.step(np.array([0, 0, 1, 0, 0, 0], np.float32))
    self.assertTrue(done)
    self.assertEqual(float(obs['is_success']), 1.)
    self.assertEqual(reward, 1.)

  def test_return_order_for_every_completion_time(self):
    for steps in range(1, 101):
      prefix = sum(-.01 * .99 ** t for t in range(steps-1))
      self.assertAlmostEqual(prefix - .99 ** (steps-1), -1., places=6)
      self.assertGreater(prefix + .99 ** (steps-1), -1.)


class ActionContractTest(unittest.TestCase):

  def test_catch_is_parameter_free_in_all_action_paths(self):
    catch = tf.constant([[0., 0., 1., .1, .2, .9]])
    expected = [[0., 0., 1., 0., 0., 0.]]
    np.testing.assert_array_equal(dreamer.canonicalize_action(catch, 3), expected)
    np.testing.assert_array_equal(uav_actions.canonicalize_numpy(catch, 3), expected)
    env = DreamerV2UAVEnv('relay')
    discrete, parameter = env.decode_action(catch.numpy()[0])
    self.assertEqual(discrete, 2)
    np.testing.assert_array_equal(parameter, [0.])
    dist = models.HybridDist(tf.constant([[-1e9, -1e9, 0.]]),
                             tf.ones([1, 3]) * 10, tf.ones([1, 3]))
    np.testing.assert_array_equal(dist.sample(), expected)
    np.testing.assert_array_equal(dist.mode(), expected)

  def test_catch_parameter_has_no_dynamics_or_entropy_gradient(self):
    mean = tf.Variable([[1., 2., 3.]])
    std = tf.Variable([[.2, .3, .4]])
    with tf.GradientTape() as tape:
      dist = models.HybridDist(tf.zeros([1, 3]), mean, std)
      objective = tf.reduce_sum(dist.sample() + dist.mode()) + tf.reduce_sum(dist.parameter_entropy())
    grads = tape.gradient(objective, [mean, std])
    for grad in grads:
      self.assertEqual(float(grad[0, 2]), 0.)

  def test_reset_action_stays_zero(self):
    actions = tf.constant([[0., 0., 0., 0., 0., 0.],
                           [1., 0., 0., .5, .7, .9]])
    expected = [[0, 0, 0, 0, 0, 0], [1, 0, 0, .5, 0, 0]]
    np.testing.assert_allclose(dreamer.canonicalize_action(actions, 3), expected)

  def test_replay_actions_are_canonicalized(self):
    action = tf.constant([[0.1, 0.9, 0.2, -0.4, 0.7, -0.8]])
    actual = dreamer.canonicalize_action(action, 3).numpy()
    expected = np.asarray([[0, 1, 0, 0, 0.7, 0]], np.float32)
    np.testing.assert_allclose(actual, expected)

  def test_random_prefill_only_populates_selected_parameter(self):
    agent = dreamer.make_random_agent(3, seed=5)
    actions, _ = agent({}, np.zeros(100, bool), None)
    selections, parameters = actions[:, :3], actions[:, 3:]
    np.testing.assert_allclose(selections.sum(-1), 1.0)
    np.testing.assert_allclose(parameters * (1.0 - selections), 0.0)
    np.testing.assert_array_equal(parameters[:, 2], 0.)


class ReturnAlignmentTest(unittest.TestCase):

  def test_successor_values_and_bootstrap(self):
    reward = tf.constant([-1., -1.])
    next_value = tf.constant([10., 20.])
    discount = tf.constant([.9, .9])
    for lam, expected in [(0., [8., 17.]), (.95, [13.985, 17.]),
                          (1., [14.3, 17.])]:
      actual = tools.lambda_return_from_next_value(reward, next_value, discount, lam)
      np.testing.assert_allclose(actual, expected, rtol=1e-6)

  def test_terminal_transition_blocks_future_value(self):
    actual = tools.lambda_return_from_next_value(
        tf.constant([-1., -1., -1.]), tf.constant([10., 20., 1000.]),
        tf.constant([.9, 0., .9]), .95)
    np.testing.assert_allclose(actual[:2], [-1.405, -1.], atol=1e-5)


class ReplayMaskTest(unittest.TestCase):

  def test_short_failure_is_sampled_with_terminal_and_mask(self):
    with tempfile.TemporaryDirectory() as directory:
      episode = dict(reward=np.array([0., -1., -1.], np.float32),
                     discount=np.array([1., 1., 0.], np.float32),
                     action=np.array([[0., 0.], [1., .4], [1., .5]], np.float32))
      tools.save_episodes(directory, [episode])
      actual = next(tools.load_episodes(directory, 1, length=6))
      np.testing.assert_array_equal(actual['valid'], [1, 1, 1, 0, 0, 0])
      np.testing.assert_array_equal(actual['reward'], [0, -1, -1, 0, 0, 0])
      np.testing.assert_array_equal(actual['discount'], [1, 1, 0, 0, 0, 0])
      np.testing.assert_array_equal(actual['action'][3:], 0)
      # Sampling must not modify the cached original episode.
      original = next(tools.load_episodes(directory, 1))
      np.testing.assert_array_equal(original['reward'], episode['reward'])

  def test_padding_has_zero_loss_and_gradient(self):
    values = tf.Variable([2., 4., 100000.])
    with tf.GradientTape() as tape:
      loss = tools.masked_mean(values, tf.constant([1., 1., 0.]))
    self.assertEqual(float(loss), 3.)
    np.testing.assert_allclose(tape.gradient(loss, values), [.5, .5, 0.])
    self.assertEqual(float(tools.masked_mean(values, tf.zeros(3))), 0.)

  def test_kl_excludes_padding(self):
    rssm = models.RSSM(stoch=1, discrete=2)
    post = {'logits': tf.constant([[[[1., -1.]], [[50., -50.]]]])}
    prior = {'logits': tf.constant([[[[0., 0.]], [[-50., 50.]]]])}
    loss, kl = rssm.kl_loss(post, prior, valid=tf.constant([[1., 0.]]))
    short_post = {'logits': post['logits'][:, :1]}
    short_prior = {'logits': prior['logits'][:, :1]}
    reference_loss, reference_kl = rssm.kl_loss(short_post, short_prior)
    np.testing.assert_allclose([loss, kl], [reference_loss, reference_kl])


class ExplorationTest(unittest.TestCase):

  def test_entropy_pushes_saturated_mean_back_towards_zero(self):
    mean = tf.Variable([[20., 20., 20.]])
    with tf.GradientTape() as tape:
      dist = models.HybridDist(tf.zeros([1, 3]), mean, tf.ones([1, 3]) * .1)
      entropy = tf.reduce_sum(dist.entropy())
    gradient = tape.gradient(entropy, mean).numpy()
    self.assertTrue(np.all(np.isfinite(gradient)))
    self.assertTrue(np.all(gradient[:, :2] < -.5))
    self.assertEqual(float(gradient[0, 2]), 0.)

  def test_mixture_keeps_all_actions_sampleable(self):
    actor = models.HybridActionDecoder(3, layers=0, unimix=.01)
    actor(tf.zeros([1, 2]))
    actor._modules['logits'].kernel.assign(tf.zeros([2, 3]))
    actor._modules['logits'].bias.assign([1000., -1000., -1000.])
    probs = tf.nn.softmax(actor(tf.zeros([1, 2])).logits).numpy()[0]
    np.testing.assert_allclose(probs, [.99333333, .00333333, .00333333], rtol=1e-5)


class TrainingIntegrationTest(unittest.TestCase):

  def test_padded_batch_initialization_update_and_checkpoint(self):
    with tempfile.TemporaryDirectory() as directory:
      config = argparse.Namespace(**dreamer.define_config())
      config.logdir = pathlib.Path(directory)
      for key, value in dict(batch_size=2, batch_length=5, train_steps=1,
          dataset_prefetch=1, rssm_hidden=16, rssm_deter=16, rssm_stoch=2,
          rssm_discrete=2, cnn_depth=2, vector_units=8, num_units=16,
          imag_horizon=3).items():
        setattr(config, key, value)
      env = DreamerV2UAVEnv('relay', seed=7)
      observations = [env.reset()]
      action = np.asarray([1, 0, 0, .5, 0, 0], np.float32)
      observations.append(env.step(action)[0])
      observations.append(env.step(action)[0])
      episode = {key: np.stack([obs[key] for obs in observations])
                 for key in observations[0]}
      episode.update(action=np.stack([np.zeros(6, np.float32), action, action]),
                     reward=np.array([0, -1, -1], np.float32),
                     discount=np.array([1, 1, 0], np.float32))
      datadir = config.logdir / 'episodes'
      tools.save_episodes(datadir, [episode])
      agent = dreamer.DreamerV2(config, datadir, env.action_space,
                              env.observation_space, None)
      self.assertEqual(int(agent._updates), 0)
      self.assertEqual(int(agent.model_opt._opt.iterations), 0)
      batch = next(agent.dataset)
      np.testing.assert_array_equal(batch['valid'], [[1, 1, 1, 0, 0]] * 2)
      behavior = [np.array(v.numpy(), copy=True) for module in
                  (agent.actor, agent.critic, agent.slow_critic) for v in module.variables]
      model_before = [np.array(v.numpy(), copy=True) for v in agent.encoder.variables]
      agent.train(batch, world_model_only=True)
      self.assertEqual(int(agent._updates), 0)
      self.assertEqual(int(agent.model_opt._opt.iterations), 1)
      self.assertEqual(int(agent.actor_opt._opt.iterations), 0)
      self.assertEqual(int(agent.critic_opt._opt.iterations), 0)
      self.assertTrue(any(not np.array_equal(a, b.numpy())
                          for a, b in zip(model_before, agent.encoder.variables)))
      for old, now in zip(behavior, [v for module in
          (agent.actor, agent.critic, agent.slow_critic) for v in module.variables]):
        np.testing.assert_array_equal(old, now.numpy())
      agent.train(batch)
      self.assertEqual(int(agent._updates), 1)
      for metric in agent.metrics.values():
        self.assertTrue(np.isfinite(float(metric.result())))
      checkpoint = config.logdir / 'test.pkl'
      agent.save(checkpoint)
      saved = [np.array(v.numpy(), copy=True) for v in agent.variables]
      agent.train(batch)
      agent.load(checkpoint)
      for expected, variable in zip(saved, agent.variables):
        np.testing.assert_array_equal(variable.numpy(), expected)
      env.close()


class ControllerBaselineTest(unittest.TestCase):

  def test_feedback_controller_completes_fixed_relay_scenarios(self):
    result = evaluate(seeds=range(10000, 10010))
    self.assertEqual(result['success_rate'], 1.)
    self.assertEqual(result['pickup_rate'], 1.)
    self.assertEqual(result['out_of_bounds_rate'], 0.)


if __name__ == '__main__':
  unittest.main()
