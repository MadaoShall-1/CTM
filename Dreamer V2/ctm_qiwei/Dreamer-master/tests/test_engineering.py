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
    action = np.asarray([1, 0, 0, 0], np.float32)
    next_obs, _, _, _ = env.step(action)
    self.assertEqual(obs['vector'].shape, env.observation_space['vector'].shape)
    self.assertFalse(np.allclose(obs['vector'], next_obs['vector']))
    self.assertNotIn('collision', next_obs)

  def test_obstacle_variant_is_not_part_of_the_scene(self):
    with self.assertRaises(ValueError):
      DreamerV2UAVEnv('relay_obstacles')

  def test_vector_v2_heading_is_continuous_and_keeps_both_goals(self):
    env = DreamerV2UAVEnv('relay', seed=7)
    env.reset()
    env.env.state = UAVState(500, 600, 12, np.pi - 1e-4)
    before = env._vector_observation(env.env._get_obs())
    env.env.state = UAVState(500, 600, 12, -np.pi + 1e-4)
    after = env._vector_observation(env.env._get_obs())
    self.assertEqual(before.shape, (13,))
    self.assertLess(np.linalg.norm(before[3:5] - after[3:5]), 1e-3)
    np.testing.assert_allclose(before[5:9], after[5:9])
    self.assertAlmostEqual(float(np.linalg.norm(after[3:5])), 1., places=5)

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
      for action in ([1, 0, .964, 0], [1, 0, 0, 0]):
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

  def test_automatic_pickup_and_delivery_still_required(self):
    env = DreamerV2UAVEnv('relay', seed=0, shaping_scale=0)
    env.reset()
    env.env.reset(options=dict(start=[500, 500], relay_goal=[500, 500],
                               final_goal=[1500, 500], speed=0, heading=0))
    obs, reward, done, _ = env.step(np.array([1, 0, 0, 0], np.float32))
    self.assertFalse(done)
    self.assertEqual(float(obs['carrying_supply']), 1.)
    env.env.state = UAVState(1490, 500, 0, 0)
    obs, reward, done, _ = env.step(np.array([1, 0, 0, 0], np.float32))
    self.assertTrue(done)
    self.assertEqual(float(obs['is_success']), 1.)
    self.assertEqual(reward, 1.)

  def test_return_order_for_every_completion_time(self):
    for steps in range(1, 101):
      prefix = sum(-.01 * .99 ** t for t in range(steps-1))
      self.assertAlmostEqual(prefix - .99 ** (steps-1), -1., places=6)
      self.assertGreater(prefix + .99 ** (steps-1), -1.)


class ActionContractTest(unittest.TestCase):

  def test_old_six_channel_actions_are_rejected(self):
    for task in ('direct', 'relay'):
      env = DreamerV2UAVEnv(task)
      self.assertEqual(env.num_actions, 2)
      self.assertEqual(env.action_space.shape, (4,))
      with self.assertRaises(ValueError):
        env.decode_action(np.zeros(6, np.float32))
    with self.assertRaises(ValueError):
      models.canonical_action(tf.zeros([1, 6]))
    with self.assertRaises(ValueError):
      uav_actions.canonicalize_numpy(np.zeros(6, np.float32), 3)

  def test_both_actions_keep_their_selected_parameter(self):
    for selected in (0, 1):
      logits = np.full([1, 2], -1e9, np.float32)
      logits[0, selected] = 0
      dist = models.HybridDist(tf.constant(logits), tf.ones([1, 2]), tf.ones([1, 2]))
      for action in (dist.mode(), dist.sample()):
        self.assertEqual(action.shape, (1, 4))
        self.assertEqual(float(action[0, 2 + (1 - selected)]), 0.)
        self.assertNotEqual(float(action[0, 2 + selected]), 0.)

  def test_both_parameter_branches_have_entropy_gradients(self):
    mean = tf.Variable([[1., 2.]])
    std = tf.Variable([[.2, .3]])
    with tf.GradientTape() as tape:
      dist = models.HybridDist(tf.zeros([1, 2]), mean, std)
      objective = tf.reduce_sum(dist.parameter_entropy())
    grads = tape.gradient(objective, [mean, std])
    for grad in grads:
      self.assertTrue(np.all(np.isfinite(grad)))
      self.assertTrue(np.all(np.abs(grad) > 0))

  def test_reset_action_stays_zero(self):
    actions = tf.constant([[0., 0., 0., 0.], [1., 0., .5, .7]])
    expected = [[0, 0, 0, 0], [1, 0, .5, 0]]
    np.testing.assert_allclose(dreamer.canonicalize_action(actions, 2), expected)

  def test_replay_actions_are_canonicalized(self):
    action = tf.constant([[0.1, 0.9, -0.4, 0.7]])
    actual = dreamer.canonicalize_action(action, 2).numpy()
    expected = np.asarray([[0, 1, 0, 0.7]], np.float32)
    np.testing.assert_allclose(actual, expected)

  def test_rssm_boundary_ignores_unexecuted_parameters(self):
    rssm = models.RSSM(stoch=2, deter=8, hidden=8, discrete=3)
    state = rssm.initial(1)
    first = rssm.img_step(state, tf.constant([[1., 0., .2, .7]]))
    second = rssm.img_step(state, tf.constant([[1., 0., .2, -.3]]))
    np.testing.assert_allclose(first['logits'], second['logits'])
    np.testing.assert_allclose(first['deter'], second['deter'])

  def test_random_prefill_only_populates_selected_parameter(self):
    agent = dreamer.make_random_agent(2, seed=5)
    actions, _ = agent({}, np.zeros(100, bool), None)
    selections, parameters = actions[:, :2], actions[:, 2:]
    np.testing.assert_allclose(selections.sum(-1), 1.0)
    np.testing.assert_allclose(parameters * (1.0 - selections), 0.0)
    self.assertEqual(set(selections.argmax(-1)), {0, 1})


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

  def test_priority_sampling_anchors_every_window_on_a_task_event(self):
    with tempfile.TemporaryDirectory() as directory:
      length = 30
      episode = dict(
          marker=np.arange(length, dtype=np.int32),
          phase=np.r_[np.zeros(12), np.ones(length - 12)].astype(np.float32),
          is_success=np.r_[np.zeros(length - 1), 1].astype(np.float32),
          discount=np.r_[np.ones(length - 1), 0].astype(np.float32),
          reward=np.zeros(length, np.float32))
      tools.save_episodes(directory, [episode])
      sampler = tools.load_episodes(
          directory, 20, length=6, seed=3, priority_fraction=1.)
      for _ in range(80):
        sample = next(sampler)
        indices = set(sample['marker'].tolist())
        self.assertTrue(12 in indices or 29 in indices)


class ReplayBootstrapTest(unittest.TestCase):

  def test_behavior_targets_are_aligned_to_source_observations(self):
    states = tf.constant([[[10.], [20.], [30.]]])
    actions = tf.constant([[[0.], [1.], [2.]]])
    inputs, targets, weights = dreamer.align_behavior_supervision(
        states, actions, tf.constant([[0., 1., 1.]]),
        tf.constant([[1., 1., 1.]]))
    np.testing.assert_array_equal(inputs, [[[10.], [20.]]])
    np.testing.assert_array_equal(targets, [[[1.], [2.]]])
    np.testing.assert_array_equal(weights, [[1., 1.]])

  def test_controller_bootstrap_supplies_pickup_success_and_terminal(self):
    with tempfile.TemporaryDirectory() as directory:
      config = argparse.Namespace(**dreamer.define_config())
      config.demo_episodes = 2
      config.demo_seed_start = 41000
      datadir = pathlib.Path(directory) / 'episodes'
      summary = dreamer.collect_controller_demonstrations(config, datadir)
      self.assertEqual(summary['episodes'], 2)
      episodes = [np.load(path) for path in datadir.glob('*.npz')]
      self.assertEqual(len(episodes), 2)
      for episode in episodes:
        self.assertTrue(np.any(episode['phase'] > 0))
        self.assertEqual(float(episode['is_success'][-1]), 1.)
        self.assertEqual(float(episode['discount'][-1]), 0.)
        self.assertEqual(episode['action'].shape[-1], 4)
        canonical = uav_actions.canonicalize_numpy(episode['action'], 2)
        np.testing.assert_array_equal(episode['action'], canonical)
        self.assertEqual(float(episode['demonstration'][0]), 0.)
        np.testing.assert_array_equal(episode['demonstration'][1:], 1.)
        episode.close()

  def test_clone_preserves_frequency_and_has_finite_turn_gradient(self):
    logits = tf.Variable(np.tile([2., 0.], (10, 1)), dtype=tf.float32)
    mean = tf.zeros([10, 2])
    std = tf.ones([10, 2])
    actions = np.zeros([10, 4], np.float32)
    actions[:9, 0] = 1
    actions[9, 1] = 1
    with tf.GradientTape() as tape:
      dist = models.HybridDist(logits, mean, std)
      loss = dreamer.behavior_cloning_loss(
          dist, tf.constant(actions), tf.ones(10))
    gradient = tape.gradient(loss, logits).numpy()
    self.assertTrue(np.isfinite(float(loss)))
    self.assertTrue(np.all(np.isfinite(gradient)))


class ExplorationTest(unittest.TestCase):

  def test_entropy_pushes_saturated_mean_back_towards_zero(self):
    mean = tf.Variable([[20., 20.]])
    with tf.GradientTape() as tape:
      dist = models.HybridDist(tf.zeros([1, 2]), mean, tf.ones([1, 2]) * .1)
      entropy = tf.reduce_sum(dist.entropy())
    gradient = tape.gradient(entropy, mean).numpy()
    self.assertTrue(np.all(np.isfinite(gradient)))
    self.assertTrue(np.all(gradient[:, :2] < -.5))

  def test_mixture_keeps_all_actions_sampleable(self):
    actor = models.HybridActionDecoder(2, layers=0, unimix=.01)
    actor(tf.zeros([1, 2]))
    actor._modules['logits'].kernel.assign(tf.zeros([2, 2]))
    actor._modules['logits'].bias.assign([1000., -1000.])
    probs = tf.nn.softmax(actor(tf.zeros([1, 2])).logits).numpy()[0]
    np.testing.assert_allclose(probs, [.995, .005], rtol=1e-5)


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
      action = np.asarray([1, 0, .5, 0], np.float32)
      observations.append(env.step(action)[0])
      observations.append(env.step(action)[0])
      episode = {key: np.stack([obs[key] for obs in observations])
                 for key in observations[0]}
      episode.update(action=np.stack([np.zeros(4, np.float32), action, action]),
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
