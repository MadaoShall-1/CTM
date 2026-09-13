"""Regression tests for the two-action, post-motion automatic pickup contract."""
import argparse
import json
import pathlib
import pickle
import tempfile
import unittest

import numpy as np
import tensorflow as tf

import dreamer
import tools
import uav_actions
from envs import (DreamerV2UAVEnv, DynamicsConfig, MultiRelayNavigationEnv,
                  RelayNavigationEnv, UAVState, advance)


class AutomaticPickupTest(unittest.TestCase):
  def environment(self, **options):
    env = DreamerV2UAVEnv('relay', shaping_scale=0)
    defaults = dict(start=[360, 500], relay_goal=[500, 500],
                    final_goal=[1500, 500], speed=40, heading=0)
    defaults.update(options)
    env.env.reset(options=defaults)
    self.addCleanup(env.close)
    return env

  def test_pickup_on_movement_step_synchronizes_all_state(self):
    env = self.environment()
    obs, reward, done, info = env.step(np.array([1, 0, 0, 0], np.float32))
    self.assertFalse(done)
    self.assertAlmostEqual(reward, -.01)
    self.assertTrue(info['phase_changed'])
    self.assertEqual(env.env.elapsed_steps, 1)
    np.testing.assert_array_equal(env.env.state.position, [400, 500])
    np.testing.assert_array_equal(env.env.supply_position, [400, 500])
    np.testing.assert_array_equal(obs['achieved_goal'], [400, 500])
    np.testing.assert_array_equal(obs['desired_goal'], [1500, 500])
    self.assertEqual(float(obs['phase']), 1.)
    self.assertEqual(float(obs['carrying_supply']), 1.)
    self.assertEqual(float(obs['vector'][10]), 1.)
    np.testing.assert_allclose(obs['vector'][11:13], [.55, 0])
    self.assertEqual(float(info['discount']), 1.)
    obs, _, _, info = env.step(np.array([1, 0, 0, 0], np.float32))
    self.assertFalse(info['phase_changed'])
    np.testing.assert_array_equal(obs['achieved_goal'], [440, 500])

  def test_radius_is_inclusive_and_not_checked_before_motion(self):
    for start_x, pickup in ((359, False), (360, True), (361, True)):
      with self.subTest(start_x=start_x):
        env = self.environment(start=[start_x, 500])
        obs, _, _, _ = env.step(np.array([1, 0, 0, 0], np.float32))
        self.assertEqual(bool(obs['carrying_supply']), pickup)
    env = self.environment(start=[590, 500])
    obs, _, _, _ = env.step(np.array([1, 0, 0, 0], np.float32))
    self.assertEqual(float(obs['carrying_supply']), 0.)  # Ends outside the disk.

  def test_turn_can_trigger_pickup_without_a_third_action(self):
    env = self.environment(start=[380, 500], heading=-np.pi / 3)
    obs, _, done, _ = env.step(np.array([0, 1, 0, 1], np.float32))
    self.assertFalse(done)
    self.assertEqual(float(obs['carrying_supply']), 1.)
    np.testing.assert_allclose(obs['achieved_goal'], [420, 500], atol=1e-4)

  def test_delivery_requires_pickup(self):
    env = self.environment(start=[1460, 500])
    obs, _, done, _ = env.step(np.array([1, 0, 0, 0], np.float32))
    self.assertFalse(done)
    self.assertEqual(float(obs['is_success']), 0.)
    self.assertEqual(float(obs['carrying_supply']), 0.)

  def test_delivery_after_pickup_requires_no_extra_action(self):
    env = self.environment()
    env.step(np.array([1, 0, 0, 0], np.float32))
    env.env.state = UAVState(1360, 500, 40, 0)
    obs, reward, done, info = env.step(np.array([1, 0, 0, 0], np.float32))
    self.assertTrue(done)
    self.assertEqual(reward, 1.)
    self.assertEqual(float(obs['is_success']), 1.)
    self.assertEqual(float(info['discount']), 0.)
    np.testing.assert_array_equal(env.env.supply_position, [1400, 500])

  def test_same_step_delivery_uses_current_cargo_not_relay_position(self):
    for final_x, success in ((300, True), (600, False)):
      with self.subTest(final_x=final_x):
        env = self.environment(final_goal=[final_x, 500])
        obs, _, done, _ = env.step(np.array([1, 0, 0, 0], np.float32))
        self.assertEqual(bool(obs['is_success']), success)
        self.assertEqual(done, success)
        np.testing.assert_array_equal(obs['achieved_goal'], [400, 500])

  def test_out_of_bounds_prevents_pickup(self):
    env = self.environment(start=[1980, 500], relay_goal=[1990, 500])
    obs, reward, done, info = env.step(np.array([1, 0, 0, 0], np.float32))
    self.assertTrue(done and info['out_of_bounds'])
    self.assertEqual(reward, -1.)
    self.assertEqual(float(obs['carrying_supply']), 0.)
    self.assertFalse(info['phase_changed'])

  def test_clipped_position_is_used_for_pickup_and_cargo(self):
    env = self.environment(start=[1980, 500], relay_goal=[1990, 500])
    env.env.boundary_mode = 'clip'
    obs, _, done, _ = env.step(np.array([1, 0, 0, 0], np.float32))
    self.assertFalse(done)
    self.assertEqual(float(obs['carrying_supply']), 1.)
    np.testing.assert_array_equal(obs['achieved_goal'], [2000, 500])

  def test_last_step_pickup_is_not_delivery(self):
    env = self.environment()
    env.env.elapsed_steps = 99
    obs, reward, done, info = env.step(np.array([1, 0, 0, 0], np.float32))
    self.assertTrue(done and info['truncated'])
    self.assertEqual(float(obs['carrying_supply']), 1.)
    self.assertEqual(float(obs['is_success']), 0.)
    self.assertEqual(float(info['discount']), 0.)
    self.assertEqual(reward, -1.)

  def test_third_action_rejected_without_advancing_environment(self):
    env = RelayNavigationEnv()
    env.reset(seed=1)
    before = env.state.copy()
    with self.assertRaises(ValueError):
      env.step((2, 0))
    self.assertEqual(env.elapsed_steps, 0)
    self.assertEqual(env.state, before)
    with self.assertRaises(ValueError):
      advance(before, 2, 0, DynamicsConfig())

  def test_multi_relay_uses_the_same_two_action_contract(self):
    env = MultiRelayNavigationEnv(num_relays=2, relay_radius=10, goal_radius=10)
    env.reset(options=dict(start=[100, 500], relay_goals=[[140, 500], [180, 500]],
                           final_goal=[220, 500], speed=40, heading=0))
    self.assertEqual(env.action_space.spaces[0].n, 2)
    for phase in (1, 2, 2):
      obs, _, done, _, info = env.step((0, 0))
      self.assertEqual(info['phase'], phase)
    self.assertTrue(done and info['is_success'])
    self.assertEqual(env.compute_reward(
        np.array([140, 500]), np.array([140, 500]),
        dict(is_her=True, her_source_phase=0, her_source_action=0)), 0.)


class ContractMigrationTest(unittest.TestCase):
  def setUp(self):
    temporary = tempfile.TemporaryDirectory()
    self.addCleanup(temporary.cleanup)
    self.root = pathlib.Path(temporary.name)
    self.config = argparse.Namespace(**dreamer.define_config())
    self.config.logdir = self.root
    self.datadir = self.root / 'episodes'

  def save_episode(self, width=4):
    tools.save_episodes(self.datadir, [dict(
        action=np.zeros([2, width], np.float32),
        reward=np.zeros(2, np.float32), discount=np.ones(2, np.float32))])

  def metadata(self, contract):
    config = dict(vars(self.config), logdir=str(self.root), action_contract=contract)
    (self.root / 'run_metadata.jsonl').write_text(json.dumps(dict(config=config)))

  def test_old_replay_contract_rejected_before_resume(self):
    self.save_episode(6)
    self.metadata('move_turn_parameters_v2')
    with self.assertRaisesRegex(ValueError, 'action_contract'):
      dreamer.validate_run_contract(self.config, self.datadir)

  def test_legacy_action_width_rejected_even_if_metadata_is_relabelled(self):
    self.save_episode(6)
    with self.assertRaisesRegex(ValueError, '4-channel'):
      dreamer.load_dataset(self.datadir, self.config)

  def test_config_cannot_override_contract_to_legacy(self):
    self.config.action_contract = 'move_turn_parameters_v2'
    with self.assertRaisesRegex(ValueError, 'move_turn_autopickup_v3'):
      dreamer.validate_run_contract(self.config, self.datadir)

  def test_current_contract_and_fresh_directory_are_accepted(self):
    dreamer.validate_run_contract(self.config, self.datadir)
    self.save_episode()
    self.metadata(uav_actions.ACTION_CONTRACT)
    dreamer.validate_run_contract(self.config, self.datadir)

  def test_checkpoint_without_replay_rejected(self):
    (self.root / 'variables.pkl').write_bytes(b'placeholder')
    with self.assertRaisesRegex(ValueError, 'without replay'):
      dreamer.validate_run_contract(self.config, self.datadir)

  def test_checkpoint_contract_and_shapes_checked_before_assignment(self):
    agent = argparse.Namespace(c=self.config, variables=[tf.Variable([1., 2.])])
    path = self.root / 'variables.pkl'
    for payload, error in (
        ([np.array([9., 9.])], 'Legacy checkpoint'),
        (dict(action_contract='move_turn_parameters_v2'), 'action_contract'),
        (dict(action_contract=uav_actions.ACTION_CONTRACT,
              observation_contract=self.config.observation_contract,
              training_contract=self.config.training_contract,
              task=self.config.task, variables=[np.array([9.])]), 'shapes')):
      with self.subTest(error=error):
        path.write_bytes(pickle.dumps(payload))
        with self.assertRaisesRegex(ValueError, error):
          dreamer.DreamerV2.load(agent, path)
        np.testing.assert_array_equal(agent.variables[0], [1., 2.])
    dreamer.DreamerV2.save(agent, path)
    agent.variables[0].assign([3., 4.])
    dreamer.DreamerV2.load(agent, path)
    np.testing.assert_array_equal(agent.variables[0], [1., 2.])


if __name__ == '__main__':
  unittest.main()
