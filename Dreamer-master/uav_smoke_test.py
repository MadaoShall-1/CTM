"""Smoke test for the Dreamer <-> UAV benchmark interface."""

import functools
import pathlib
import tempfile

import numpy as np
import tensorflow as tf

import models
import tools
import wrappers
import dreamer
from uav_adapter import DreamerUAV


def check_action_codec():
  direct = DreamerUAV('direct', seed=7)
  direct.reset()
  action = np.array([-0.5, 0.8, -1.0, 0.25], np.float32)
  discrete, parameter = direct.decode_action(action)
  assert discrete == 1
  np.testing.assert_allclose(parameter, [0.25])

  relay = DreamerUAV('relay', seed=7)
  relay.reset()
  action = np.array([-1.0, -0.5, 0.9, 0.1, 0.2, -0.7], np.float32)
  discrete, parameter = relay.decode_action(action)
  assert discrete == 2
  np.testing.assert_allclose(parameter, [-0.7])
  print('Hybrid action codec: OK')


def collect_episode(task):
  episodes = []
  ctor = functools.partial(wrappers.make_base_env, task, 1, 12)
  env = wrappers.Async(ctor, 'process')
  env = wrappers.Collect(env, callbacks=[episodes.append])
  env = wrappers.RewardObs(env)
  try:
    obs = env.reset()
    assert obs['image'].shape == (64, 64, 3)
    assert obs['image'].dtype == np.uint8
    assert obs['state'].ndim == 1
    assert obs['achieved_goal'].shape == (2,)
    assert obs['desired_goal'].shape == (2,)
    assert obs['phase'].shape == ()
    assert obs['is_success'].shape == ()
    assert obs['terminated'].shape == ()
    assert obs['truncated'].shape == ()
    obs_space = env.observation_space
    if not obs_space.contains(obs):
      details = {
          key: {
              'shape': np.asarray(obs[key]).shape,
              'dtype': np.asarray(obs[key]).dtype,
              'in_space': space.contains(obs[key]),
          }
          for key, space in obs_space.spaces.items()
      }
      raise AssertionError(f'Observation does not match declared space: {details}')

    done = False
    steps = 0
    while not done:
      obs, reward, done, info = env.step(env.action_space.sample())
      assert reward in (-1.0, 0.0)
      steps += 1
      assert steps <= 12
    assert len(episodes) == 1
    episode = episodes[0]
    assert episode['image'].shape == (steps + 1, 64, 64, 3)
    assert episode['action'].shape == (steps + 1, env.action_space.shape[0])
    assert episode['phase'].shape == (steps + 1,)
    assert episode['is_success'].shape == (steps + 1,)
    assert episode['out_of_bounds'].shape == (steps + 1,)
    assert episode['truncated'].shape == (steps + 1,)
    assert episode['reward'].shape == (steps + 1,)
    return episode, env.action_space.shape[0]
  finally:
    env.close()


def check_model_path(episode, action_size):
  images = tf.convert_to_tensor(episode['image'][:2], tf.float32) / 255.0 - 0.5
  encoder = models.ConvEncoder(32, tf.nn.relu)
  embed = encoder({'image': images})
  assert embed.shape == (2, 1024)

  dynamics = models.RSSM(stoch=30, deter=200, hidden=200)
  state = dynamics.initial(2)
  action = tf.zeros([2, action_size], tf.float32)
  post, prior = dynamics.obs_step(state, action, embed)
  assert post['stoch'].shape == (2, 30)
  assert prior['deter'].shape == (2, 200)


def check_metrics(episode, task):
  metrics = dreamer.compute_uav_episode_metrics(episode, task)
  required = {
      'success', 'out_of_bounds', 'terminated', 'truncated',
      'relay_reached', 'phase_switch_step', 'action_move_fraction',
      'action_turn_fraction', 'selected_parameter_abs_mean'}
  assert required.issubset(metrics)
  if task == 'uav_relay':
    assert 'action_catch_fraction' in metrics
  action_fraction = sum(
      value for key, value in metrics.items()
      if key.startswith('action_') and key.endswith('_fraction'))
  np.testing.assert_allclose(action_fraction, 1.0)


def check_replay(episodes):
  with tempfile.TemporaryDirectory() as directory:
    directory = pathlib.Path(directory)
    tools.save_episodes(directory, episodes)
    for expected in episodes:
      sample = next(tools.load_episodes(
          directory, rescan=1, length=5, capacity=100))
      assert sample['image'].shape == (5, 64, 64, 3)
      assert sample['action'].ndim == 2
      assert expected['image'].dtype == np.uint8


def main():
  print('TensorFlow:', tf.__version__)
  check_action_codec()
  episodes = []
  for task in ('uav_direct', 'uav_relay'):
    episode, action_size = collect_episode(task)
    check_model_path(episode, action_size)
    check_metrics(episode, task)
    episodes.append(episode)
    print(f'{task}: environment, collection, and RSSM path OK')
  check_replay(episodes)
  print('UAV replay save/load: OK')
  print('UAV_SMOKE_TEST_OK')


if __name__ == '__main__':
  main()
