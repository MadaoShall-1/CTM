"""Fast dependency, GPU, environment, and model compatibility check."""

import argparse
import functools
import pathlib
import tempfile

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import dreamer
import models
import tools
import wrappers


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--skip-env', action='store_true')
  args = parser.parse_args()
  config = dreamer.define_config()
  print('TensorFlow:', tf.__version__)
  print('TensorFlow Probability:', tfp.__version__)
  print('GPU devices:', tf.config.list_physical_devices('GPU'))
  print('Host defaults:', {
      key: config[key] for key in (
          'precision', 'batch_size', 'replay_capacity',
          'dataset_prefetch', 'parallel')})

  if not args.skip_env:
    ctor = functools.partial(
        wrappers.make_base_env, 'dmc_cartpole_balance', 1, 100)
    env = wrappers.Async(ctor, 'process')
    try:
      obs = env.reset()
      assert obs['image'].shape == (64, 64, 3)
      obs, reward, done, info = env.step(
          np.zeros(env.action_space.shape))
      assert obs['image'].shape == (64, 64, 3)
      action_size = env.action_space.shape[0]
      print('dm_control process render: OK')
    finally:
      env.close()
  else:
    action_size = 1

  dynamics = models.RSSM(stoch=30, deter=200, hidden=200)
  state = dynamics.initial(2)
  action = tf.zeros([2, action_size], tf.float32)
  embed = tf.zeros([2, 1024], tf.float32)
  post, prior = dynamics.obs_step(state, action, embed)
  assert post['stoch'].shape == (2, 30)
  assert prior['deter'].shape == (2, 200)
  print('RSSM forward pass: OK')

  with tempfile.TemporaryDirectory() as directory:
    directory = pathlib.Path(directory)
    episode = {
        'image': np.zeros([8, 64, 64, 3], np.uint8),
        'action': np.zeros([8, action.shape[-1]], np.float32),
        'reward': np.zeros([8], np.float32),
        'discount': np.ones([8], np.float32),
    }
    tools.save_episodes(directory, [episode])
    sample = next(tools.load_episodes(
        directory, rescan=1, length=5, capacity=100))
    assert sample['image'].shape == (5, 64, 64, 3)
  print('Replay save/load: OK')
  print('SMOKE_TEST_OK')


if __name__ == '__main__':
  main()
