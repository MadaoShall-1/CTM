"""Dreamer interface for the HER-MPDQN UAV navigation environments.

Dreamer expects image observations, continuous Box actions, and the legacy
four-value Gym step API.  The benchmark environments use structured goals,
hybrid actions, and Gymnasium's five-value API.  This adapter performs only
those representation conversions; task dynamics and rewards stay in the
isolated reproduction package.
"""

import pathlib
import sys

try:
  import gymnasium as gym
except ImportError:  # Keep the host project's original compatibility policy.
  import gym
import numpy as np
from PIL import Image, ImageDraw


_REPRODUCTION = pathlib.Path(__file__).resolve().parent.parent / 'her_mpdqn_reproduction'
if not _REPRODUCTION.exists():
  raise ImportError(
      'Expected the UAV benchmark at '
      f'{_REPRODUCTION}. Run from the CTM workspace containing both projects.')
if str(_REPRODUCTION) not in sys.path:
  sys.path.insert(0, str(_REPRODUCTION))

from envs import DirectNavigationEnv, RelayNavigationEnv  # noqa: E402


class DreamerUAV:
  """Convert Direct/Relay Navigation to Dreamer's environment contract.

  The continuous action vector is ``[selection_1..K, parameter_1..K]``.
  Selection uses argmax and only the selected action's parameter is forwarded.
  This keeps all dimensions bounded in ``[-1, 1]`` for Dreamer's tanh policy.
  """

  TASKS = ('direct', 'relay')

  def __init__(self, task, size=(64, 64), max_episode_steps=100, seed=None):
    if task not in self.TASKS:
      raise ValueError(f'Unknown UAV task {task!r}; expected one of {self.TASKS}.')
    self._task = task
    self._size = tuple(size)
    self._seed = seed
    self._seed_on_next_reset = True
    if task == 'direct':
      self._env = DirectNavigationEnv(max_episode_steps=int(max_episode_steps))
      self._num_actions = 2
    else:
      self._env = RelayNavigationEnv(max_episode_steps=int(max_episode_steps))
      self._num_actions = 3
    self._last_obs = None

  @property
  def action_space(self):
    shape = (2 * self._num_actions,)
    return gym.spaces.Box(-1.0, 1.0, shape=shape, dtype=np.float32)

  @property
  def observation_space(self):
    base = self._env.observation_space.spaces
    spaces = {
        'image': gym.spaces.Box(
            0, 255, self._size + (3,), dtype=np.uint8),
        'state': base['observation'],
        'achieved_goal': base['achieved_goal'],
        'desired_goal': base['desired_goal'],
        'phase': gym.spaces.Box(
            0, self._env.num_phases - 1, shape=(), dtype=np.float32),
    }
    for key in (
        'is_success', 'out_of_bounds', 'terminated', 'truncated',
        'relay_reached', 'carrying_supply'):
      spaces[key] = gym.spaces.Box(0.0, 1.0, shape=(), dtype=np.float32)
    return gym.spaces.Dict(spaces)

  @property
  def current_phase(self):
    return self._env.current_phase

  @property
  def num_phases(self):
    return self._env.num_phases

  @property
  def current_goal(self):
    return self._env.current_goal

  @property
  def unwrapped(self):
    return self._env

  def decode_action(self, action):
    """Decode Dreamer's continuous vector into a benchmark hybrid action."""
    action = np.asarray(action, dtype=np.float32)
    if action.shape != self.action_space.shape:
      raise ValueError(
          f'Expected action shape {self.action_space.shape}, got {action.shape}.')
    action = np.clip(action, -1.0, 1.0)
    discrete = int(np.argmax(action[:self._num_actions]))
    parameter = np.array([action[self._num_actions + discrete]], np.float32)
    return discrete, parameter

  def reset(self):
    seed = self._seed if self._seed_on_next_reset else None
    self._seed_on_next_reset = False
    obs, info = self._env.reset(seed=seed)
    self._last_obs = obs
    info = dict(info)
    info.update(terminated=False, truncated=False)
    return self._convert_obs(obs, info)

  def step(self, action):
    hybrid_action = self.decode_action(action)
    obs, reward, terminated, truncated, info = self._env.step(hybrid_action)
    self._last_obs = obs
    done = bool(terminated or truncated)
    info = dict(info)
    info.update({
        'discount': np.array(0.0 if terminated else 1.0, np.float32),
        'terminated': bool(terminated),
        'truncated': bool(truncated),
        'action_discrete': np.int32(hybrid_action[0]),
        'action_parameter': np.float32(hybrid_action[1][0]),
    })
    return self._convert_obs(obs, info), float(reward), done, info

  def render(self, mode='rgb_array'):
    if mode != 'rgb_array':
      raise ValueError("Only render mode 'rgb_array' is supported.")
    if self._last_obs is None:
      raise RuntimeError('Call reset before render.')
    return self._render_top_down(self._last_obs)

  def close(self):
    return self._env.close()

  def _convert_obs(self, obs, info=None):
    info = info or {}
    converted = {
        'image': self._render_top_down(obs),
        'state': np.asarray(obs['observation'], np.float32),
        'achieved_goal': np.asarray(obs['achieved_goal'], np.float32),
        'desired_goal': np.asarray(obs['desired_goal'], np.float32),
        'phase': np.asarray(self._env.current_phase, np.float32),
    }
    for key in (
        'is_success', 'out_of_bounds', 'terminated', 'truncated',
        'relay_reached', 'carrying_supply'):
      converted[key] = np.asarray(float(bool(info.get(key, False))), np.float32)
    return converted

  def _render_top_down(self, obs):
    height, width = self._size
    image = Image.new('RGB', (width, height), (18, 22, 30))
    draw = ImageDraw.Draw(image)

    def pixel(point):
      point = np.asarray(point, np.float32) / self._env.map_size
      x = int(np.clip(point[0], 0, 1) * (width - 1))
      y = int((1 - np.clip(point[1], 0, 1)) * (height - 1))
      return x, y

    def marker(point, color, radius=3, outline=None):
      x, y = pixel(point)
      draw.ellipse(
          (x - radius, y - radius, x + radius, y + radius),
          fill=color, outline=outline)

    # Final goal is green; the relay/supply is amber. The active goal gets a
    # white outline so phase changes remain visually observable.
    if self._task == 'relay':
      active_relay = self._env.current_phase == 0
      marker(
          self._env.relay_goal, (225, 160, 45), radius=3,
          outline=(255, 255, 255) if active_relay else None)
      marker(
          self._env.final_goal, (50, 190, 90), radius=4,
          outline=(255, 255, 255) if not active_relay else None)
    else:
      marker(self._env.current_goal, (50, 190, 90), radius=4,
             outline=(255, 255, 255))

    # UAV position and heading.
    x, y = pixel(obs['achieved_goal'])
    heading = float(self._env.state.heading)
    length = 6
    tip = (x + int(length * np.cos(heading)),
           y - int(length * np.sin(heading)))
    draw.line((x, y, tip[0], tip[1]), fill=(80, 180, 255), width=2)
    draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(80, 180, 255))
    return np.asarray(image, dtype=np.uint8)
