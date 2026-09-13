"""Shared two-action contract: MOVE/TURN each take one parameter."""
import numpy as np

NUM_ACTIONS = 2
ACTION_CONTRACT = 'move_turn_autopickup_v3'


def parameter_mask(num_actions):
  if num_actions != NUM_ACTIONS:
    raise ValueError('UAV tasks now have exactly 2 actions: MOVE and TURN')
  return np.ones(NUM_ACTIONS, np.float32)


def canonicalize_numpy(action, num_actions):
  action = np.asarray(action, np.float32)
  if action.shape[-1] != 2 * num_actions:
    raise ValueError('Invalid UAV action shape')
  select = np.eye(num_actions, dtype=np.float32)[action[..., :num_actions].argmax(-1)]
  select = np.where(np.all(action == 0, axis=-1, keepdims=True), 0., select)
  params = np.clip(action[..., num_actions:], -1., 1.) * select * parameter_mask(num_actions)
  return np.concatenate([select, params], axis=-1)
