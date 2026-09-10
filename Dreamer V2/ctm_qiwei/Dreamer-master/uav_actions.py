"""Shared parameter contract: MOVE/TURN take parameters, CATCH does not."""
import numpy as np


def parameter_mask(num_actions):
  if num_actions not in (2, 3):
    raise ValueError('UAV tasks have 2 or 3 actions')
  return np.array([1., 1., 0.][:num_actions], np.float32)


def canonicalize_numpy(action, num_actions):
  action = np.asarray(action, np.float32)
  if action.shape[-1] != 2 * num_actions:
    raise ValueError('Invalid UAV action shape')
  select = np.eye(num_actions, dtype=np.float32)[action[..., :num_actions].argmax(-1)]
  select = np.where(np.all(action == 0, axis=-1, keepdims=True), 0., select)
  params = np.clip(action[..., num_actions:], -1., 1.) * select * parameter_mask(num_actions)
  return np.concatenate([select, params], axis=-1)
