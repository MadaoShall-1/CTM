"""Feedback-controller task sanity check; this is not a learned Dreamer policy."""
import argparse
import json
import math
import pathlib

import numpy as np

from envs import DreamerV2UAVEnv, MOVE, TURN, wrap_angle


def controller_action(env):
  """Navigate to the active goal; the environment performs automatic pickup."""
  base = env.unwrapped
  state = base.state
  delta = base.current_goal - state.position
  distance = float(np.linalg.norm(delta))
  radius = base.relay_radius if env.task == 'relay' and base.current_phase == 0 else base.goal_radius
  error = wrap_angle(math.atan2(delta[1], delta[0]) - state.heading)
  acceleration = base.dynamics.max_acceleration * base.dynamics.dt
  target_speed = min(base.dynamics.max_speed,
      math.sqrt(2 * base.dynamics.max_acceleration * max(0., distance - .6 * radius)))
  if abs(error) > .5 and state.speed > 8:
    selection, parameter = MOVE, -1.
  elif abs(error) > .08:
    selection, parameter = TURN, np.clip(error / base.dynamics.max_turn_angle, -1., 1.)
  else:
    selection, parameter = MOVE, np.clip((target_speed - state.speed) / acceleration, -1., 1.)
  action = np.zeros(2 * env.num_actions, np.float32)
  action[selection] = 1.
  action[env.num_actions + selection] = parameter
  return action


def evaluate(task='relay', seeds=range(10000, 10100)):
  records = []
  for seed in seeds:
    env = DreamerV2UAVEnv(task, seed=int(seed))
    env.reset()
    total, discounted, pickup = 0., 0., False
    for step in range(env.unwrapped.max_episode_steps):
      obs, reward, done, info = env.step(controller_action(env))
      total += reward
      discounted += env.reward_discount ** step * reward
      pickup |= bool(obs['carrying_supply'])
      if done:
        break
    records.append(dict(seed=int(seed), length=step+1, success=bool(obs['is_success']),
                        pickup=pickup, out_of_bounds=bool(obs['out_of_bounds']),
                        timeout=bool(obs['truncated']), episode_return=total,
                        discounted_return=discounted))
    env.close()
  return dict(controller='exact_state_feedback', task=task, episodes=len(records),
              success_rate=float(np.mean([x['success'] for x in records])),
              pickup_rate=float(np.mean([x['pickup'] for x in records])),
              out_of_bounds_rate=float(np.mean([x['out_of_bounds'] for x in records])),
              mean_length=float(np.mean([x['length'] for x in records])), records=records)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--task', choices=['direct', 'relay'], default='relay')
  parser.add_argument('--episodes', type=int, default=100)
  parser.add_argument('--seed-start', type=int, default=10000)
  parser.add_argument('--output', type=pathlib.Path)
  args = parser.parse_args()
  if args.episodes <= 0:
    parser.error('episodes must be positive')
  result = evaluate(args.task, range(args.seed_start, args.seed_start + args.episodes))
  if args.output:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
  print(json.dumps({k:v for k,v in result.items() if k != 'records'}, indent=2))
