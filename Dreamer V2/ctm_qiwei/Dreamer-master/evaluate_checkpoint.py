"""Evaluate a frozen run checkpoint on a fixed, explicit set of UAV seeds."""
import argparse
import json
import os
import pathlib
import sys


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--run', type=pathlib.Path, required=True)
  parser.add_argument('--episodes', type=int, default=100)
  parser.add_argument('--seed-start', type=int, default=10000)
  parser.add_argument('--output', type=pathlib.Path, required=True)
  args = parser.parse_args()
  if args.episodes <= 0:
    parser.error('episodes must be positive')
  run = args.run.resolve()
  source = run.parent / 'source'
  if not source.is_dir():
    raise ValueError('Expected the run to have a frozen source snapshot')
  sys.path.insert(0, str(source))
  os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
  import numpy as np
  import tensorflow as tf
  import dreamer
  from envs import DreamerV2UAVEnv

  config = argparse.Namespace(**json.loads(
      (run / 'run_metadata.jsonl').read_text().splitlines()[-1])['config'])
  config.logdir = run
  config.batch_size = 1  # Only materialize variables; no optimization is done.
  config.dataset_prefetch = 1
  np.random.seed(config.seed)
  tf.random.set_seed(config.seed)
  for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
  tf.keras.mixed_precision.set_global_policy(
      'mixed_float16' if config.precision == 16 else 'float32')
  task = config.task.split('_', 1)[1]
  def make_env(seed):
    return DreamerV2UAVEnv(task, seed=seed, max_episode_steps=config.time_limit,
        reward_mode=config.reward_mode, reward_discount=config.discount,
        shaping_scale=config.shaping_scale)
  env = make_env(args.seed_start)
  agent = dreamer.DreamerV2(config, run / 'episodes', env.action_space,
                           env.observation_space, None)
  env.close()
  agent.load(run / 'variables.pkl')
  records = []
  for seed in range(args.seed_start, args.seed_start + args.episodes):
    env = make_env(seed)
    obs, state = env.reset(), None
    pickup, total, discounted = False, 0., 0.
    actions, parameters = [], []
    for step in range(config.time_limit):
      batch = {key: value[None] for key, value in obs.items()}
      action, state = agent.policy(batch, state, False)
      action = action.numpy()[0]
      selected = int(np.argmax(action[:env.num_actions]))
      if selected == 2 and action[-1] != 0:
        raise AssertionError('CATCH has a nonzero parameter')
      actions.append(selected)
      if selected < 2:
        parameters.append(float(action[env.num_actions + selected]))
      obs, reward, done, info = env.step(action)
      total += reward
      discounted += config.discount ** step * reward
      pickup |= bool(obs['carrying_supply'])
      if done:
        break
    records.append(dict(seed=seed, length=step + 1, success=bool(obs['is_success']),
        pickup=pickup, out_of_bounds=bool(obs['out_of_bounds']),
        timeout=bool(obs['truncated']), episode_return=total,
        discounted_return=discounted,
        action_counts=[actions.count(i) for i in range(env.num_actions)],
        parameter_count=len(parameters),
        saturated_parameter_count=sum(abs(x) > .95 for x in parameters)))
    env.close()
    if len(records) % 10 == 0:
      print(f"Evaluated {len(records)}/{args.episodes}: "
            f"success={sum(x['success'] for x in records)}, "
            f"pickup={sum(x['pickup'] for x in records)}", flush=True)
  counts = np.sum([x['action_counts'] for x in records], axis=0)
  parameter_count = sum(x['parameter_count'] for x in records)
  result = dict(run=str(run), checkpoint_step=int(agent.step.numpy()),
      seed_start=args.seed_start, episodes=args.episodes,
      policy='mode_action_with_seeded_stochastic_RSSM',
      runtime_devices=[str(x) for x in tf.config.list_physical_devices('GPU')],
      success_rate=float(np.mean([x['success'] for x in records])),
      pickup_rate=float(np.mean([x['pickup'] for x in records])),
      out_of_bounds_rate=float(np.mean([x['out_of_bounds'] for x in records])),
      timeout_rate=float(np.mean([x['timeout'] for x in records])),
      mean_length=float(np.mean([x['length'] for x in records])),
      action_fractions=(counts / counts.sum()).tolist(),
      parameter_saturation=sum(x['saturated_parameter_count'] for x in records) / max(1, parameter_count),
      records=records)
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
  print(json.dumps({key: value for key, value in result.items() if key != 'records'}, indent=2))


if __name__ == '__main__':
  main()
