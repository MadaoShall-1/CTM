"""Read-only checkpoint diagnostics; generated trajectories never enter replay."""
import argparse
import hashlib
import json
import os
import pathlib
import sys

import numpy as np


LEGACY_FIELDS = ['x', 'y', 'speed', 'heading', 'distance_final', 'distance_relay',
                 'elapsed_steps', 'phase', 'achieved_x', 'achieved_y',
                 'desired_x', 'desired_y', 'phase_copy']
VECTOR_V2_FIELDS = ['x', 'y', 'speed', 'sin_heading', 'cos_heading',
                    'final_dx', 'final_dy', 'supply_dx', 'supply_dy',
                    'elapsed_steps', 'phase', 'active_dx', 'active_dy']


def sha256(path):
  digest = hashlib.sha256()
  with path.open('rb') as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
      digest.update(chunk)
  return digest.hexdigest()


def circular(value):
  return (value + np.pi) % (2 * np.pi) - np.pi


def mean(value):
  return float(np.mean(value)) if np.size(value) else None


def decode_recorded_action(action, num_actions):
  """Decode current or historical replay without a hard-coded parameter offset."""
  action = np.asarray(action)
  if action.shape != (2 * num_actions,):
    raise ValueError('Recorded action width does not match its frozen environment')
  selection = int(np.argmax(action[:num_actions]))
  return selection, action[num_actions + selection]


def errors(prediction, target, contract='legacy'):
  difference = np.asarray(prediction, np.float64) - target
  position = np.linalg.norm(difference[:, :2] * 1000, axis=-1)
  if contract == 'uav_vector_v2':
    target_heading = np.arctan2(target[:, 3], target[:, 4])
    predicted_heading = np.arctan2(prediction[:, 3], prediction[:, 4])
    heading_error = circular(predicted_heading - target_heading)
    phase_index, time_index = 10, 9
    goal_error = np.linalg.norm(difference[:, 11:13], axis=-1) * 2000
    fields = VECTOR_V2_FIELDS
  else:
    heading_error = circular(difference[:, 3] * np.pi)
    phase_index, time_index = 7, 6
    goal_error = np.abs(difference[:, 4]) * np.sqrt(2) * 1000
    fields = LEGACY_FIELDS
  phase = target[:, phase_index] > 0
  phase_prediction = prediction[:, phase_index] > 0
  return dict(n=len(target), position_mae_m=mean(position),
      position_rmse_m=float(np.sqrt(np.mean(position ** 2))),
      speed_mae=mean(np.abs(difference[:, 2]) * 20),
      heading_mae_deg=mean(np.abs(heading_error) * 180 / np.pi),
      goal_distance_mae_m=mean(goal_error),
      time_mae_steps=mean(np.abs(difference[:, time_index]) * 50),
      phase_accuracy=mean(phase == phase_prediction),
      phase1_count=int(phase.sum()), phase1_recall=mean(phase_prediction[phase]),
      phase0_false_positive_rate=mean(phase_prediction[~phase]),
      normalized_rmse_by_field=dict(zip(fields,
          np.sqrt(np.mean(difference ** 2, axis=0)).tolist())))


def summarize(rows, reward_baseline, contract='legacy'):
  if not rows:
    return {}
  data = {key: np.concatenate([row[key] for row in rows]) for key in rows[0]}
  target, prediction = data['target'], data['prediction']
  phase_index = 10 if contract == 'uav_vector_v2' else 7
  result = dict(model=errors(prediction, target, contract),
                persistence=errors(data['persistence'], target, contract))
  for label, mask in [('all', np.ones(len(target), bool)),
      ('nonterminal', data['discount'] > 0), ('terminal', data['discount'] == 0),
      ('boundary', data['boundary'] > 0), ('timeout', data['timeout'] > 0),
      ('success', data['success'] > 0), ('pickup_transition', data['pickup'] > 0)]:
    result[label] = dict(n=int(mask.sum()),
        reward_actual_mean=mean(data['reward'][mask]),
        reward_predicted_mean=mean(data['pred_reward'][mask]),
        reward_mae=mean(np.abs(data['reward'][mask] - data['pred_reward'][mask])),
        constant_reward_baseline_mae=mean(np.abs(data['reward'][mask] - reward_baseline)),
        continuation_predicted_mean=mean(data['pred_discount'][mask]),
        predicted_terminal_fraction=mean(data['pred_discount'][mask] < .5),
        phase1_prediction_fraction=mean(prediction[mask, phase_index] > 0))
  for label, mask in [('time_1_20', data['time'] <= 20), ('time_21_100', data['time'] > 20)]:
    if mask.any():
      result[label] = errors(prediction[mask], target[mask], contract)
  result['continuation_brier'] = mean((data['pred_discount'] - .99 * data['discount']) ** 2)
  return result


def sampling_weights(total, length, balance=False):
  """Expected valid inclusion count of each frame per sampled sequence."""
  if total < length:
    return np.ones(total)
  starts = total - length + 1
  weights = np.zeros(total)
  for start in range(starts):
    probability = ((length if start == starts - 1 else 1) / total
                   if balance else 1 / starts)
    weights[start:start + length] += probability
  return weights


def audit_sampling(run, length):
  names = ['terminal', 'pickup', 'carrying']
  raw = dict(valid=0, **{name: 0 for name in names})
  expected = dict(valid=0., **{name: 0. for name in names})
  balanced = expected.copy()
  paths = sorted((run / 'episodes').glob('*.npz'))
  for path in paths:
    with np.load(path) as episode:
      total = len(episode['reward'])
      labels = dict(terminal=episode['discount'] == 0,
          pickup=np.r_[False, np.diff(episode['phase']) > 0], carrying=episode['phase'] > 0)
      raw['valid'] += total - 1
      for key, mask in labels.items():
        raw[key] += int(mask[1:].sum())
      for accumulator, balance in [(expected, False), (balanced, True)]:
        weights = sampling_weights(total, length, balance)
        accumulator['valid'] += float(weights.sum())
        for key, mask in labels.items():
          accumulator[key] += float(weights[mask].sum())
  return dict(episodes=len(paths), sequence_length=length, raw_transition_counts=raw,
      interpretation='Analytical expectation on final replay, not historical logged minibatches; '
        'episodes sampled uniformly. Training-valid denominator includes reset frames.',
      expected_counts_per_pass_of_uniform_episode_selection=expected,
      raw_terminal_fraction_excluding_reset=raw['terminal'] / raw['valid'],
      raw_terminal_fraction_including_reset=raw['terminal'] / (raw['valid'] + len(paths)),
      expected_training_terminal_fraction=expected['terminal'] / expected['valid'],
      hypothetical_balance_true_terminal_fraction=balanced['terminal'] / balanced['valid'],
      expected_batch50_terminal_samples=50 * expected['terminal'] / len(paths),
      expected_batch50_pickup_transition_samples=50 * expected['pickup'] / len(paths),
      expected_training_carrying_fraction=expected['carrying'] / expected['valid'],
      full_100_step_episode=dict(number_of_windows=101-length+1,
        window_contains_terminal_probability=1/(101-length+1),
        terminal_fraction_within_sampled_steps=1/((101-length+1)*length)))


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--run', type=pathlib.Path, required=True)
  parser.add_argument('--output', type=pathlib.Path, required=True)
  parser.add_argument('--replay-limit', type=int, default=0, help='0 means all replay episodes')
  parser.add_argument('--heldout-episodes', type=int, default=32)
  parser.add_argument('--batch-size', type=int, default=4)
  parser.add_argument('--mc-samples', type=int, default=32)
  parser.add_argument('--reset-context-every', type=int, default=0,
                      help='Diagnostic ablation: reset filter context every N observations')
  parser.add_argument('--sampling-only', action='store_true',
                      help='Analyze replay inclusion probabilities without loading TensorFlow')
  args = parser.parse_args()
  if (min(args.heldout_episodes, args.batch_size, args.mc_samples) < 1 or
      min(args.replay_limit, args.reset_context_every) < 0):
    parser.error('Invalid diagnostic budget')
  run = args.run.resolve()
  source = run.parent / 'source'
  if not source.is_dir():
    raise ValueError('Frozen run source is required')
  if args.sampling_only:
    config = json.loads((run / 'run_metadata.jsonl').read_text().splitlines()[-1])['config']
    result = audit_sampling(run, config['batch_length'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(result, indent=2))
    return
  sys.path.insert(0, str(source))
  os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
  import tensorflow as tf
  import dreamer
  import envs
  from controller_baseline import controller_action
  config = argparse.Namespace(**json.loads(
      (run / 'run_metadata.jsonl').read_text().splitlines()[-1])['config'])
  observation_contract = getattr(config, 'observation_contract', 'legacy')
  assert config.task == 'uav_relay' and config.time_limit == 100
  checkpoint = run / 'variables.pkl'
  checkpoint_hash = sha256(checkpoint)
  source_hashes = {p.name: sha256(p) for p in source.glob('*.py')}
  for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
  tf.keras.mixed_precision.set_global_policy('float32')
  tf.random.set_seed(20260910)
  np.random.seed(20260910)
  config.logdir, config.batch_size, config.dataset_prefetch = run, 1, 1

  def make_env(seed):
    return envs.DreamerV2UAVEnv('relay', seed=seed,
        max_episode_steps=config.time_limit, reward_mode=config.reward_mode,
        reward_discount=config.discount, shaping_scale=config.shaping_scale)

  environment = make_env(20000)
  agent = dreamer.DreamerV2(config, run / 'episodes', environment.action_space,
                           environment.observation_space, None)
  agent.load(checkpoint)
  environment.close()

  def parameter_digest():
    digest = hashlib.sha256()
    for variable in agent.variables:
      digest.update(variable.numpy().tobytes())
    return digest.hexdigest()

  before_parameters = parameter_digest()
  before_updates = int(agent._updates.numpy())
  before_iterations = [int(opt._opt.iterations.numpy())
                       for opt in [agent.model_opt, agent.actor_opt, agent.critic_opt]]
  print(f'Loaded checkpoint at {int(agent.step.numpy())} steps; inference only.', flush=True)

  @tf.function(reduce_retracing=True)
  def filter_sequence(image, vector, action):
    embedded = agent.encoder(dreamer.preprocess(dict(image=image, vector=vector)))
    initial = agent.rssm.initial(tf.shape(action)[0])
    # action[t] leads from observation[t-1] to observation[t]; reset action[0]=0.
    def transition(previous, inputs):
      context = previous[0]
      if args.reset_context_every:
        context = tf.nest.map_structure(lambda old, reset: tf.where(
            tf.equal(inputs[2] % args.reset_context_every, 0), reset, old), context, initial)
      return agent.rssm.obs_step(context, inputs[0], inputs[1])
    post, prior = tf.scan(transition,
        (tf.transpose(action, [1, 0, 2]), tf.transpose(embedded, [1, 0, 2]),
         tf.range(tf.shape(action)[1])),
        (initial, initial))
    transpose = lambda x: tf.transpose(x, [1, 0] + list(range(2, x.shape.rank)))
    return tf.nest.map_structure(transpose, post), tf.nest.map_structure(transpose, prior)

  @tf.function(reduce_retracing=True)
  def decode(state):
    # Always give Keras a rank-2 tensor: filtering has [B,T,...] states while
    # open-loop probes have [B,...], and rank-relaxed tracing loses Dense's rank.
    leading = tf.shape(state['deter'])[:-1]
    feat = tf.concat([
        tf.reshape(state['stoch'], [-1, config.rssm_stoch * config.rssm_discrete]),
        tf.reshape(state['deter'], [-1, config.rssm_deter])], -1)
    return (tf.reshape(agent.vector(feat).mean(), tf.concat([leading, [agent.vector_size]], 0)),
            tf.reshape(agent.reward(feat).mean(), leading),
            tf.reshape(agent.discount(feat).mean(), leading))

  @tf.function(reduce_retracing=True)
  def decode_images(state):
    return agent.decoder(agent.rssm.get_feat(state)).mean()

  @tf.function(reduce_retracing=True)
  def open_loop(state, actions):
    outputs = []
    for index in range(15):
      state = agent.rssm.img_step(state, actions[:, index])
      if index in (0, 4, 14):
        outputs.append(decode(state))
    return outputs

  @tf.function(reduce_retracing=True)
  def probe_step(state, actions):
    return decode(agent.rssm.img_step(state, actions))

  # Validate the diagnostic's physical unit conversions and action alignment.
  records = []
  replay_paths = sorted((run / 'episodes').glob('*.npz'))
  if args.replay_limit:
    rng = np.random.RandomState(42)
    replay_paths = [replay_paths[i] for i in sorted(rng.choice(
        len(replay_paths), min(args.replay_limit, len(replay_paths)), replace=False))]
  keys = ['image', 'vector', 'state', 'action', 'reward', 'discount',
          'is_success', 'out_of_bounds', 'truncated', 'phase']
  for path in replay_paths:
    with np.load(path) as episode:
      record = {key: episode[key].copy() for key in keys}
    record['name'] = path.name
    records.append(record)
  replay_rewards = np.concatenate([record['reward'][1:] for record in records])
  reward_baseline = float(np.mean(replay_rewards))

  def collect(kind, seed_start):
    episodes = []
    rng = np.random.RandomState(314159)
    for seed in range(seed_start, seed_start + args.heldout_episodes):
      environment = make_env(seed)
      obs = environment.reset()
      steps = [dict(obs, action=np.zeros(2 * environment.num_actions, np.float32), reward=np.float32(0),
                    discount=np.float32(1))]
      for _ in range(config.time_limit):
        if kind == 'controller':
          action = controller_action(environment)
        else:
          selected = rng.randint(environment.num_actions)
          action = np.zeros(2 * environment.num_actions, np.float32)
          action[selected] = 1
          if selected < 2:
            action[environment.num_actions + selected] = rng.uniform(-1, 1)
        obs, reward, done, info = environment.step(action)
        steps.append(dict(obs, action=action, reward=np.float32(reward),
                          discount=info['discount']))
        if done:
          break
      episode = {key: np.asarray([step[key] for step in steps]) for key in keys}
      episode['name'] = f'{kind}_seed_{seed}'
      episodes.append(episode)
      environment.close()
    return episodes

  datasets = dict(replay=records, heldout_random=collect('random', 20000),
                  heldout_controller=collect('controller', 20000))
  result = dict(checkpoint=str(checkpoint), checkpoint_step=int(agent.step.numpy()),
      checkpoint_sha256=checkpoint_hash, frozen_source_sha256=source_hashes,
      runtime_gpu=[str(x) for x in tf.config.list_physical_devices('GPU')],
      diagnostic_seed=20260910, heldout_seed_start=20000,
      reset_context_every=args.reset_context_every,
      constant_reward_baseline=reward_baseline,
      methodology=dict(posterior='Reconstruction after observing target image and vector',
        prior='One-step prediction from previous posterior, no target observation',
        open_loop='Actual future actions, no future observations, starts every 10 steps from t=10',
        latent_sampling='One seeded stochastic sample; action probes average Monte Carlo samples',
        persistence='Last observed vector held constant; evaluated on identical targets',
        heldout_controller='Exact-state feedback controller; intentional successful OOD stress test',
        images='RMSE in 0..255 units; foreground selected from true non-background pixels',
        units='Map metres; speed simulation units (metres per step); circular heading degrees',
        phase='Decoded vector phase > 0 is treated as phase 1; not a calibrated probability',
        no_training='No train calls after loading, no save calls, generated data kept outside replay'),
      datasets={}, action_probes={})
  probe_contexts = {'motion': [], 'pickup': []}
  physical_errors = []

  def row(episode, t, pred, t0):
    return dict(prediction=pred[0], pred_reward=pred[1], pred_discount=pred[2],
        target=episode['vector'][t], persistence=episode['vector'][t0],
        reward=episode['reward'][t], discount=episode['discount'][t],
        success=episode['is_success'][t], boundary=episode['out_of_bounds'][t],
        timeout=episode['truncated'][t], time=np.asarray(t),
        pickup=(episode['phase'][t] > episode['phase'][np.maximum(t - 1, 0)]).astype(float))

  for dataset_name, episodes in datasets.items():
    stores = {key: [] for key in ['posterior', 'prior', 'open_1', 'open_5', 'open_15']}
    image_stats = {key: [] for key in ['posterior', 'prior', 'background']}
    for offset in range(0, len(episodes), args.batch_size):
      batch = episodes[offset:offset + args.batch_size]
      arrays = {}
      for key in ['image', 'vector', 'action']:
        arrays[key] = np.stack([np.pad(ep[key], [(0, 101 - len(ep[key]))] +
            [(0, 0)] * (ep[key].ndim - 1), mode='edge') for ep in batch])
      arrays['action'] = arrays['action'].astype(np.float32)
      post_tf, prior_tf = filter_sequence(arrays['image'], arrays['vector'], arrays['action'])
      post = {key: value.numpy() for key, value in post_tf.items()}
      decoded = {label: [value.numpy() for value in decode(state)]
                 for label, state in [('posterior', post_tf), ('prior', prior_tf)]}
      # Sample every 10th frame for image errors to keep this diagnostic small.
      for label, state in ([('posterior', post_tf), ('prior', prior_tf)]
                           if hasattr(agent, 'decoder') else []):
        small = {key: value[:, 10::10] for key, value in state.items()}
        recon = decode_images(small).numpy() + .5
        for b, ep in enumerate(batch):
          indices = np.arange(10, len(ep['image']), 10)
          target = ep['image'][indices].astype(np.float32) / 255
          if not len(indices):
            continue
          foreground = np.any(ep['image'][indices] != [18, 22, 30], axis=-1)
          difference = recon[b, :len(indices)] - target
          image_stats[label].append([float(np.sum(difference ** 2)), difference.size,
              float(np.sum(difference[foreground] ** 2)), int(foreground.sum()) * 3])
          if label == 'posterior':
            difference = np.asarray([18, 22, 30]) / 255 - target
            image_stats['background'].append([float(np.sum(difference ** 2)), difference.size,
                float(np.sum(difference[foreground] ** 2)), int(foreground.sum()) * 3])
      starts, action_windows, descriptions = [], [], []
      for b, episode in enumerate(batch):
        length = len(episode['reward'])
        target_indices = np.arange(1, length)
        if args.reset_context_every:
          # A prior at the first observation of a reset window has no observed
          # starting state; exclude these points from one-step comparisons.
          target_indices = target_indices[target_indices % args.reset_context_every != 0]
        for label in ['posterior', 'prior']:
          prediction = [value[b, target_indices] for value in decoded[label]]
          stores[label].append(row(episode, target_indices, prediction, target_indices - 1))
        # Known dynamics provide a read-only alignment oracle, not a learned baseline.
        for t in range(1, length):
          selection, parameter = decode_recorded_action(
              episode['action'][t], agent.num_actions)
          state = envs.UAVState(*map(float, episode['state'][t - 1, :4]))
          true_next = envs.advance(state, selection, parameter, envs.DynamicsConfig())
          delta = np.asarray([true_next.x, true_next.y, true_next.speed, true_next.heading]) - episode['state'][t, :4]
          delta[3] = circular(delta[3])
          physical_errors.append(np.abs(delta))
        for t in range(10, length - 15, 10):
          starts.append({key: value[b, t] for key, value in post.items()})
          action_windows.append(episode['action'][t + 1:t + 16])
          descriptions.append((episode, t))
        if dataset_name == 'replay' and len(probe_contexts['motion']) < 32:
          for t in range(10, length - 1):
            x, y, speed, heading = episode['state'][t, :4]
            if 80 < x < 1920 and 80 < y < 1920 and 4 < speed < 36:
              probe_contexts['motion'].append(dict(name=episode['name'], t=t,
                  state=episode['state'][t, :4].copy(),
                  latent={key: value[b, t] for key, value in post.items()}))
              break
        for t in np.flatnonzero(np.diff(episode['phase']) > 0):
          probe_contexts['pickup'].append(dict(name=episode['name'], t=int(t),
              state=episode['state'][t, :4].copy(),
              latent={key: value[b, t] for key, value in post.items()}))
      if starts:
        latent = {key: np.stack([value[key] for value in starts]) for key in starts[0]}
        predicted = [[tensor.numpy() for tensor in output] for output in
                     open_loop(latent, np.asarray(action_windows, np.float32))]
        for b, (episode, t) in enumerate(descriptions):
          for h, output in zip([1, 5, 15], predicted):
            stores[f'open_{h}'].append(row(episode, np.asarray([t + h]),
                [value[b:b + 1] for value in output], np.asarray([t])))
      print(f'{dataset_name}: checked {offset + len(batch)}/{len(episodes)} episodes', flush=True)
    summary = dict(episodes=len(episodes), transitions=sum(len(ep['reward']) - 1 for ep in episodes),
        pickup_episodes=sum(bool(np.max(ep['phase'])) for ep in episodes),
        success_episodes=sum(bool(np.max(ep['is_success'])) for ep in episodes),
        episode_names=[ep['name'] for ep in episodes],
        predictions={label: summarize(rows, reward_baseline, observation_contract)
                     for label, rows in stores.items()},
        image_errors={})
    for label, values in image_stats.items():
      if not values:
        continue
      sums = np.sum(values, axis=0)
      summary['image_errors'][label] = dict(
          rmse_255=float(np.sqrt(sums[0] / sums[1]) * 255),
          foreground_rmse_255=float(np.sqrt(sums[2] / sums[3]) * 255))
    result['datasets'][dataset_name] = summary
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')

  for kind, contexts in probe_contexts.items():
    if not contexts:
      continue
    choices = [(0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]
    labels = ['MOVE(-1)', 'MOVE(0)', 'MOVE(+1)', 'TURN(-1)', 'TURN(0)', 'TURN(+1)']
    # Old frozen-source runs remain inspectable without adding a removed
    # action to the current two-action model.
    if agent.num_actions == 3:
      choices.append((2, 0))
      labels.append('CATCH')
    actions = np.zeros((len(choices), 2 * agent.num_actions), np.float32)
    for i, (choice, parameter) in enumerate(choices):
      actions[i, choice], actions[i, agent.num_actions + choice] = 1, parameter
    vectors = []
    predictions_reward, predictions_discount = [], []
    for context in contexts:
      count = len(choices) * args.mc_samples
      state = {key: np.repeat(value[None], count, axis=0) for key, value in context['latent'].items()}
      output = [value.numpy().reshape(len(choices), args.mc_samples, -1)
                for value in probe_step(state, np.repeat(actions, args.mc_samples, axis=0))]
      vectors.append(output[0])
      predictions_reward.append(output[1].mean(axis=1)[:, 0])
      predictions_discount.append(output[2].mean(axis=1)[:, 0])
    samples = np.asarray(vectors)
    predictions = samples.mean(axis=2)
    speed_response = (predictions[:, 2, 2] - predictions[:, 0, 2]) * 20
    if observation_contract == 'uav_vector_v2':
      angles = np.arctan2(predictions[:, :, 3], predictions[:, :, 4])
      heading_response = circular(angles[:, 5] - angles[:, 3]) * 180 / np.pi
      phase_index = 10
    else:
      heading_response = circular(
          (predictions[:, 5, 3] - predictions[:, 3, 3]) * np.pi) * 180 / np.pi
      phase_index = 7
    result['action_probes'][kind] = dict(contexts=len(contexts), mc_samples=args.mc_samples,
        choices=labels,
        predicted_speed_by_action=np.mean((predictions[:, :, 2] + 1) * 20, axis=0).tolist(),
        move_plus_minus_speed_difference_mean=mean(speed_response),
        move_plus_minus_speed_difference_expected=8.0 if kind == 'motion' else None,
        move_speed_direction_correct_fraction=mean(speed_response > 0),
        move_speed_response_mc_standard_error_mean=mean(
            20 * np.sqrt(samples[:, 2, :, 2].var(axis=1) / args.mc_samples +
                         samples[:, 0, :, 2].var(axis=1) / args.mc_samples)),
        turn_plus_minus_heading_difference_deg_mean=mean(heading_response),
        turn_plus_minus_heading_difference_deg_expected=120.0,
        turn_heading_direction_correct_fraction=mean(heading_response > 0),
        turn_response_mc_standard_error_deg_mean=mean(
            180 * np.sqrt(samples[:, 5, :, 3].var(axis=1) / args.mc_samples +
                          samples[:, 3, :, 3].var(axis=1) / args.mc_samples)),
        phase1_fraction_by_action=np.mean(
            predictions[:, :, phase_index] > 0, axis=0).tolist(),
        phase_value_by_action=np.mean(
            (predictions[:, :, phase_index] + 1) / 2, axis=0).tolist(),
        reward_by_action=np.mean(predictions_reward, axis=0).tolist(),
        continuation_by_action=np.mean(predictions_discount, axis=0).tolist(),
        records=[dict(name=context['name'], t=context['t'],
            move_speed_response=float(speed_response[i]), turn_heading_response=float(heading_response[i]),
            predicted_phase_by_action=(
                (predictions[i, :, phase_index] + 1) / 2).tolist())
            for i, context in enumerate(contexts)])
    print(f'Action probes: {kind}, {len(contexts)} contexts x {len(choices)} actions x {args.mc_samples} samples', flush=True)

  after_iterations = [int(opt._opt.iterations.numpy())
                      for opt in [agent.model_opt, agent.actor_opt, agent.critic_opt]]
  assert before_parameters == parameter_digest(), 'Diagnostic changed loaded model parameters'
  assert before_iterations == after_iterations and before_updates == int(agent._updates.numpy())
  assert checkpoint_hash == sha256(checkpoint), 'Diagnostic changed checkpoint file'
  result['invariants'] = dict(parameters_unchanged=True, checkpoint_unchanged=True,
      optimizer_iterations_before=before_iterations, optimizer_iterations_after=after_iterations,
      physical_replay_alignment_max_abs_error=np.max(physical_errors, axis=0).tolist(),
      gaussian_nll_constants=dict(image=64 * 64 * 3 * .5 * np.log(2 * np.pi),
                                  vector=13 * .5 * np.log(2 * np.pi), reward=.5 * np.log(2 * np.pi)))
  args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
  print(f'World model diagnostic completed: {args.output}', flush=True)


if __name__ == '__main__':
  main()
