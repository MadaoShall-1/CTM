"""DreamerV2 for the UAV parameterized-action benchmark.

Core algorithm follows the official danijar/dreamerv2 design:
- categorical RSSM with straight-through samples
- KL balancing
- image, reward, and discount world-model heads
- latent imagination actor-critic
- mixed dynamics/REINFORCE actor gradient
- slow target critic

The only benchmark-specific extension is a hybrid actor distribution for the
paper's (discrete action, continuous parameter) action space. There is no
separate UAV adapter module; the environment contract lives in envs.py.
"""
from __future__ import annotations
import argparse, collections, datetime, functools, json, os, pathlib, subprocess, sys, time
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras import mixed_precision as prec
import models
import tools
import wrappers
import uav_actions


def define_config():
  c = tools.AttrDict()
  # Runtime / benchmark. Paper-comparison defaults use Task 2.
  c.logdir = pathlib.Path('./outputs/dreamerv2_uav_relay')
  c.seed = 0
  c.task = 'uav_relay'
  c.reward_mode = 'goal_safe_v1'
  c.shaping_scale = 1.0
  c.action_contract = 'move_turn_parameters_v2'
  c.steps = 5e6
  c.eval_every = 1e4
  c.log_every = 1e3
  c.envs = 1
  c.parallel = 'process'
  c.action_repeat = 1
  c.time_limit = 100
  c.prefill = 5000
  c.precision = 32
  c.gpu_growth = True
  c.log_images = False
  c.smoke = False
  c.eval_episodes = 3

  # Replay. Official V2 defaults are batch=50,length=50; length=20 is the
  # explicit UAV adaptation because paper episodes can terminate before 50.
  c.batch_size = 50
  c.batch_length = 20
  c.replay_capacity = 2_000_000
  c.dataset_prefetch = 2
  c.train_every = 5
  c.train_steps = 1
  c.pretrain = 100

  # Official DreamerV2 world-model structure/default scale.
  c.rssm_hidden = 400
  c.rssm_deter = 400
  c.rssm_stoch = 32
  c.rssm_discrete = 32
  c.cnn_depth = 48
  c.vector_units = 128
  c.vector_scale = 10.0
  c.num_units = 400
  c.model_lr = 3e-4
  c.kl_scale = 1.0
  c.kl_balance = 0.8
  c.kl_free = 0.0
  c.pred_discount = True
  c.discount_scale = 1.0
  c.grad_clip = 100.0

  # Official V2 actor-critic structure.
  c.actor_lr = 1e-4
  c.critic_lr = 1e-4
  c.discount = 0.99
  c.discount_lambda = 0.95
  c.imag_horizon = 15
  # UAV hybrid actor: discrete branch uses REINFORCE; continuous parameters use dynamics gradients.
  c.parameter_grad_scale = 1.0
  c.actor_discrete_ent = 1e-2
  c.actor_parameter_ent = 1e-3
  c.actor_unimix = 0.01
  c.slow_target = True
  c.slow_target_update = 100
  c.slow_target_fraction = 1.0
  c.actor_min_std = 0.1
  return c


class DreamerV2(tools.Module):
  def __init__(self, config, datadir, actspace, obspace, writer):
    self.c = config
    self.writer = writer
    self.actdim = int(actspace.shape[0])
    if not str(config.task).startswith('uav_'):
      raise ValueError('This cleaned build is intentionally UAV-only.')
    self.num_actions = 2 if config.task.startswith('uav_direct') else 3
    if self.actdim != 2 * self.num_actions:
      raise ValueError('UAV action vector must be [K selections, K parameters].')
    with tf.device('/CPU:0'):
      self.step = tf.Variable(count_steps(datadir, config), dtype=tf.int64)
    self.should_train = tools.Every(config.train_every)
    self.should_log = tools.Every(config.log_every)
    self.should_pretrain = tools.Once()
    if 'vector' not in obspace.spaces:
      raise ValueError('UAV observation space must expose a normalized vector.')
    self.vector_size = int(np.prod(obspace['vector'].shape))
    metric_names = (
        'model_loss', 'image_loss', 'vector_loss', 'reward_loss', 'discount_loss', 'kl',
        'actor_loss', 'critic_loss', 'model_grad_norm', 'actor_grad_norm',
        'critic_grad_norm', 'actor_discrete_entropy',
        'actor_parameter_abs_mean', 'actor_parameter_saturation',
        'replay_valid_fraction', 'actor_parameter_active_fraction')
    self.metrics = {
        name: tf.keras.metrics.Mean(name=name) for name in metric_names}
    self.float = prec.global_policy().compute_dtype
    self.dataset = iter(load_dataset(datadir, config))
    self._build_model()

  def _build_model(self):
    self.encoder = models.ConvEncoder(
        self.c.cnn_depth, tf.nn.elu, self.c.vector_units)
    self.rssm = models.RSSM(
        self.c.rssm_stoch, self.c.rssm_deter, self.c.rssm_hidden,
        self.c.rssm_discrete, tf.nn.elu)
    self.decoder = models.ConvDecoder(self.c.cnn_depth, tf.nn.elu)
    self.vector = models.DenseHead(
        (self.vector_size,), 2, self.c.num_units, 'mse', tf.nn.elu)
    self.reward = models.DenseHead((), 4, self.c.num_units, 'mse', tf.nn.elu)
    self.discount = models.DenseHead((), 4, self.c.num_units, 'binary', tf.nn.elu)
    self.actor = models.HybridActionDecoder(
        self.num_actions, 4, self.c.num_units, self.c.actor_min_std, tf.nn.elu,
        unimix=self.c.actor_unimix)
    self.critic = models.DenseHead((), 4, self.c.num_units, 'mse', tf.nn.elu)
    self.slow_critic = models.DenseHead((), 4, self.c.num_units, 'mse', tf.nn.elu)
    self.model_opt = tools.Adam('model', [
        self.encoder,self.rssm,self.decoder,self.vector,self.reward,self.discount],
                                 self.c.model_lr, clip=self.c.grad_clip)
    self.actor_opt = tools.Adam('actor', [self.actor], self.c.actor_lr, clip=min(self.c.grad_clip, 20.0))
    self.critic_opt = tools.Adam('critic', [self.critic], self.c.critic_lr, clip=self.c.grad_clip)
    # Materialize variables and initialize target critic.
    self._updates = tf.Variable(0, dtype=tf.int64, trainable=False)
    self.train(next(self.dataset), init_only=True)
    self._update_slow_target(1.0)

  def __call__(self, obs, reset, state=None, training=True):
    step = int(self.step.numpy())
    if state is not None and np.any(reset):
      mask = tf.cast(1 - reset, self.float)[:, None]
      latent, action = state
      latent = {k: v * tf.reshape(mask, [len(reset)] + [1]*(len(v.shape)-1)) for k,v in latent.items()}
      action *= mask
      state = latent, action
    if training and self.should_train(step):
      if self.should_pretrain() and int(self.model_opt._opt.iterations.numpy()) == 0:
        for _ in range(self.c.pretrain):
          self.train(next(self.dataset), world_model_only=True)
      for _ in range(self.c.train_steps): self.train(next(self.dataset))
      if self.should_log(step): self._write_summaries()
    action, state = self.policy(obs, state, training)
    if training: self.step.assign_add(len(reset) * self.c.action_repeat)
    return action, state

  @tf.function
  def policy(self, obs, state, training):
    if state is None:
      latent = self.rssm.initial(tf.shape(obs['image'])[0])
      action = tf.zeros([tf.shape(obs['image'])[0], self.actdim], self.float)
    else:
      latent, action = state
    embed = self.encoder(preprocess(obs))
    latent, _ = self.rssm.obs_step(latent, action, embed)
    feat = self.rssm.get_feat(latent)
    dist = self.actor(feat)
    action = dist.sample() if training else dist.mode()
    return action, (latent, action)

  @tf.function
  def train(self, data, init_only=False, world_model_only=False):
    data = preprocess(data)
    data['action'] = canonicalize_action(data['action'], self.num_actions)
    valid = data.get('valid', tf.ones_like(data['reward']))
    with tf.GradientTape() as model_tape:
      embed = self.encoder(data)
      post, prior = self.rssm.observe(embed, data['action'])
      feat = self.rssm.get_feat(post)
      image_dist = self.decoder(feat)
      vector_dist = self.vector(feat)
      reward_dist = self.reward(feat)
      discount_dist = self.discount(feat)
      image_loss = -tools.masked_mean(image_dist.log_prob(data['image']), valid)
      vector_loss = -tools.masked_mean(vector_dist.log_prob(data['vector']), valid)
      reward_loss = -tools.masked_mean(reward_dist.log_prob(data['reward']), valid)
      discount_target = self.c.discount * data['discount']
      discount_loss = -tools.masked_mean(discount_dist.log_prob(discount_target), valid)
      kl_loss, kl_value = self.rssm.kl_loss(
          post, prior, self.c.kl_balance, self.c.kl_free, False, valid=valid)
      model_loss = (image_loss + self.c.vector_scale * vector_loss + reward_loss
                    + self.c.discount_scale*discount_loss + self.c.kl_scale*kl_loss)

    if world_model_only and not init_only:
      norm = self.model_opt(model_tape, model_loss)
      for name, value in dict(model_loss=model_loss, image_loss=image_loss,
          vector_loss=vector_loss, reward_loss=reward_loss, discount_loss=discount_loss,
          kl=kl_value, model_grad_norm=norm,
          replay_valid_fraction=tf.reduce_mean(tf.cast(valid, tf.float32))).items():
        self.metrics[name].update_state(value)
      return

    with tf.GradientTape() as actor_tape:
      actor_feat, imag_feat, imag_action = self._imagine(post)
      reward = self.reward(imag_feat).mean()
      discount = self.discount(imag_feat).mean()
      value = self.slow_critic(imag_feat).mean()
      returns = tools.lambda_return_from_next_value(
          reward, value, discount, self.c.discount_lambda)
      # All H actions are trained at their generating states. Do not imagine
      # learning targets from replay padding or true terminal states.
      start_weight = tf.reshape(
          tf.cast(valid, discount.dtype) * tf.cast(data['discount'], discount.dtype), [1, -1])
      weights = tf.stop_gradient(tf.math.cumprod(tf.concat(
          [tf.ones_like(discount[:1]), discount[:-1]], 0), 0) * start_weight)
      # Hybrid parameterized-action gradient split:
      # discrete MOVE/TURN/CATCH -> REINFORCE; continuous parameters -> dynamics gradient.
      # Score each action under the state that generated it, not the successor
      # state returned by the world model.
      actor_dist = self.actor(tf.stop_gradient(actor_feat))
      baseline = self.critic(actor_feat).mean()
      advantage = tf.stop_gradient(returns - baseline)
      discrete_score = actor_dist.discrete_log_prob(imag_action) * advantage
      entropy_bonus = (self.c.actor_discrete_ent * actor_dist.discrete_entropy()
                       + self.c.actor_parameter_ent * actor_dist.parameter_entropy())
      dynamics_target = returns
      actor_loss = -tools.masked_mean((
          discrete_score + self.c.parameter_grad_scale * dynamics_target
          + entropy_bonus), weights)

    with tf.GradientTape() as critic_tape:
      critic_dist = self.critic(tf.stop_gradient(actor_feat))
      critic_loss = -tools.masked_mean(
          critic_dist.log_prob(tf.stop_gradient(returns)), weights)

    if init_only:
      # Materialize optimizer slots without changing randomly initialized
      # weights. This also makes checkpoint restoration structurally stable.
      self.model_opt.initialize()
      self.actor_opt.initialize()
      self.critic_opt.initialize()
      return

    model_norm = self.model_opt(model_tape, model_loss)
    actor_norm = self.actor_opt(actor_tape, actor_loss)
    critic_norm = self.critic_opt(critic_tape, critic_loss)

    self._updates.assign_add(1)
    if self.c.slow_target and tf.equal(self._updates % self.c.slow_target_update, 0):
      self._update_slow_target(self.c.slow_target_fraction)
    parameter_weights = weights * tf.cast(actor_dist.mode_parameter_active(), weights.dtype)
    for name, value in dict(model_loss=model_loss, image_loss=image_loss,
        vector_loss=vector_loss, reward_loss=reward_loss,
        discount_loss=discount_loss, kl=kl_value,
        actor_loss=actor_loss, critic_loss=critic_loss,
        model_grad_norm=model_norm, actor_grad_norm=actor_norm,
        critic_grad_norm=critic_norm,
        replay_valid_fraction=tf.reduce_mean(tf.cast(valid, tf.float32)),
        actor_discrete_entropy=tools.masked_mean(actor_dist.discrete_entropy(), weights),
        actor_parameter_active_fraction=tools.masked_mean(actor_dist.mode_parameter_active(), weights),
        actor_parameter_abs_mean=tools.masked_mean(actor_dist.selected_parameter_abs_mean(), parameter_weights),
        actor_parameter_saturation=tools.masked_mean(tf.cast(
            actor_dist.selected_parameter_abs_mean() > 0.95, tf.float32), parameter_weights)).items():
      self.metrics[name].update_state(value)

  def _imagine(self, post):
    flatten = lambda x: tf.reshape(x, [-1] + list(x.shape[2:]))
    start = {k: tf.stop_gradient(flatten(v)) for k,v in post.items()}
    state = start
    actor_features, states, actions = [], [], []
    for _ in range(self.c.imag_horizon):
      feat = self.rssm.get_feat(state)
      action = self.actor(feat).sample()
      state = self.rssm.img_step(state, action)
      actor_features.append(feat); states.append(state); actions.append(action)
    actor_features = tf.stack(actor_features, 0)
    states = {k: tf.stack([s[k] for s in states], 0) for k in states[0]}
    actions = tf.stack(actions, 0)
    return actor_features, self.rssm.get_feat(states), actions

  def _update_slow_target(self, fraction):
    # Variables are created lazily; force both critics once before this call.
    if not self.critic.variables or not self.slow_critic.variables:
      dummy = tf.zeros([1, self.c.rssm_deter + self.c.rssm_stoch*self.c.rssm_discrete], self.float)
      self.critic(dummy); self.slow_critic(dummy)
    for src, dst in zip(self.critic.variables, self.slow_critic.variables):
      dst.assign(fraction*src + (1-fraction)*dst)

  def _write_summaries(self):
    step = int(self.step.numpy())
    values = {k: float(v.result()) for k,v in self.metrics.items()}
    for v in self.metrics.values(): v.reset_state()
    with (self.c.logdir/'metrics.jsonl').open('a') as f:
      f.write(json.dumps({'step':step, **values})+'\n')
    print(f'[{step}] ' + ' / '.join(f'{k} {v:.3g}' for k,v in values.items()))


def preprocess(obs):
  obs = obs.copy()
  dtype = prec.global_policy().compute_dtype
  if 'image' in obs:
    obs['image'] = tf.cast(obs['image'], dtype) / 255.0 - 0.5
  if 'reward' in obs:
    obs['reward'] = tf.cast(obs['reward'], dtype)
  if 'discount' in obs:
    obs['discount'] = tf.cast(obs['discount'], dtype)
  if 'action' in obs:
    obs['action'] = tf.cast(obs['action'], dtype)
  return obs


def canonicalize_action(action, num_actions):
  """Match the model action to the parameterized action executed by the env."""
  action = tf.convert_to_tensor(action)
  select_raw = action[..., :num_actions]
  index = tf.argmax(select_raw, -1, output_type=tf.int32)
  select = tf.one_hot(index, num_actions, dtype=action.dtype)
  # Collect.reset() and policy initialization use all-zero sentinel actions.
  # They are not MOVE transitions and must remain identical in both paths.
  is_reset = tf.reduce_all(tf.equal(action, 0), axis=-1, keepdims=True)
  select = tf.where(is_reset, tf.zeros_like(select), select)
  params = tf.clip_by_value(action[..., num_actions:2 * num_actions], -1., 1.) * select
  params *= tf.cast(uav_actions.parameter_mask(num_actions), params.dtype)
  return tf.concat([select, params], -1)


def count_steps(datadir, config):
  return tools.count_episodes(datadir)[1] * config.action_repeat


def load_dataset(directory, config):
  episode = next(tools.load_episodes(directory, 1))
  episode['valid'] = np.ones(len(episode['reward']), np.float32)
  types = {k:v.dtype for k,v in episode.items()}
  shapes = {k:(None,)+v.shape[1:] for k,v in episode.items()}
  gen = lambda: tools.load_episodes(
      directory, config.train_steps, config.batch_length, False,
      seed=config.seed, capacity=config.replay_capacity)
  sig = {k:tf.TensorSpec(shapes[k], types[k]) for k in types}
  ds = tf.data.Dataset.from_generator(gen, output_signature=sig)
  return ds.batch(config.batch_size, drop_remainder=True).prefetch(config.dataset_prefetch)


def summarize_episode(ep, config, datadir, writer, prefix, progress=None, counters=None):
  """Record one completed episode and update the terminal progress display.

  DreamerV2 still trains from replay sequences and uses environment steps as
  its optimization budget.  This callback only makes the interaction loop
  episode-centric for monitoring, matching the UAV baseline output style.
  """
  length = int((len(ep['reward']) - 1) * config.action_repeat)
  ret = float(np.sum(ep['reward']))
  success = bool(float(np.asarray(ep.get('is_success', [0]))[-1]))
  relay_reached = bool(float(np.asarray(ep.get('relay_reached', [0]))[-1]))
  pickup = bool(np.any(np.asarray(ep.get('carrying_supply', [0]))))
  out_of_bounds = bool(float(np.asarray(ep.get('out_of_bounds', [0]))[-1]))
  terminated = bool(float(np.asarray(ep.get('terminated', [0]))[-1]))
  truncated = bool(float(np.asarray(ep.get('truncated', [0]))[-1]))
  actions = np.asarray(ep['action'])[1:]
  num_actions = 2 if str(config.task).startswith('uav_direct') else 3
  selected = np.argmax(actions[:, :num_actions], -1)
  params = actions[np.arange(len(actions)), num_actions + selected]
  active_params = params[selected < 2]
  total_steps = int(count_steps(datadir, config))

  if counters is not None:
    counters[prefix] = counters.get(prefix, 0) + 1
    episode = counters[prefix]
  else:
    episode = 0

  record = {
      'step': total_steps,
      f'{prefix}/episode': episode,
      f'{prefix}/return': ret,
      f'{prefix}/base_return': float(np.sum(ep.get('base_reward', ep['reward']))),
      f'{prefix}/shaping_return': float(np.sum(ep.get('shaping_reward', [0]))),
      f'{prefix}/discounted_return': float(np.sum(
          np.asarray(ep['reward'])[1:] * config.discount ** np.arange(len(ep['reward']) - 1))),
      f'{prefix}/length': length,
      f'{prefix}/success': float(success),
      f'{prefix}/relay_reached': float(relay_reached),
      f'{prefix}/pickup': float(pickup),
      f'{prefix}/out_of_bounds': float(out_of_bounds),
      f'{prefix}/terminated': float(terminated),
      f'{prefix}/truncated': float(truncated),
      f'{prefix}/action_move_fraction': float(np.mean(selected == 0)),
      f'{prefix}/action_turn_fraction': float(np.mean(selected == 1)),
      f'{prefix}/action_catch_fraction': (
          float(np.mean(selected == 2)) if num_actions == 3 else 0.0),
      f'{prefix}/parameter_action_fraction': float(np.mean(selected < 2)),
      f'{prefix}/selected_parameter_abs_mean': float(np.mean(np.abs(active_params))) if len(active_params) else 0.0,
      f'{prefix}/selected_parameter_saturation': float(np.mean(np.abs(active_params) > 0.95)) if len(active_params) else 0.0,
  }
  with (config.logdir / 'metrics.jsonl').open('a') as f:
    f.write(json.dumps(record) + '\n')

  # Only training episodes advance the environment-step progress bar.
  if progress is not None and prefix == 'train':
    progress.update(length)
    progress.set_postfix({
        'ep': episode,
        'ep_steps': length,
        'total_steps': total_steps,
        'success': int(success),
        'relay': int(relay_reached),
        'oob': int(out_of_bounds),
        'return': f'{ret:.0f}',
    }, refresh=True)
  else:
    tqdm.write(
        f'{prefix.title()} EP {episode:05d} | steps={length:3d} | '
        f'total={total_steps:7d} | success={int(success)} | '
        f'relay={int(relay_reached)} | '
        f'oob={int(out_of_bounds)} | return={ret:.1f}')


def make_env(
    config, writer, prefix, datadir, store, progress=None, counters=None,
    seed=None):
  ctor = functools.partial(wrappers.make_base_env, config.task, config.action_repeat,
                           config.time_limit, config.seed if seed is None else seed,
                           config.reward_mode, config.discount, config.shaping_scale)
  env = wrappers.Async(ctor, config.parallel)
  callbacks = []
  if store:
    callbacks.append(lambda ep: tools.save_episodes(datadir, [ep]))
  callbacks.append(lambda ep: summarize_episode(
      ep, config, datadir, writer, prefix, progress, counters))
  env = wrappers.Collect(env, callbacks, config.precision)
  env = wrappers.RewardObs(env)
  return env


def write_run_metadata(config, resumed):
  """Append a complete, immutable description of this invocation."""
  try:
    revision = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=pathlib.Path(__file__).parent,
        text=True, stderr=subprocess.DEVNULL).strip()
  except (OSError, subprocess.CalledProcessError):
    revision = None
  resolved = {
      key: str(value) if isinstance(value, pathlib.Path) else value
      for key, value in vars(config).items()}
  record = {
      'started_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'resumed_checkpoint': bool(resumed),
      'git_commit': revision,
      'python': sys.version,
      'tensorflow': tf.__version__,
      'config': resolved,
  }
  with (config.logdir / 'run_metadata.jsonl').open('a') as f:
    f.write(json.dumps(record, sort_keys=True) + '\n')


def make_random_agent(num_actions, seed):
  """Sample valid parameterized actions for replay prefill."""
  random = np.random.RandomState(seed)

  def agent(obs, reset, state):
    actions = np.zeros((len(reset), 2 * num_actions), np.float32)
    selected = random.randint(0, num_actions, size=len(reset))
    actions[np.arange(len(reset)), selected] = 1.0
    actions[np.arange(len(reset)), num_actions + selected] = random.uniform(
        -1.0, 1.0, size=len(reset))
    return uav_actions.canonicalize_numpy(actions, num_actions), None

  return agent


def main(config):
  if config.reward_mode == 'goal_safe_v1' and config.action_repeat != 1:
    raise ValueError('goal_safe_v1 requires action_repeat=1 to preserve per-step reward discounting')
  np.random.seed(config.seed); tf.random.set_seed(config.seed)
  if config.gpu_growth:
    for gpu in tf.config.list_physical_devices('GPU'):
      tf.config.experimental.set_memory_growth(gpu, True)
  prec.set_global_policy('mixed_float16' if config.precision==16 else 'float32')
  devices=tf.config.list_physical_devices('GPU')
  print('Runtime', devices[0].name if devices else 'CPU', '/ DreamerV2 categorical RSSM')
  config.steps=int(config.steps); config.logdir=pathlib.Path(config.logdir); config.logdir.mkdir(parents=True,exist_ok=True)
  datadir = config.logdir / 'episodes'
  metadata_path = config.logdir / 'run_metadata.jsonl'
  if count_steps(datadir, config):
    if not metadata_path.exists():
      raise ValueError('Existing replay has no reward provenance; use a fresh logdir')
    previous = json.loads(metadata_path.read_text().splitlines()[-1])['config']
    for key, fallback in [('reward_mode', 'legacy'), ('discount', .99),
                          ('shaping_scale', 0.0), ('task', None), ('time_limit', 100),
                          ('action_contract', 'legacy')]:
      if previous.get(key, fallback) != getattr(config, key):
        raise ValueError(f'Cannot mix replay across {key} changes; use a fresh logdir')
  writer = tf.summary.create_file_writer(str(config.logdir))
  initial_steps = count_steps(datadir, config)
  counters = {'train': tools.count_episodes(datadir)[0], 'test': 0}
  progress = tqdm(
      total=config.steps, initial=min(initial_steps, config.steps), unit='step',
      dynamic_ncols=True, desc='DreamerV2 UAV')
  train = [make_env(
      config, writer, 'train', datadir, True, progress, counters,
      seed=config.seed + index) for index in range(config.envs)]
  test = [make_env(
      config, writer, 'test', datadir, False, None, counters,
      seed=config.seed + 10_000 + index) for index in range(config.envs)]
  actspace = train[0].action_space
  obspace = train[0].observation_space
  step = count_steps(datadir, config)
  prefill = max(0, config.prefill - step)
  tqdm.write(f'Prefill: {prefill} environment steps')
  num_actions = 2 if str(config.task).startswith('uav_direct') else 3
  random_agent = make_random_agent(num_actions, config.seed + 20_000)
  tools.simulate(random_agent, train, prefill / config.action_repeat)
  agent = DreamerV2(config, datadir, actspace, obspace, writer)
  checkpoint = config.logdir / 'variables.pkl'
  resumed = checkpoint.exists()
  if resumed:
    agent.load(checkpoint)
    # Replay files are the source of truth if a process stopped after saving an
    # episode but before the next checkpoint write.
    agent.step.assign(count_steps(datadir, config))
    tqdm.write(f'Resumed checkpoint: {checkpoint}')
  write_run_metadata(config, resumed)
  state = None
  step = count_steps(datadir, config)
  while step < config.steps:
    remaining = min(config.eval_every, config.steps - step)
    state = tools.simulate(
        agent, train, remaining / config.action_repeat, state=state)
    step = count_steps(datadir, config)
    agent.save(checkpoint)
    if config.eval_episodes:
      tools.simulate(
          functools.partial(agent, training=False), test,
          episodes=config.eval_episodes)
  progress.close()
  writer.flush()
  for env in train + test:
    env.close()


if __name__=='__main__':
  parser=argparse.ArgumentParser()
  for key,value in define_config().items():
    parser.add_argument(f'--{key}',type=tools.args_type(value),default=value)
  main(parser.parse_args())
