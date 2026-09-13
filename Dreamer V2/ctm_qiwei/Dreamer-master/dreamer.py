"""DreamerV2 for the UAV parameterized-action benchmark.

Core algorithm follows the official danijar/dreamerv2 design:
- categorical RSSM with straight-through samples
- KL balancing
- vector, reward, and discount world-model heads
- latent imagination actor-critic
- mixed dynamics/REINFORCE actor gradient
- slow target critic

UAV adaptations include a two-action hybrid Actor, exact-vector policy input,
and controller demonstration supervision. Pickup is automatic in envs.py.
"""
from __future__ import annotations
import argparse, collections, datetime, functools, json, os, pathlib, pickle, subprocess, sys, time
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras import mixed_precision as prec
import models
import tools
import wrappers
import uav_actions
from run_support import RunLock, append_jsonl, read_jsonl, snapshot_checkpoint


def define_config():
  c = tools.AttrDict()
  # Runtime / benchmark. Paper-comparison defaults use Task 2.
  c.logdir = pathlib.Path('./outputs/dreamerv2_uav_relay_autopickup_v3')
  c.seed = 0
  c.task = 'uav_relay'
  c.reward_mode = 'goal_safe_v1'
  c.shaping_scale = 1.0
  c.action_contract = uav_actions.ACTION_CONTRACT
  c.observation_contract = 'uav_vector_v2'
  c.training_contract = 'bounded_replay_uniform_demo_bc_v1'
  c.steps = 5e6
  c.eval_every = 1e4
  c.checkpoint_every = 1000
  c.checkpoint_snapshot_every = 10000
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
  c.replay_rescan = 10000
  c.replay_rescan_seconds = 10.0
  c.minimum_free_gb = 2.0
  c.pretrain_checkpoint_every = 100
  # Half of replay windows are anchored on success, pickup, or terminal
  # events (classes selected uniformly).  Ordinary windows remain available
  # for background dynamics coverage.
  c.replay_priority_fraction = 0.5
  c.dataset_prefetch = 2
  c.train_every = 5
  c.train_steps = 1
  # The previous 100 updates left the model unable to predict TURN or terminal
  # events; the actor then exploited those errors and saturated immediately.
  c.pretrain = 1000
  # Successful trajectories seed rare task events and provide a supervised
  # actor warm start. This closes the sparse-reward deadlock where the world
  # model sees pickup/success but the actor cannot navigate to those states.
  c.demo_episodes = 64
  c.demo_seed_start = 30_000
  c.actor_pretrain = 3000
  c.actor_bc_scale = 5.0
  c.actor_imagination_scale = 0.1

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
  c.parameter_grad_scale = 0.1
  c.actor_discrete_ent = 3e-2
  c.actor_parameter_ent = 2e-2
  c.actor_unimix = 0.05
  c.slow_target = True
  c.slow_target_update = 100
  c.slow_target_fraction = 1.0
  c.actor_min_std = 0.1
  return c


class DreamerV2(tools.Module):
  def __init__(self, config, datadir, actspace, obspace, writer):
    validate_action_contract(config)
    self.c = config
    self.writer = writer
    self.actdim = int(actspace.shape[0])
    if not str(config.task).startswith('uav_'):
      raise ValueError('This cleaned build is intentionally UAV-only.')
    self.num_actions = uav_actions.NUM_ACTIONS
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
        'model_loss', 'vector_loss', 'reward_loss', 'discount_loss', 'kl',
        'actor_loss', 'actor_bc_loss', 'critic_loss',
        'model_grad_norm', 'actor_grad_norm',
        'critic_grad_norm', 'actor_discrete_entropy',
        'actor_parameter_abs_mean', 'actor_parameter_saturation',
        'replay_valid_fraction', 'actor_parameter_active_fraction')
    self.metrics = {
        name: tf.keras.metrics.Mean(name=name) for name in metric_names}
    self.float = prec.global_policy().compute_dtype
    self.warmup = dict(model=0, actor=0)
    self.dataset = iter(load_dataset(datadir, config))
    # Actor imitation must retain the demonstrated action prior. Reusing the
    # rare-event world-model sampler overrepresents pickup neighborhoods.
    demo_directory = pathlib.Path(datadir) / 'demonstrations'
    self.behavior_dataset = (iter(load_dataset(
        demo_directory, config, priority_fraction=0.0, seed=config.seed + 1729,
        demonstrations_only=True)) if any(demo_directory.glob('*.npz')) else None)
    self._build_model()

  def save(self, filename):
    tools.require_free_space(pathlib.Path(filename).parent, int(self.c.minimum_free_gb * 2**30))
    payload = dict(action_contract=self.c.action_contract,
                   observation_contract=self.c.observation_contract,
                   training_contract=self.c.training_contract,
                   task=self.c.task,
                   step=int(self.step.numpy()) if hasattr(self, 'step') else None,
                   warmup=getattr(self, 'warmup', dict(model=0, actor=0)),
                   variables=[variable.numpy() for variable in self.variables])
    if not all(np.isfinite(value).all() for value in payload['variables']):
      raise ValueError('Refusing to checkpoint non-finite state; previous checkpoint retained')
    filename = pathlib.Path(filename)
    if filename.exists():
      previous = filename.read_bytes()
      tools.atomic_write(filename.with_suffix('.prev.pkl'), lambda stream: stream.write(previous))
    tools.atomic_write(filename, lambda stream: pickle.dump(payload, stream))

  def load(self, filename):
    with pathlib.Path(filename).open('rb') as stream:
      payload = pickle.load(stream)
    if not isinstance(payload, dict):
      raise ValueError('Legacy checkpoint predates automatic pickup; use a fresh logdir')
    for key in ('action_contract', 'observation_contract', 'task', 'training_contract'):
      if payload.get(key) != getattr(self.c, key):
        raise ValueError(f'Checkpoint {key} mismatch; use a fresh logdir')
    variables, values = self.variables, payload.get('variables', [])
    if len(variables) != len(values) or any(
        tuple(variable.shape) != np.shape(value)
        for variable, value in zip(variables, values)):
      raise ValueError('Checkpoint variable shapes do not match the current model')
    if not all(np.isfinite(value).all() for value in values):
      raise ValueError('Checkpoint contains non-finite state')
    warmup = payload.get('warmup', {})
    if any(not isinstance(warmup.get(key), int) or warmup[key] < 0 for key in ('model', 'actor')):
      raise ValueError('Checkpoint has invalid warmup progress')
    # Validate the entire checkpoint before assigning any state.
    for variable, value in zip(variables, values):
      variable.assign(value)
    self.warmup = dict(warmup)

  def pretrain(self):
    """Resume either warmup phase without skipping partially completed work."""
    def checkpoint_progress():
      if sum(self.warmup.values()) % self.c.pretrain_checkpoint_every == 0:
        self.save(self.c.logdir / 'variables.pkl')
    while self.warmup['model'] < self.c.pretrain:
      self.train(next(self.dataset), world_model_only=True)
      self.warmup['model'] += 1
      checkpoint_progress()
    while self.behavior_dataset is not None and self.warmup['actor'] < self.c.actor_pretrain:
      self.train_behavior(next(self.behavior_dataset))
      self.warmup['actor'] += 1
      checkpoint_progress()

  def _build_model(self):
    self.encoder = models.VectorEncoder(self.c.num_units, tf.nn.elu)
    self.rssm = models.RSSM(
        self.c.rssm_stoch, self.c.rssm_deter, self.c.rssm_hidden,
        self.c.rssm_discrete, tf.nn.elu)
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
        self.encoder,self.rssm,self.vector,self.reward,self.discount],
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
      if self.should_pretrain():
        self.pretrain()
      for _ in range(self.c.train_steps):
        behavior = next(self.behavior_dataset) if self.behavior_dataset is not None else None
        self.train(next(self.dataset), behavior_data=behavior)
      if self.should_log(step): self._write_summaries()
    action, state = self.policy(obs, state, training)
    if training: self.step.assign_add(len(reset) * self.c.action_repeat)
    return action, state

  @tf.function
  def policy(self, obs, state, training):
    if state is None:
      latent = self.rssm.initial(tf.shape(obs['vector'])[0])
      action = tf.zeros([tf.shape(obs['vector'])[0], self.actdim], self.float)
    else:
      latent, action = state
    embed = self.encoder(preprocess(obs))
    latent, _ = self.rssm.obs_step(latent, action, embed)
    feat = self.rssm.get_feat(latent)
    # The exact compact state is available at deployment; do not force the
    # policy to recover it through a stochastic RSSM bottleneck.
    dist = self._actor_dist(self._actor_input(feat, obs['vector']))
    action = dist.sample() if training else dist.mode()
    return action, (latent, action)

  @tf.function
  def train(self, data, init_only=False, world_model_only=False, behavior_data=None):
    data = preprocess(data)
    data['action'] = canonicalize_action(data['action'], self.num_actions)
    valid = data.get('valid', tf.ones_like(data['reward']))
    with tf.GradientTape() as model_tape:
      embed = self.encoder(data)
      post, prior = self.rssm.observe(embed, data['action'])
      feat = self.rssm.get_feat(post)
      vector_dist = self.vector(feat)
      reward_dist = self.reward(feat)
      discount_dist = self.discount(feat)
      vector_loss = -tools.masked_mean(vector_dist.log_prob(data['vector']), valid)
      reward_loss = -tools.masked_mean(reward_dist.log_prob(data['reward']), valid)
      discount_target = self.c.discount * data['discount']
      discount_loss = -tools.masked_mean(discount_dist.log_prob(discount_target), valid)
      kl_loss, kl_value = self.rssm.kl_loss(
          post, prior, self.c.kl_balance, self.c.kl_free, False, valid=valid)
      model_loss = (self.c.vector_scale * vector_loss + reward_loss
                    + self.c.discount_scale*discount_loss + self.c.kl_scale*kl_loss)

    if world_model_only and not init_only:
      norm = self.model_opt(model_tape, model_loss)
      for name, value in dict(model_loss=model_loss,
          vector_loss=vector_loss, reward_loss=reward_loss, discount_loss=discount_loss,
          kl=kl_value, model_grad_norm=norm,
          replay_valid_fraction=tf.reduce_mean(tf.cast(valid, tf.float32))).items():
        self.metrics[name].update_state(value)
      return

    with tf.GradientTape() as actor_tape:
      actor_input, actor_feat, imag_feat, imag_action = self._imagine(post)
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
      # discrete MOVE/TURN -> REINFORCE; continuous parameters -> dynamics gradient.
      # Score each action under the state that generated it, not the successor
      # state returned by the world model.
      actor_dist = self._actor_dist(tf.stop_gradient(actor_input))
      baseline = self.critic(actor_feat).mean()
      advantage = tf.stop_gradient(returns - baseline)
      discrete_score = actor_dist.discrete_log_prob(imag_action) * advantage
      entropy_bonus = (self.c.actor_discrete_ent * actor_dist.discrete_entropy()
                       + self.c.actor_parameter_ent * actor_dist.parameter_entropy())
      dynamics_target = returns
      imagination_loss = -tools.masked_mean((
          self.c.actor_imagination_scale * (
              discrete_score + self.c.parameter_grad_scale * dynamics_target)
          + entropy_bonus), weights)
      # Successful controller trajectories are explicitly tagged in replay.
      # Clone them from posterior states while continuing to optimize imagined
      # returns everywhere. The reset row has demonstration=0 and is excluded.
      # Replay action[t+1] generated observation[t+1]. A policy target must be
      # paired with its source observation[t], not its successor.
      bc_loss = tf.constant(0., tf.float32)
      if behavior_data is not None:
        behavior = preprocess(behavior_data)
        behavior_input, behavior_action, demo_weight = align_behavior_supervision(
            behavior['vector'], canonicalize_action(behavior['action'], self.num_actions),
            behavior['demonstration'], behavior['valid'])
        replay_actor = self._actor_dist(tf.stop_gradient(behavior_input))
        bc_loss = behavior_cloning_loss(
            replay_actor, tf.stop_gradient(behavior_action), demo_weight)
      actor_loss = imagination_loss + self.c.actor_bc_scale * bc_loss

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
    for name, value in dict(model_loss=model_loss,
        vector_loss=vector_loss, reward_loss=reward_loss,
        discount_loss=discount_loss, kl=kl_value,
        actor_loss=actor_loss, actor_bc_loss=bc_loss, critic_loss=critic_loss,
        model_grad_norm=model_norm, actor_grad_norm=actor_norm,
        critic_grad_norm=critic_norm,
        replay_valid_fraction=tf.reduce_mean(tf.cast(valid, tf.float32)),
        actor_discrete_entropy=tools.masked_mean(actor_dist.discrete_entropy(), weights),
        actor_parameter_active_fraction=tools.masked_mean(actor_dist.mode_parameter_active(), weights),
        actor_parameter_abs_mean=tools.masked_mean(actor_dist.selected_parameter_abs_mean(), parameter_weights),
        actor_parameter_saturation=tools.masked_mean(tf.cast(
            actor_dist.selected_parameter_abs_mean() > 0.95, tf.float32), parameter_weights)).items():
      self.metrics[name].update_state(value)

  @tf.function
  def train_behavior(self, data):
    """Warm-start only the Actor from tagged successful replay states."""
    data = preprocess(data)
    data['action'] = canonicalize_action(data['action'], self.num_actions)
    valid = data.get('valid', tf.ones_like(data['reward']))
    with tf.GradientTape() as tape:
      behavior_input, behavior_action, demo_weight = align_behavior_supervision(
          data['vector'], data['action'],
          data.get('demonstration', tf.zeros_like(valid)), valid)
      dist = self._actor_dist(tf.stop_gradient(behavior_input))
      loss = behavior_cloning_loss(
          dist, behavior_action, demo_weight)
    norm = self.actor_opt(tape, loss)
    self.metrics['actor_bc_loss'].update_state(loss)
    self.metrics['actor_grad_norm'].update_state(norm)

  @staticmethod
  def _actor_input(feat, vector):
    del feat
    return tf.cast(vector, prec.global_policy().compute_dtype)

  def _actor_dist(self, vector):
    return self.actor(vector)

  def _imagine(self, post):
    flatten = lambda x: tf.reshape(x, [-1] + list(x.shape[2:]))
    start = {k: tf.stop_gradient(flatten(v)) for k,v in post.items()}
    state = start
    actor_inputs, actor_features, states, actions = [], [], [], []
    for _ in range(self.c.imag_horizon):
      feat = self.rssm.get_feat(state)
      predicted_vector = self.vector(feat).mean() if hasattr(self, 'vector') else None
      actor_input = DreamerV2._actor_input(
          feat, predicted_vector) if predicted_vector is not None else feat
      action = self._actor_dist(actor_input).sample()
      state = self.rssm.img_step(state, action)
      actor_inputs.append(actor_input); actor_features.append(feat)
      states.append(state); actions.append(action)
    actor_inputs = tf.stack(actor_inputs, 0)
    actor_features = tf.stack(actor_features, 0)
    states = {k: tf.stack([s[k] for s in states], 0) for k in states[0]}
    actions = tf.stack(actions, 0)
    return actor_inputs, actor_features, self.rssm.get_feat(states), actions

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
    append_jsonl(self.c.logdir / 'metrics.jsonl', {'step': step, **values})
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
  if action.shape[-1] != 2 * num_actions:
    raise ValueError('Invalid UAV action width')
  return models.canonical_action(action)


def behavior_cloning_loss(actor_dist, action, weight):
  """Clone demonstrations without boundary or rare-action pathologies.

  A squashed Gaussian likelihood has very large gradients for controller
  targets at +/-1. Instead, regress the bounded executed parameter directly.
  Samples retain their demonstrated frequency. Rare-event balancing belongs
  to world-model replay; applying it to Actor imitation distorts the action
  prior and can make TURN dominate away from demonstrations.
  """
  action = tf.cast(action, actor_dist.logits.dtype)
  weight = tf.cast(weight, tf.float32)
  k = tf.shape(actor_dist.logits)[-1]
  select, target_parameter = action[..., :k], action[..., k:]
  discrete_loss = tf.nn.softmax_cross_entropy_with_logits(
      labels=select, logits=actor_dist.logits)
  predicted_parameter = tf.tanh(actor_dist.mean_tensor)
  parameter_loss = tf.reduce_sum(
      tf.square(predicted_parameter - target_parameter)
      * select * actor_dist.parameter_mask, -1)
  per_step = tf.cast(discrete_loss + parameter_loss, tf.float32)
  return tools.masked_mean(per_step, weight)


def align_behavior_supervision(actor_input, action, demonstration, valid):
  """Pair each policy state with the next replay transition's action."""
  return (actor_input[:, :-1], action[:, 1:],
          tf.cast(demonstration[:, 1:], tf.float32)
          * tf.cast(valid[:, 1:], tf.float32))


def validate_action_contract(config):
  if config.action_contract != uav_actions.ACTION_CONTRACT:
    raise ValueError('This build requires move_turn_autopickup_v3; use a fresh logdir')
  if config.training_contract != 'bounded_replay_uniform_demo_bc_v1':
    raise ValueError('This build requires the bounded replay training contract; use a fresh logdir')


def validate_run_contract(config, datadir):
  """Reject old replay/checkpoints before collecting data or updating metadata."""
  validate_action_contract(config)
  metadata_path = pathlib.Path(config.logdir) / 'run_metadata.jsonl'
  checkpoint = pathlib.Path(config.logdir) / 'variables.pkl'
  if not tools.count_episodes(datadir)[0] and not any((pathlib.Path(datadir) / 'demonstrations').glob('*.npz')):
    if checkpoint.exists():
      raise ValueError('Checkpoint without replay cannot be resumed; use a fresh logdir')
    return
  if not metadata_path.exists():
    raise ValueError('Existing replay has no provenance; use a fresh logdir')
  records = list(read_jsonl(metadata_path))
  if not records:
    raise ValueError('Existing replay has no complete provenance; use a fresh logdir')
  previous = records[-1]['config']
  for key, fallback in [('reward_mode', 'legacy'), ('discount', .99),
                        ('shaping_scale', 0.0), ('task', None), ('time_limit', 100),
                        ('action_contract', 'legacy'),
                        ('observation_contract', 'legacy'),
                        ('training_contract', 'legacy')]:
    if previous.get(key, fallback) != getattr(config, key):
      raise ValueError(f'Cannot mix replay across {key} changes; use a fresh logdir')
  for key in ('seed', 'demo_episodes', 'demo_seed_start', 'pretrain', 'actor_pretrain',
              'actor_bc_scale', 'actor_imagination_scale', 'model_lr', 'actor_lr', 'critic_lr',
              'rssm_hidden', 'rssm_deter', 'rssm_stoch', 'rssm_discrete', 'num_units'):
    if previous.get(key) != getattr(config, key):
      raise ValueError(f'Cannot resume across {key} changes; use a fresh logdir')


def count_steps(datadir, config):
  return tools.count_episodes(datadir)[1] * config.action_repeat


def load_dataset(directory, config, priority_fraction=None, seed=None, demonstrations_only=False):
  validate_action_contract(config)
  def validate_episode(episode):
    if episode['action'].ndim != 2 or episode['action'].shape[-1] != 4:
      raise ValueError('Replay must contain 4-channel MOVE/TURN actions; use a fresh logdir')
    return episode
  fields = ('vector', 'action', 'reward', 'discount', 'demonstration')
  if not demonstrations_only:
    fields += ('phase', 'carrying_supply', 'is_success')
  pinned = None if demonstrations_only else pathlib.Path(directory) / 'demonstrations'
  episode = validate_episode(next(tools.load_episodes(
      directory, 1, capacity=1,
      keys=fields, pinned_directory=pinned)))
  episode['valid'] = np.ones(len(episode['reward']), np.float32)
  types = {k:v.dtype for k,v in episode.items()}
  shapes = {k:(None,)+v.shape[1:] for k,v in episode.items()}
  gen = lambda: (validate_episode(item) for item in tools.load_episodes(
      directory, config.replay_rescan, config.batch_length, False,
      seed=config.seed if seed is None else seed,
      capacity=None if demonstrations_only else config.replay_capacity,
      keys=fields, pinned_directory=pinned, rescan_seconds=config.replay_rescan_seconds,
      priority_fraction=(config.replay_priority_fraction if priority_fraction is None
                         else priority_fraction)))
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
  num_actions = uav_actions.NUM_ACTIONS
  selected = np.argmax(actions[:, :num_actions], -1)
  params = actions[np.arange(len(actions)), num_actions + selected]
  active_params = params[selected < 2]
  if counters is not None and 'steps' in counters:
    if prefix == 'train':
      counters['steps'] += length
    total_steps = counters['steps']
  else:
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
      f'{prefix}/parameter_action_fraction': float(np.mean(selected < 2)),
      f'{prefix}/selected_parameter_abs_mean': float(np.mean(np.abs(active_params))) if len(active_params) else 0.0,
      f'{prefix}/selected_parameter_saturation': float(np.mean(np.abs(active_params) > 0.95)) if len(active_params) else 0.0,
  }
  append_jsonl(config.logdir / 'metrics.jsonl', record)

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
    callbacks.append(lambda ep: tools.save_episodes(
        datadir, [ep], minimum_free_bytes=int(config.minimum_free_gb * 2**30)))
  callbacks.append(lambda ep: summarize_episode(
      ep, config, datadir, writer, prefix, progress, counters))
  env = wrappers.Collect(env, callbacks, config.precision)
  env = wrappers.RewardObs(env)
  return env


def write_run_metadata(config, resumed, phase='running'):
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
      'phase': phase,
      'git_commit': revision,
      'python': sys.version,
      'tensorflow': tf.__version__,
      'config': resolved,
  }
  append_jsonl(config.logdir / 'run_metadata.jsonl', record, durable=True)


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


def collect_controller_demonstrations(config, datadir):
  """Seed fresh replay with successful dynamics and actor supervision.

  These trajectories make pickup, delivery reward, phase switching, and
  termination observable to the world model, and break the Actor's sparse
  pickup/success supervision deadlock through a tagged behavior-cloning loss.
  """
  if config.demo_episodes <= 0:
    return dict(episodes=0, transitions=0)
  if tools.count_episodes(datadir)[0]:
    raise ValueError('Controller demonstrations may only seed an empty replay')
  from controller_baseline import controller_action
  from envs import DreamerV2UAVEnv
  task = str(config.task).split('_', 1)[1]
  episodes = []
  for offset in range(int(config.demo_episodes)):
    environment = DreamerV2UAVEnv(
        task, max_episode_steps=config.time_limit,
        seed=int(config.demo_seed_start) + offset,
        reward_mode=config.reward_mode, reward_discount=config.discount,
        shaping_scale=config.shaping_scale)
    observation = environment.reset()
    transitions = [dict(
        observation, action=np.zeros(2 * environment.num_actions, np.float32),
        reward=np.float32(0), discount=np.float32(1),
        demonstration=np.float32(0))]
    for _ in range(config.time_limit):
      action = controller_action(environment)
      observation, reward, done, info = environment.step(action)
      transitions.append(dict(
          observation, action=action, reward=np.float32(reward),
          discount=np.float32(info['discount']), demonstration=np.float32(1)))
      if done:
        break
    environment.close()
    if not done or not bool(observation['is_success']):
      raise RuntimeError(
          f'Controller demonstration failed for seed {config.demo_seed_start + offset}')
    episodes.append({
        key: np.asarray([transition[key] for transition in transitions])
        for key in transitions[0]})
  # The immutable demo directory is authoritative; root copies count the
  # originally collected transitions and remain available to diagnostics.
  for filename in tools.save_episodes(datadir / 'demonstrations', episodes):
    contents = filename.read_bytes()
    tools.atomic_write(datadir / filename.name,
                       lambda stream: stream.write(contents))
  result = dict(episodes=len(episodes),
                transitions=sum(len(episode['reward']) - 1 for episode in episodes))
  print('Controller replay seed: '
        f"{result['episodes']} successful episodes, {result['transitions']} transitions")
  return result


def main(config):
  config.logdir = pathlib.Path(config.logdir)
  config.logdir.mkdir(parents=True, exist_ok=True)
  with RunLock(config.logdir / '.run.lock'):
    return _run(config)


def _run(config):
  validate_action_contract(config)
  if config.reward_mode == 'goal_safe_v1' and config.action_repeat != 1:
    raise ValueError('goal_safe_v1 requires action_repeat=1 to preserve per-step reward discounting')
  if (config.demo_episodes < 0 or config.actor_pretrain < 0 or
      config.actor_bc_scale < 0 or
      config.actor_imagination_scale < 0 or
      config.replay_priority_fraction < 0 or config.replay_priority_fraction > 1):
    raise ValueError('Invalid demonstration or replay-priority configuration')
  if min(config.checkpoint_every, config.eval_every, config.pretrain_checkpoint_every,
         config.replay_rescan, config.replay_rescan_seconds, config.replay_capacity,
         config.envs, config.train_every, config.train_steps, config.batch_size) <= 0:
    raise ValueError('Training intervals, replay capacity and batch sizes must be positive')
  if config.batch_length < 2 or min(config.pretrain, config.actor_pretrain,
      config.minimum_free_gb, config.checkpoint_snapshot_every) < 0:
    raise ValueError('Invalid sequence length, warmup or disk guard configuration')
  np.random.seed(config.seed); tf.random.set_seed(config.seed)
  if config.gpu_growth:
    for gpu in tf.config.list_physical_devices('GPU'):
      tf.config.experimental.set_memory_growth(gpu, True)
  prec.set_global_policy('mixed_float16' if config.precision==16 else 'float32')
  devices=tf.config.list_physical_devices('GPU')
  print('Runtime', devices[0].name if devices else 'CPU', '/ DreamerV2 categorical RSSM')
  config.steps=int(config.steps); config.logdir=pathlib.Path(config.logdir); config.logdir.mkdir(parents=True,exist_ok=True)
  datadir = config.logdir / 'episodes'
  validate_run_contract(config, datadir)
  tools.require_free_space(config.logdir, int(config.minimum_free_gb * 2**30))
  checkpoint = config.logdir / 'variables.pkl'
  write_run_metadata(config, checkpoint.exists(), phase='initializing')
  pinned = list((datadir / 'demonstrations').glob('*.npz'))
  if pinned:
    if len(pinned) != config.demo_episodes:
      raise ValueError('Incomplete demonstration initialization; use a fresh logdir')
    for filename in pinned:
      if not (datadir / filename.name).exists():
        contents = filename.read_bytes()
        tools.atomic_write(datadir / filename.name, lambda stream: stream.write(contents))
  if not tools.count_episodes(datadir)[0] and config.demo_episodes:
    collect_controller_demonstrations(config, datadir)
  if config.demo_episodes and not any((datadir / 'demonstrations').glob('*.npz')):
    raise ValueError('Missing permanent demonstration store; use a fresh logdir')
  writer = tf.summary.create_file_writer(str(config.logdir))
  initial_steps = count_steps(datadir, config)
  counters = {'train': tools.count_episodes(datadir)[0], 'test': 0, 'steps': initial_steps}
  progress = tqdm(
      total=config.steps, initial=min(initial_steps, config.steps), unit='step',
      dynamic_ncols=True, desc='DreamerV2 UAV')
  train, test = [], []
  try:
    for index in range(config.envs):
      train.append(make_env(config, writer, 'train', datadir, True, progress, counters,
                            seed=config.seed + index))
      test.append(make_env(config, writer, 'test', datadir, False, None, counters,
                           seed=config.seed + 10_000 + index))
    step = counters['steps']
    prefill = max(0, config.prefill - step)
    tqdm.write(f'Prefill: {prefill} environment steps')
    random_agent = make_random_agent(uav_actions.NUM_ACTIONS, config.seed + 20_000)
    tools.simulate(random_agent, train, prefill / config.action_repeat)
    agent = DreamerV2(config, datadir, train[0].action_space, train[0].observation_space, writer)
    resumed = checkpoint.exists()
    if resumed:
      agent.load(checkpoint)
      agent.step.assign(counters['steps'])
      tqdm.write(f'Resumed checkpoint: {checkpoint}')
    else:
      agent.save(checkpoint)
    write_run_metadata(config, resumed)
    state = None
    step = counters['steps']
    next_eval = (step // int(config.eval_every) + 1) * int(config.eval_every)
    snapshot_every = int(config.checkpoint_snapshot_every)
    next_snapshot = (step // snapshot_every + 1) * snapshot_every if snapshot_every else float('inf')
    while step < config.steps:
      remaining = min(config.checkpoint_every, next_eval - step, config.steps - step)
      if state is not None:
        # Completed-episode overshoot is already reflected in the absolute
        # counter; do not subtract it a second time from the next interval.
        state = (0, *state[1:])
      state = tools.simulate(agent, train, remaining / config.action_repeat, state=state)
      step = counters['steps']
      agent.save(checkpoint)
      if snapshot_every and (step >= next_snapshot or step >= config.steps):
        snapshot_checkpoint(checkpoint, int(agent.step.numpy()),
                            int(config.minimum_free_gb * 2**30))
        next_snapshot = (step // snapshot_every + 1) * snapshot_every
      if step >= next_eval or step >= config.steps:
        if config.eval_episodes:
          tools.simulate(functools.partial(agent, training=False), test, episodes=config.eval_episodes)
        next_eval = (step // int(config.eval_every) + 1) * int(config.eval_every)
  finally:
    progress.close()
    for env in train + test:
      try:
        env.close()
      except Exception as error:
        tqdm.write(f'Environment cleanup failed: {error}')
    writer.flush()
    writer.close()


if __name__=='__main__':
  parser=argparse.ArgumentParser()
  for key,value in define_config().items():
    parser.add_argument(f'--{key}',type=tools.args_type(value),default=value)
  main(parser.parse_args())
