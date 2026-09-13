"""DreamerV2 model components adapted from the official DreamerV2 design.

The world-model structure follows danijar/dreamerv2: categorical RSSM,
KL balancing, vector/reward/discount heads, and imagined actor-critic learning.
The only task-specific extension is HybridActionDecoder for the UAV benchmark's
parameterized action space (categorical action + continuous per-action parameter).
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as tfkl
from tensorflow_probability import distributions as tfd
from tensorflow.keras import mixed_precision as prec
import tools
import uav_actions


def canonical_action(action):
  """Canonicalize the hybrid action at the world-model boundary."""
  action = tf.convert_to_tensor(action)
  if action.shape[-1] != 2 * uav_actions.NUM_ACTIONS:
    raise ValueError('World-model actions must have 4 MOVE/TURN channels')
  num_actions = int(action.shape[-1]) // 2
  select_raw = action[..., :num_actions]
  index = tf.argmax(select_raw, -1, output_type=tf.int32)
  select = tf.one_hot(index, num_actions, dtype=action.dtype)
  is_reset = tf.reduce_all(tf.equal(action, 0), axis=-1, keepdims=True)
  select = tf.where(is_reset, tf.zeros_like(select), select)
  mask = tf.cast(uav_actions.parameter_mask(num_actions), action.dtype)
  params = tf.clip_by_value(
      action[..., num_actions:2 * num_actions], -1., 1.) * select * mask
  return tf.concat([select, params], -1)


class RSSM(tools.Module):
  """DreamerV2 categorical recurrent state-space model.

  State = deterministic GRU state + `stoch` categorical variables, each with
  `discrete` classes. Samples use the straight-through estimator.
  """

  def __init__(self, stoch=32, deter=400, hidden=400, discrete=32, act=tf.nn.elu):
    super().__init__()
    self._stoch = int(stoch)
    self._deter = int(deter)
    self._hidden = int(hidden)
    self._discrete = int(discrete)
    self._act = act
    self._cell = tfkl.GRUCell(self._deter)

  def initial(self, batch_size):
    dtype = prec.global_policy().compute_dtype
    logits = tf.zeros([batch_size, self._stoch, self._discrete], dtype)
    stoch = tf.zeros_like(logits)
    deter = tf.zeros([batch_size, self._deter], dtype)
    return dict(logits=logits, stoch=stoch, deter=deter)

  @tf.function
  def observe(self, embed, action, state=None):
    if state is None:
      state = self.initial(tf.shape(action)[0])
    embed = tf.transpose(embed, [1, 0, 2])
    action = tf.transpose(action, [1, 0, 2])
    post, prior = tools.static_scan(
        lambda prev, inputs: self.obs_step(prev[0], *inputs),
        (action, embed), (state, state))
    post = {k: tf.transpose(v, [1, 0] + list(range(2, len(v.shape)))) for k, v in post.items()}
    prior = {k: tf.transpose(v, [1, 0] + list(range(2, len(v.shape)))) for k, v in prior.items()}
    return post, prior

  @tf.function
  def imagine(self, action, state=None):
    if state is None:
      state = self.initial(tf.shape(action)[0])
    action = tf.transpose(action, [1, 0, 2])
    prior = tools.static_scan(self.img_step, action, state)
    return {k: tf.transpose(v, [1, 0] + list(range(2, len(v.shape)))) for k, v in prior.items()}

  def get_feat(self, state):
    stoch = tf.reshape(state['stoch'], tf.concat([tf.shape(state['stoch'])[:-2], [-1]], 0))
    return tf.concat([stoch, state['deter']], -1)

  def get_dist(self, state):
    return tfd.Independent(tfd.OneHotCategorical(logits=state['logits']), 1)

  def _sample(self, logits):
    # Straight-through categorical sample used by official DreamerV2.
    dist = tfd.OneHotCategorical(logits=logits)
    sample = tf.cast(dist.sample(), logits.dtype)
    probs = tf.nn.softmax(logits, -1)
    return sample + probs - tf.stop_gradient(probs)

  def _stats(self, x):
    logits = self.get('logits', tfkl.Dense, self._stoch * self._discrete)(x)
    logits = tf.reshape(logits, tf.concat([tf.shape(x)[:-1], [self._stoch, self._discrete]], 0))
    stoch = self._sample(logits)
    return dict(logits=logits, stoch=stoch)

  @tf.function
  def obs_step(self, prev_state, prev_action, embed):
    prior = self.img_step(prev_state, prev_action)
    x = tf.concat([prior['deter'], embed], -1)
    x = self.get('obs1', tfkl.Dense, self._hidden, self._act)(x)
    stats = self._stats(x)
    return dict(**stats, deter=prior['deter']), prior

  @tf.function
  def img_step(self, prev_state, prev_action):
    prev_action = canonical_action(prev_action)
    stoch = tf.reshape(prev_state['stoch'], [tf.shape(prev_state['stoch'])[0], -1])
    x = tf.concat([stoch, prev_action], -1)
    x = self.get('img1', tfkl.Dense, self._hidden, self._act)(x)
    x, deter = self._cell(x, [prev_state['deter']])
    deter = deter[0]
    x = self.get('img2', tfkl.Dense, self._hidden, self._act)(x)
    stats = self._stats(x)
    return dict(**stats, deter=deter)

  def kl_loss(self, post, prior, balance=0.8, free=0.0, forward=False, valid=None):
    """DreamerV2 KL balancing with stop-gradient on opposite sides."""
    lhs, rhs = (prior, post) if forward else (post, prior)
    # balance weights the prior-side KL gradient, regardless of KL direction.
    # With KL(post || prior), the prior is rhs, so lhs gets 1 - balance.
    mix = float(balance) if forward else (1.0 - float(balance))
    lhs_sg = {k: tf.stop_gradient(v) for k, v in lhs.items()}
    rhs_sg = {k: tf.stop_gradient(v) for k, v in rhs.items()}
    value_lhs = tfd.kl_divergence(self.get_dist(lhs), self.get_dist(rhs_sg))
    value_rhs = tfd.kl_divergence(self.get_dist(lhs_sg), self.get_dist(rhs))
    reduce = tf.reduce_mean if valid is None else lambda x: tools.masked_mean(x, valid)
    loss_lhs = tf.maximum(reduce(value_lhs), float(free))
    loss_rhs = tf.maximum(reduce(value_rhs), float(free))
    loss = mix * loss_lhs + (1.0 - mix) * loss_rhs
    value = reduce(tfd.kl_divergence(self.get_dist(post), self.get_dist(prior)))
    return loss, value


class ConvEncoder(tools.Module):
  def __init__(self, depth=48, act=tf.nn.elu, vector_units=128):
    self._act, self._depth = act, depth
    self._vector_units = int(vector_units)
  def __call__(self, obs):
    kwargs = dict(strides=2, activation=self._act)
    x = tf.reshape(obs['image'], (-1,) + tuple(obs['image'].shape[-3:]))
    x = self.get('h1', tfkl.Conv2D, 1*self._depth, 4, **kwargs)(x)
    x = self.get('h2', tfkl.Conv2D, 2*self._depth, 4, **kwargs)(x)
    x = self.get('h3', tfkl.Conv2D, 4*self._depth, 4, **kwargs)(x)
    x = self.get('h4', tfkl.Conv2D, 8*self._depth, 4, **kwargs)(x)
    shape = tf.concat([tf.shape(obs['image'])[:-3], [32*self._depth]], 0)
    image = tf.reshape(x, shape)
    if 'vector' not in obs:
      return image
    vector = tf.cast(obs['vector'], image.dtype)
    vector = self.get('vector1', tfkl.Dense, self._vector_units, self._act)(vector)
    vector = self.get('vector2', tfkl.Dense, self._vector_units, self._act)(vector)
    return tf.concat([image, vector], -1)


class VectorEncoder(tools.Module):
  """Encode the exact low-dimensional UAV control state.

  Rendering remains available for diagnostics, but raster reconstruction is
  not a useful auxiliary objective when the simulator already exposes the
  Markov state required for control.
  """

  def __init__(self, units=400, act=tf.nn.elu):
    self._units = int(units)
    self._act = act

  def __call__(self, obs):
    x = tf.cast(obs['vector'], prec.global_policy().compute_dtype)
    for index in range(2):
      x = self.get(f'h{index}', tfkl.Dense, self._units, self._act)(x)
    return x


class ConvDecoder(tools.Module):
  def __init__(self, depth=48, act=tf.nn.elu, shape=(64,64,3)):
    self._act, self._depth, self._shape = act, depth, shape
  def __call__(self, features):
    kwargs = dict(strides=2, activation=self._act)
    x = self.get('h1', tfkl.Dense, 32*self._depth)(features)
    x = tf.reshape(x, [-1,1,1,32*self._depth])
    x = self.get('h2', tfkl.Conv2DTranspose, 4*self._depth, 5, **kwargs)(x)
    x = self.get('h3', tfkl.Conv2DTranspose, 2*self._depth, 5, **kwargs)(x)
    x = self.get('h4', tfkl.Conv2DTranspose, 1*self._depth, 6, **kwargs)(x)
    x = self.get('h5', tfkl.Conv2DTranspose, self._shape[-1], 6, strides=2)(x)
    mean = tf.reshape(x, tf.concat([tf.shape(features)[:-1], self._shape], 0))
    return tfd.Independent(tfd.Normal(mean, 1), len(self._shape))


class DenseHead(tools.Module):
  def __init__(self, shape, layers=4, units=400, dist='mse', act=tf.nn.elu):
    self._shape, self._layers, self._units, self._dist, self._act = shape, layers, units, dist, act
  def __call__(self, features):
    x = features
    for i in range(self._layers):
      x = self.get(f'h{i}', tfkl.Dense, self._units, self._act)(x)
    x = self.get('out', tfkl.Dense, int(np.prod(self._shape) or 1))(x)
    x = tf.reshape(x, tf.concat([tf.shape(features)[:-1], self._shape], 0))
    if self._dist == 'mse':
      return tfd.Independent(tfd.Normal(x, 1), len(self._shape))
    if self._dist == 'binary':
      return tfd.Independent(tfd.Bernoulli(logits=x), len(self._shape))
    raise NotImplementedError(self._dist)


class HybridDist:
  """Stable parameterized-action distribution for UAV control.

  The discrete branch uses a hard categorical sample trained with REINFORCE. The
  continuous branch uses a reparameterized Gaussian followed by tanh.
  Entropy includes the tanh Jacobian so it penalizes saturation of executed
  actions, rather than rewarding spread only in the unbounded Gaussian.
  """
  def __init__(self, logits, mean, std):
    self.logits = logits
    self.mean_tensor = mean
    self.std_tensor = std
    self.parameter_mask = tf.cast(uav_actions.parameter_mask(int(logits.shape[-1])), mean.dtype)
    self._cat = tfd.Categorical(logits=logits)
    self._normal = tfd.Normal(mean, std)

  def _onehot_st(self, index):
    # Hard non-reparameterized categorical sample. Discrete logits are trained
    # only by REINFORCE, not by a straight-through dynamics surrogate.
    hard = tf.one_hot(index, tf.shape(self.logits)[-1], dtype=self.logits.dtype)
    return tf.stop_gradient(hard)

  def sample(self):
    idx = self._cat.sample()
    select = self._onehot_st(idx)
    # Reparameterized continuous sample. Keep the pre-tanh value internally in
    # the graph and expose only bounded parameters to the RSSM/environment.
    raw = self.mean_tensor + self.std_tensor * tf.random.normal(
        tf.shape(self.mean_tensor), dtype=self.mean_tensor.dtype)
    # Canonical action contract: parameters of unselected branches are exactly
    # zero. The environment and RSSM therefore receive the same effective
    # action and the actor cannot exploit parameters that the environment
    # ignores.
    params = tf.tanh(raw) * select * self.parameter_mask
    return tf.concat([select, params], -1)

  def mode(self):
    idx = tf.argmax(self.logits, -1, output_type=tf.int32)
    select = tf.one_hot(idx, tf.shape(self.logits)[-1], dtype=self.logits.dtype)
    params = tf.tanh(self.mean_tensor) * select * self.parameter_mask
    return tf.concat([select, params], -1)

  def discrete_entropy(self):
    return self._cat.entropy()

  def selected_parameter_abs_mean(self):
    idx = tf.argmax(self.logits, -1, output_type=tf.int32)
    onehot = tf.one_hot(idx, tf.shape(self.logits)[-1], dtype=self.logits.dtype)
    selected = tf.reduce_sum(tf.abs(tf.tanh(self.mean_tensor)) * onehot * self.parameter_mask, -1)
    return selected

  def mode_parameter_active(self):
    return tf.gather(self.parameter_mask, tf.argmax(self.logits, -1, output_type=tf.int32))

  def parameter_entropy(self):
    # H(A, X_A) = H(A) + sum_a p(a) H(tanh(X_a)). Estimate the
    # transformation term in pre-tanh space for finite saturation gradients.
    raw = self._normal.sample()
    log_jacobian = 2.0 * (tf.math.log(tf.cast(2.0, raw.dtype))
                          - raw - tf.nn.softplus(-2.0 * raw))
    entropies = self._normal.entropy() + log_jacobian
    return tf.reduce_sum(entropies * self._cat.probs_parameter() * self.parameter_mask, -1)

  def entropy(self):
    return self.discrete_entropy() + self.parameter_entropy()

  def discrete_log_prob(self, action):
    """Log-probability of only the discrete action branch."""
    k = tf.shape(self.logits)[-1]
    select = action[..., :k]
    idx = tf.argmax(tf.stop_gradient(select), -1, output_type=tf.int32)
    return self._cat.log_prob(idx)

  def log_prob(self, action):
    k = tf.shape(self.logits)[-1]
    select, params = action[..., :k], action[..., k:]
    idx = tf.argmax(select, -1, output_type=tf.int32)
    onehot = tf.one_hot(idx, k, dtype=params.dtype)

    # Stop gradients through the sampled action
    # value for the score-function term; gradients flow through distribution
    # parameters only, as required by REINFORCE.
    safe = tf.clip_by_value(tf.stop_gradient(params), -0.999, 0.999)
    raw = tf.atanh(safe)
    log_jacobian = 2.0 * (tf.math.log(tf.cast(2.0, raw.dtype))
                          - raw - tf.nn.softplus(-2.0 * raw))
    param_logprob = tf.reduce_sum(
        (self._normal.log_prob(raw) - log_jacobian) * onehot * self.parameter_mask, -1)
    return self._cat.log_prob(idx) + param_logprob


class HybridActionDecoder(tools.Module):
  """DreamerV2 actor extended to categorical + action-conditioned parameter."""
  def __init__(self, num_actions, layers=4, units=400, min_std=0.1, act=tf.nn.elu,
               unimix=0.01):
    self._num_actions = int(num_actions)
    self._layers, self._units, self._min_std, self._act = layers, units, min_std, act
    if not 0 <= unimix < 1 or not 0 < min_std < 1.5:
      raise ValueError('Require 0 <= unimix < 1 and 0 < min_std < 1.5')
    self._unimix = float(unimix)

  def __call__(self, features):
    x = features
    for i in range(self._layers):
      x = self.get(f'h{i}', tfkl.Dense, self._units, self._act)(x)
    logits = self.get('logits', tfkl.Dense, self._num_actions)(x)
    if self._unimix:
      probs = (1.0 - self._unimix) * tf.nn.softmax(logits, -1)
      probs += self._unimix / self._num_actions
      logits = tf.math.log(probs)
    stats = self.get('params', tfkl.Dense, 2 * self._num_actions)(x)
    mean, raw_std = tf.split(stats, 2, -1)

    # Only squash the executed action, not the Gaussian mean as well.
    # Smoothly bound std: hard clipping creates exactly zero gradients.
    std = self._min_std + (1.5 - self._min_std) * tf.nn.sigmoid(raw_std)
    return HybridDist(logits, mean, std)
