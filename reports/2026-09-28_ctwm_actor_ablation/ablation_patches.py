"""Runtime patches for CT-WM ablations. The team's code is never edited.

learned_reward_std (experiment 2): the world model's reward head predicts its own standard deviation
(std = min_std + softplus(raw)) instead of using a fixed unit-variance Gaussian. The loss is still the
reward negative log-likelihood; the actor still consumes reward.mean().

delta_weight (experiment 3): an auxiliary head predicts the one-step change of the motion coordinates
(x, y, speed, sin/cos heading) from the same posterior features as the other heads. Targets are divided
by fixed per-dimension scales (DELTA_SCALE, ~7.5 m for x/y), so a few metres of error matter as much as
a large error on the absolute vector does. The loss adds delta_weight * (unit-variance Gaussian NLL) to
the team's unchanged world-model loss. Absolute reconstruction, all other heads and the actor/critic are
unchanged.

reward_norm (experiment 4): the reward head keeps its network (same layers, same initialisation, so seed
pairing holds) but predicts symlog((r - REWARD_MU) / REWARD_SIGMA) with a unit-variance Gaussian. The
constants are the mean/std of non-terminal rewards in the baseline's random prefill, so ordinary steps
become O(1) targets, while symlog keeps the rare +/-1 terminal rewards representable (~+/-6.6) without
dominating. mean() returns the reward in original units, so the team's world loss (log_prob) and actor
objective (mean) run unchanged.

reward_twohot (experiment 5): the reward head outputs logits over TWOHOT_BINS in the same normalised symlog
space as reward_norm, trained with DreamerV3's two-hot cross-entropy (danijar/dreamerv3@e3f02248693a,
embodied/jax/outs.py). One deliberate difference: DreamerV3 predicts symexp(sum_i p_i * bin_i), an average
in squashed space, which is the averaging that erased the terminal penalty in experiment 4. Here the
prediction is REWARD_MU + REWARD_SIGMA * sum_i p_i * symexp(bin_i), the expectation in original units, so a
p-weighted terminal outcome keeps its full penalty. Hidden layers match the team's reward network. The
output layer is zero-initialised (DreamerV3 practice; uniform probabilities predict exactly REWARD_MU) by
zeroing it after default creation, so the shared random stream, and hence seed pairing, is unchanged.

edge_weight (experiment 6): an auxiliary head predicts, from the same posterior features as the other heads, the
distance to each of the four map edges, as symlog(distance / one typical step). The steps are the motion head's
x/y scales (~7.4 m), so the target keeps metre precision next to an edge and proportional precision farther away.
The loss adds edge_weight * (unit-variance Gaussian NLL) on every valid row. Like the motion head, it uses
fixed-seed initialisers, so seed pairing holds.

stoch / classes (experiment 7): the compact world model's stochastic state (8 categorical variables of 8 classes)
is enlarged, e.g. to the NRSM default of 32 x 32; every other size stays compact. Layer shapes change, so initial
weights cannot be paired with earlier runs. The trainer records the resulting core config in manifest.json, and
agent_for_run() rebuilds each run from it.

demonstrations (experiment 8): DreamerV2's learning signal from dreamer.py, added to CT-WM's online training:
- the same 64 controller demonstrations (demo_episodes, demo_seed_start = 30000; every one a delivery), built in
  the CT-WM trainer's own replay format;
- the world model trains on them as if they were pinned in a full 500-episode replay: each batch slot is a
  random demonstration with probability 64 / 564;
- the actor loss gains DreamerV2's behaviour-cloning term, actor_bc_scale (5) * dreamer.behavior_cloning_loss on
  a batch of demonstrations. The actor reads the posterior features of the demonstration states, as it does when
  it acts, paired with the next action (dreamer.align_behavior_supervision).
The update is the team's OnlineAgent.update with that one term added. With the term and the mixing switched off,
it reproduces the original update exactly. Architecture and initial weights are unchanged, so seed pairing holds.
Not included: DreamerV2's 3,000-update behaviour-cloning warm start and its event-prioritised replay.

Training (run_ablation.py) and analysis scripts build agents through agent_for_run()/apply(), so a
checkpoint is always loaded into the architecture that produced it.
"""
import dataclasses
import json
import sys
from pathlib import Path

DV2 = Path(__file__).resolve().parents[2] / 'Dreamer V2' / 'ctm_qiwei' / 'Dreamer-master'
if str(DV2) not in sys.path:
    sys.path.insert(0, str(DV2))

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from tensorflow.keras import layers as tfkl  # noqa: E402
from tensorflow_probability import distributions as tfd  # noqa: E402

import models  # noqa: E402
import nrsm_online_agent  # noqa: E402
from nrsm import NRSMConfig  # noqa: E402
import tools  # noqa: E402
import validate_nrsm  # noqa: E402

REWARD_STD_KEY = 'world_model.reward_head_min_std'
REWARD_NORM_KEY = 'world_model.reward_head_normalised'
# Mean/std of the 1,018 non-terminal rewards in the same 12 random-prefill episodes as DELTA_SCALE.
REWARD_MU, REWARD_SIGMA = -0.007229, 0.001387
REWARD_TWOHOT_KEY = 'world_model.reward_head_twohot'
# 255 bins, exactly mirror-symmetric with one at 0, spanning ordinary steps (|symlog z| <= ~1.6) and terminal
# rewards (~+/-6.6).
_HALF = np.linspace(0.0, 7.5, 128)
TWOHOT_BINS = tuple(float(x) for x in np.concatenate([-_HALF[:0:-1], _HALF]))
DELTA_KEY = 'world_model.delta_head_weight'
DELTA_DIMS = (0, 1, 2, 3, 4)   # x, y, speed, sin(heading), cos(heading) of the 13-vector
# Std of one-step changes in the 12 random-prefill episodes (1,030 transitions) of the 2026-09-28
# baseline run, seed 17. Fixed constants, so the target scale never drifts during training.
DELTA_SCALE = (0.00746, 0.00729, 0.07891, 0.28910, 0.27888)
DELTA_CLIP = 10.0
EDGE_KEY = 'world_model.edge_head_weight'
# Distances to the left, right, bottom and top edges, in units of one typical step on that axis (DELTA_SCALE x, y).
EDGE_SCALE = (DELTA_SCALE[0], DELTA_SCALE[0], DELTA_SCALE[1], DELTA_SCALE[1])
EDGE_SEED = 20260929
STOCH_KEY, CLASSES_KEY = 'core_config.stoch', 'core_config.classes'
DEMO_KEY = 'training.dreamerv2_demonstrations'
# dreamer.py defaults: demo_episodes, demo_seed_start, actor_bc_scale. The world-model share treats the demos as
# pinned in the trainer's full 500-episode FIFO replay (uniform sampling).
DEMO_EPISODES, DEMO_SEED_START, BC_SCALE = 64, 30_000, 5.0
DEMO_SHARE = DEMO_EPISODES / (DEMO_EPISODES + 500)
DEMO_RNG_SEED = 20260930
_ORIGINAL_UPDATE = nrsm_online_agent.OnlineAgent.update
_DEMOS = dict(episodes=None, rng=None)
_ORIGINAL_WORLD_LOSS = nrsm_online_agent.OnlineAgent.world_loss


class LearnedStdHead(tools.Module):
    """Scalar Gaussian head that predicts a per-state standard deviation."""

    def __init__(self, layers=2, units=400, min_std=0.01, act=tf.nn.elu):
        super().__init__()
        self._layers, self._units, self._min_std, self._act = layers, units, float(min_std), act

    def __call__(self, features):
        x = features
        for index in range(self._layers):
            x = self.get(f'h{index}', tfkl.Dense, self._units, self._act)(x)
        mean, raw_std = tf.unstack(self.get('out', tfkl.Dense, 2)(x), axis=-1)
        return tfd.Independent(tfd.Normal(mean, self._min_std + tf.nn.softplus(raw_std)), 0)


class MotionHead(tools.Module):
    """models.DenseHead equivalent (ELU layers, linear output, unit-variance Gaussian) whose weights use
    fixed-seed initialisers. Building it draws nothing from the shared random stream, so every other
    initial weight (world model, actor, critic) stays identical to the paired run's."""

    def __init__(self, size, layers=2, units=400, act=tf.nn.elu, seed=20260928):
        super().__init__()
        self._size, self._layers, self._units, self._act, self._seed = size, layers, units, act, seed

    def __call__(self, features):
        glorot = lambda offset: tf.keras.initializers.GlorotUniform(seed=self._seed + offset)
        x = features
        for index in range(self._layers):
            x = self.get(f'h{index}', tfkl.Dense, self._units, self._act, kernel_initializer=glorot(index))(x)
        x = self.get('out', tfkl.Dense, self._size, kernel_initializer=glorot(99))(x)
        return tfd.Independent(tfd.Normal(x, 1), 1)


def symlog(x):
    return tf.sign(x) * tf.math.log1p(tf.abs(x))


def symexp(x):
    return tf.sign(x) * tf.math.expm1(tf.abs(x))


class NormalisedReward:
    """Minimal distribution interface used by the team's code: log_prob() for the world loss, mean() for
    imagination and evaluation. The Gaussian lives in the transformed space."""

    def __init__(self, prediction):
        self.prediction = prediction

    def log_prob(self, reward):
        target = symlog((tf.cast(reward, tf.float32) - REWARD_MU) / REWARD_SIGMA)
        return tfd.Normal(self.prediction, 1.).log_prob(target)

    def mean(self):
        return REWARD_MU + REWARD_SIGMA * symexp(self.prediction)

    mode = mean


class NormalisedRewardHead(tools.Module):
    """The team's reward network (models.DenseHead, identical layers and initialisation) with a normalised,
    symlog-transformed target."""

    def __init__(self, units, layers=2):
        super().__init__()
        self.net = models.DenseHead((), layers=layers, units=units)

    def __call__(self, features):
        return NormalisedReward(self.net(features).mean())


def symmetric_sum(probs, values):
    """sum(probs * values) over the last axis, pairing bins from the centre outwards (as in DreamerV3's
    TwoHot.pred), so symmetric bins with uniform probabilities give exactly zero."""
    m = (probs.shape[-1] - 1) // 2
    left = (probs[..., :m] * values[:m])[..., ::-1]
    return probs[..., m] * values[m] + tf.reduce_sum(left + probs[..., m + 1:] * values[m + 1:], -1)


class TwoHotReward:
    """Categorical reward over TWOHOT_BINS (normalised symlog space) with a mean in original units."""

    def __init__(self, logits):
        self.logits = logits
        self.bins = tf.constant(TWOHOT_BINS, tf.float32)

    def target_weights(self, reward):
        """Two-hot encoding of symlog((r - mu) / sigma) (clipped to the bin range) over the two nearest bins."""
        target = symlog((tf.cast(reward, tf.float32) - REWARD_MU) / REWARD_SIGMA)
        target = tf.clip_by_value(target, self.bins[0], self.bins[-1])
        count = len(TWOHOT_BINS)
        below = tf.clip_by_value(tf.reduce_sum(tf.cast(self.bins <= target[..., None], tf.int32), -1) - 1, 0, count - 1)
        above = tf.clip_by_value(count - tf.reduce_sum(tf.cast(self.bins > target[..., None], tf.int32), -1), 0, count - 1)
        equal = tf.equal(below, above)
        to_below = tf.where(equal, 1., tf.abs(tf.gather(self.bins, below) - target))
        to_above = tf.where(equal, 1., tf.abs(tf.gather(self.bins, above) - target))
        return (tf.one_hot(below, count) * (to_above / (to_below + to_above))[..., None]
                + tf.one_hot(above, count) * (to_below / (to_below + to_above))[..., None])

    def log_prob(self, reward):
        weights = tf.stop_gradient(self.target_weights(reward))
        return tf.reduce_sum(weights * tf.nn.log_softmax(self.logits, -1), -1)

    def mean(self):
        return REWARD_MU + REWARD_SIGMA * symmetric_sum(tf.nn.softmax(self.logits, -1), symexp(self.bins))

    mode = mean


class TwoHotRewardHead(tools.Module):
    """Hidden layers as models.DenseHead; a zero-initialised output layer with len(TWOHOT_BINS) logits."""

    def __init__(self, units, layers=2, act=tf.nn.elu):
        super().__init__()
        self._units, self._layers, self._act, self._zeroed = units, layers, act, False

    def __call__(self, features):
        x = features
        for index in range(self._layers):
            x = self.get(f'h{index}', tfkl.Dense, self._units, self._act)(x)
        out = self.get('out', tfkl.Dense, len(TWOHOT_BINS))
        logits = out(x)
        if not self._zeroed:   # first (eager) call: layer just created with the default initialiser
            for variable in out.weights:
                variable.assign(tf.zeros_like(variable))
            self._zeroed = True
            logits = out(x)
        return TwoHotReward(logits)


def delta_targets(vector):
    """Normalised one-step changes [B, T-1, 5] of the motion coordinates."""
    dims = list(DELTA_DIMS)
    change = tf.gather(vector[:, 1:], dims, axis=-1) - tf.gather(vector[:, :-1], dims, axis=-1)
    return tf.clip_by_value(change / tf.constant(DELTA_SCALE, tf.float32), -DELTA_CLIP, DELTA_CLIP)


def delta_mask(data):
    """Rows t >= 1 whose row t is a valid, non-first transition and whose row t-1 is valid."""
    return data['valid'][:, 1:] * (1. - data['is_first'][:, 1:]) * data['valid'][:, :-1]


def edge_distances(vector):
    """Distances [..., 4] to the left, right, bottom and top map edges, in normalised units (1 = 1000 m). The
    observation clips x/y to [-1, 1], so the edge a drone has just crossed is at distance 0."""
    x, y = vector[..., 0], vector[..., 1]
    return tf.stack([x + 1., 1. - x, y + 1., 1. - y], -1)


def edge_targets(vector):
    return symlog(edge_distances(vector) / tf.constant(EDGE_SCALE, tf.float32))


def edge_prediction(prediction):
    """Invert edge_targets: predicted distances to the four edges in normalised units."""
    return symexp(prediction) * tf.constant(EDGE_SCALE, tf.float32)


def controller_demonstrations(count=DEMO_EPISODES, seed_start=DEMO_SEED_START):
    """dreamer.collect_controller_demonstrations (the controller on seeds 30000-30063, each a delivery), in the
    CT-WM trainer's replay format (train_nrsm_online.environment / new_episode / add_transition)."""
    import train_nrsm_online as trainer
    from controller_baseline import controller_action
    episodes = []
    for seed in range(seed_start, seed_start + count):
        env = trainer.environment(seed)
        try:
            obs = env.reset()
            episode, done = trainer.new_episode(obs), False
            for t in range(1, 101):
                action = np.asarray(controller_action(env), np.float32)
                obs, reward, done, info = env.step(action)
                trainer.add_transition(episode, t, action, obs, reward, info)
                if done:
                    break
        finally:
            env.close()
        if not (done and bool(obs['is_success'])):
            raise RuntimeError(f'Controller demonstration failed for seed {seed}')
        episodes.append(episode)
    return episodes


def _mix_demonstrations(data):
    """Each world-model batch slot becomes a random demonstration with probability DEMO_SHARE."""
    arrays = {k: np.array(v) for k, v in data.items()}
    episodes, rng = _DEMOS['episodes'], _DEMOS['rng']
    for slot in range(arrays['valid'].shape[0]):
        if rng.random() < DEMO_SHARE:
            demo = episodes[rng.integers(len(episodes))]
            for key in arrays:
                arrays[key][slot] = demo[key]
    return {k: tf.constant(v) for k, v in arrays.items()}


def _demonstration_batch(size):
    episodes, rng = _DEMOS['episodes'], _DEMOS['rng']
    picks = rng.integers(len(episodes), size=size)
    return {k: tf.constant(np.stack([episodes[i][k] for i in picks])) for k in episodes[0]}


@tf.function(reduce_retracing=True)
def _update_graph(self, data, demo, world_only=False):
    """nrsm_online_agent.OnlineAgent.update, plus BC_SCALE * behaviour cloning on the demonstration batch."""
    from dreamer import align_behavior_supervision, behavior_cloning_loss
    with tf.GradientTape() as tape:
        post, loss, metrics = self.world_loss(data)
    wg, wn = self.checked_gradients(tape, loss, self.world.trainable_variables, 'world')
    if world_only:
        self.model_opt.apply_gradients(zip(wg, self.world.trainable_variables))
    else:
        starts, _ = self.select_starts(post, data)
        if BC_SCALE:   # decided when the graph is built; with BC_SCALE = 0 the graph is the original one
            # Posterior features of the demonstration states (deterministic: no random draws), paired with the
            # action that followed each state. The reset row carries no demonstrated action.
            demo_post, *_ = self.world.forward(demo, False)
            demo_feat = tf.stop_gradient(self.world.core.get_feat(demo_post))
            demonstration = demo['valid'] * (1. - demo['is_first'])
            bc_input, bc_action, bc_weight = align_behavior_supervision(
                demo_feat, models.canonical_action(demo['action']), demonstration, demo['valid'])
        with tf.GradientTape() as actor_tape:
            actor_loss, source, returns, weights, actor_metrics = self.actor_objective(starts)
            if BC_SCALE:
                bc_loss = behavior_cloning_loss(self.actor(bc_input), tf.stop_gradient(bc_action), bc_weight)
                total_actor_loss = actor_loss + BC_SCALE * bc_loss
            else:
                bc_loss, total_actor_loss = tf.constant(0.), actor_loss
        ag, an = self.checked_gradients(actor_tape, total_actor_loss, self.actor.trainable_variables, 'actor')
        with tf.GradientTape() as critic_tape:
            critic_loss = -tools.masked_mean(self.critic(tf.stop_gradient(source)).log_prob(
                tf.stop_gradient(returns)), weights)
        cg, cn = self.checked_gradients(critic_tape, critic_loss, self.critic.trainable_variables, 'critic')
        with tf.control_dependencies(wg + ag + cg):
            self.model_opt.apply_gradients(zip(wg, self.world.trainable_variables))
            self.actor_opt.apply_gradients(zip(ag, self.actor.trainable_variables))
            self.critic_opt.apply_gradients(zip(cg, self.critic.trainable_variables))
            self.ac_updates.assign_add(1)
        if tf.equal(self.ac_updates % self.c.slow_update, 0):
            self.copy_target()
        # actor_loss keeps its meaning from earlier runs (the imagination part); the BC term is logged separately.
        metrics.update(actor_metrics, actor_bc_loss=bc_loss, critic_loss=critic_loss,
                       actor_gradient_norm=an, critic_gradient_norm=cn)
    for group in (self.world.variables, self.actor.variables, self.critic.variables,
                  self.slow_critic.variables):
        for var in group:
            if tf.as_dtype(var.dtype).is_floating:
                tf.debugging.assert_all_finite(var, 'Non-finite parameter after update')
    metrics['model_gradient_norm'] = wn
    return metrics


def _update_with_demonstrations(self, data, world_only=False):
    data = _mix_demonstrations(data)
    demo = data if world_only else _demonstration_batch(int(data['valid'].shape[0]))   # unused when world_only
    return _update_graph(self, data, demo, world_only)


def world_model_class(learned_reward_std=None, delta_weight=None, reward_norm=False, reward_twohot=False,
                      edge_weight=None):
    base = validate_nrsm.DiagnosticWorldModel
    if sum(bool(x) for x in (learned_reward_std, reward_norm, reward_twohot)) > 1:
        raise ValueError('learned_reward_std, reward_norm and reward_twohot each replace the reward head')
    if not (learned_reward_std or delta_weight or reward_norm or reward_twohot or edge_weight):
        return base

    class AblationWorldModel(base):
        def __init__(self, config):
            super().__init__(config)
            if learned_reward_std:
                # Replacing the attribute keeps its position, so variable order stays deterministic.
                self.reward = LearnedStdHead(layers=2, units=config.hidden, min_std=learned_reward_std)
            if reward_norm:
                self.reward = NormalisedRewardHead(units=config.hidden)
            if reward_twohot:
                self.reward = TwoHotRewardHead(units=config.hidden)
            if delta_weight:
                self.delta_weight = float(delta_weight)
                self.delta = MotionHead(len(DELTA_DIMS), layers=2, units=config.hidden)
                # Create its variables now, before OnlineAgent builds the optimizer (seeded: no RNG shift).
                self.delta(tf.zeros([1, config.stoch * config.classes + config.context]))
            if edge_weight:
                self.edge_weight = float(edge_weight)
                self.edge = MotionHead(4, layers=2, units=config.hidden, seed=EDGE_SEED)
                self.edge(tf.zeros([1, config.stoch * config.classes + config.context]))

        def delta_nll(self, feat, data):
            return tools.masked_mean(-self.delta(feat[:, 1:]).log_prob(delta_targets(data['vector'])),
                                     delta_mask(data))

        def edge_nll(self, feat, data):
            return tools.masked_mean(-self.edge(feat).log_prob(edge_targets(data['vector'])), data['valid'])

    return AblationWorldModel


def _world_loss_with_aux(self, data):
    """The team's world_loss, plus weight * NLL of each auxiliary head (motion, edge) on the same posterior
    features. With only the motion head, this is the same computation as experiments 3-5."""
    post, loss, metrics = _ORIGINAL_WORLD_LOSS(self, data)
    feat = self.world.core.get_feat(post)
    extra = {}
    if getattr(self.world, 'delta_weight', None):
        extra['delta_nll'] = self.world.delta_nll(feat, data)
        loss = loss + self.world.delta_weight * extra['delta_nll']
    if getattr(self.world, 'edge_weight', None):
        extra['edge_nll'] = self.world.edge_nll(feat, data)
        loss = loss + self.world.edge_weight * extra['edge_nll']
    return post, loss, dict(metrics, model_loss=loss, **extra)


def apply(learned_reward_std=None, delta_weight=None, reward_norm=False, reward_twohot=False, edge_weight=None,
          demo_seed=None):
    """Every OnlineAgent built after this call uses the selected world model and loss. With demo_seed, updates
    also train on the demonstrations (experiment 8), with their own random stream seeded by the run seed."""
    nrsm_online_agent.DiagnosticWorldModel = world_model_class(
        learned_reward_std, delta_weight, reward_norm, reward_twohot, edge_weight)
    nrsm_online_agent.OnlineAgent.world_loss = (
        _world_loss_with_aux if (delta_weight or edge_weight) else _ORIGINAL_WORLD_LOSS)
    if demo_seed is None:
        nrsm_online_agent.OnlineAgent.update = _ORIGINAL_UPDATE
    else:
        if _DEMOS['episodes'] is None:
            _DEMOS['episodes'] = controller_demonstrations()
        _DEMOS['rng'] = np.random.default_rng([DEMO_RNG_SEED, int(demo_seed)])
        nrsm_online_agent.OnlineAgent.update = _update_with_demonstrations


def core_config(stoch=None, classes=None):
    """The compact core config, with the stochastic state's size replaced where given."""
    compact = nrsm_online_agent.compact_config()
    return dataclasses.replace(compact, stoch=int(stoch or compact.stoch), classes=int(classes or compact.classes))


def run_changes(run_dir):
    sidecar = Path(run_dir).with_name(Path(run_dir).name + '.ablation.json')
    return json.loads(sidecar.read_text()).get('changed', {}) if sidecar.exists() else {}


def agent_for_run(run_dir):
    """Rebuild a run's exact agent (architecture, core config, loss and ac_config) from its manifest and sidecar.
    Demonstrations change only training updates, not the architecture, so evaluation needs no demo data."""
    manifest = json.loads((Path(run_dir) / 'manifest.json').read_text())
    changes = run_changes(run_dir)
    apply(changes.get(REWARD_STD_KEY, [None, None])[1], changes.get(DELTA_KEY, [None, None])[1],
          bool(changes.get(REWARD_NORM_KEY, [False, False])[1]),
          bool(changes.get(REWARD_TWOHOT_KEY, [False, False])[1]),
          changes.get(EDGE_KEY, [None, None])[1])
    return nrsm_online_agent.OnlineAgent(core_config=NRSMConfig(**manifest['core_config']),
                                         ac_config=nrsm_online_agent.ACConfig(**manifest['ac_config']))
