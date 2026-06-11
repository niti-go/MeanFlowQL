"""MeanFlow behavior-cloning agent — MeanFlowQL with the critic removed.

This is the critic-free ablation of `meanflowql.py`.  It keeps the exact
same MeanFlow action generator (flow matching + JVP target + consistency
loss) but drops everything that depends on a Q-function:

  - no critic_loss (no Bellman backup, no target critic)
  - no actor_loss q-term (no reward-guided policy improvement)
  - no best-of-N action selection (that ranks candidates by Q)
  - no alpha scheduling (there is only one loss to balance)

What remains is pure flow-based imitation of the dataset actions, with an
optional action-bound regularizer carried over from the original
actor_loss.  At inference, actions are drawn directly from the flow
(`action_mode='normal'`) since there is no critic to rank candidates.

Compare against `meanflowql` to isolate the contribution of the
Q-learning half of the algorithm.
"""
import flax
import jax
import jax.nn as nn
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.dit_jax import MFDiT_SIM


class MeanFlowBC_Agent(flax.struct.PyTreeNode):
    rng: any
    network: any
    config: any = nonpytree_field()

    # ------------------------------------------------------------------ noise
    def sample_noise(self, rng, shape):
        noise_type = self.config.get('noise_type', 'gaussian')
        if noise_type == 'gaussian':
            sigma = self.config.get('sigma', 1.0)
            if sigma <= 0:
                raise ValueError(f"sigma must be positive, got {sigma}")
            return jax.random.normal(rng, shape) * sigma
        elif noise_type == 'uniform':
            return jax.random.uniform(rng, shape, minval=-1.0, maxval=1.0)
        raise ValueError(f"Unsupported noise_type: {noise_type}")

    # ------------------------------------------------------------- flow losses
    def meanflow_loss(self, batch, grad_params, rng):
        batch_size, action_dim = batch['actions'].shape
        rng, t_rng, noise_rng = jax.random.split(rng, 3)

        time_steps = self.config.get('time_steps', 10)
        if time_steps <= 1000:
            time_values = jnp.linspace(1 / time_steps, 1.0, time_steps)
            indices = jax.random.randint(t_rng, (batch_size,), 0, time_steps)
            t = time_values[indices].reshape(-1, 1)
        else:
            t = jax.random.uniform(t_rng, (batch_size, 1))

        actions = batch['actions']
        e = self.sample_noise(noise_rng, batch['actions'].shape)
        z = (1 - t) * actions + t * e
        v = e - actions

        gn = self.network.select('actor_bc_flow')
        g, dgdt = jax.jvp(
            lambda args: gn(batch['observations'], args[0], args[1], params=grad_params),
            ((z, t),),
            ((v, jnp.ones_like(t)),),
        )
        g_tgt = z + (t - 1) * v - t * dgdt
        g_tgt = jax.lax.stop_gradient(g_tgt)
        g_tgt = jnp.clip(g_tgt, -5, 5)
        err = g - g_tgt
        mean_flow_loss = self.adaptive_l2_loss(err, t, mode="normal")

        consistency_loss = self.consistency_loss(batch, grad_params, rng)
        flow_loss = mean_flow_loss + consistency_loss * self.config.get('consistency_alpha', 1)

        return flow_loss, {
            'mean_flow_loss': mean_flow_loss,
            'consistency_loss': consistency_loss,
            'flow_loss': flow_loss,
        }

    def consistency_loss(self, batch, grad_params, rng):
        batch_size, action_dim = batch['actions'].shape
        rng, noise_rng = jax.random.split(rng, 2)
        t1, t2 = self.sample_discrete_t(rng, batch_size,
                                        time_steps=self.config.get('time_steps', 50))
        actions = batch['actions']
        e = self.sample_noise(noise_rng, batch['actions'].shape)
        z_t1 = (1 - t1) * actions + t1 * e
        z_t2 = (1 - t2) * actions + t2 * e
        z_0_t1 = z_t1 - t1 * (z_t1 - self.network.select('actor_bc_flow')(
            batch['observations'], z_t1, t1, params=grad_params))
        z_0_t2 = z_t2 - t2 * (z_t2 - self.network.select('actor_bc_flow')(
            batch['observations'], z_t2, t2))
        z_0_t2 = jax.lax.stop_gradient(z_0_t2)
        return jnp.square(z_0_t1 - z_0_t2).mean()

    # ----------------------------------------------- bound regularizer (no Q)
    def bc_reg_loss(self, batch, grad_params, rng):
        """Action-bound regularizer + BC diagnostics. No critic involved.

        Mirrors the non-Q part of the original actor_loss: generate an action
        from the flow at t=1, penalize out-of-bound predictions, and report
        the MSE to the dataset action as a behavior-cloning diagnostic.
        """
        batch_size, action_dim = batch['actions'].shape
        rng, noise_rng = jax.random.split(rng)
        t_pred = jnp.ones((batch_size, 1))
        noises = self.sample_noise(noise_rng, (batch_size, action_dim))
        actions = self.network.select('actor_bc_flow')(
            batch['observations'], noises, t=t_pred, params=grad_params)

        upper, lower = jnp.ones_like(actions), -jnp.ones_like(actions)
        bound_loss = jnp.mean(nn.relu(actions - upper)) + jnp.mean(nn.relu(lower - actions))
        mse = jnp.mean((jnp.clip(actions, -1, 1) - batch['actions']) ** 2)

        reg_loss = bound_loss * self.config.get('bound_loss_weight', 1.0)
        return reg_loss, {'bound_loss': bound_loss, 'mse': mse}

    # ------------------------------------------------------------- total loss
    @jax.jit
    def total_loss(self, batch, grad_params, rng, current_step=0):
        info = {}
        rng = rng if rng is not None else self.rng
        flow_rng, reg_rng = jax.random.split(rng)

        flow_loss, flow_info = self.meanflow_loss(batch, grad_params, flow_rng)
        for k, v in flow_info.items():
            info[f'meanflow/{k}'] = v

        reg_loss, reg_info = self.bc_reg_loss(batch, grad_params, reg_rng)
        for k, v in reg_info.items():
            info[f'actor/{k}'] = v

        total = flow_loss + reg_loss
        info['total_loss'] = total
        return total, info

    @jax.jit
    def update(self, batch, current_step=0):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng, current_step=current_step)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)

        if current_step is not None:
            lr_schedule = self.config.get('actor_lr_schedule')
            if lr_schedule is not None:
                info['metrics/actor_learning_rate'] = lr_schedule(current_step)

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def pretrain(self, batch, current_step=None):
        """Identical objective to update() — kept for driver compatibility."""
        new_rng, rng = jax.random.split(self.rng)

        def pretrain_loss(grad_params):
            return self.meanflow_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=pretrain_loss)
        if current_step is not None:
            lr_schedule = self.config.get('actor_lr_schedule')
            if lr_schedule is not None:
                info['metrics/actor_learning_rate'] = lr_schedule(current_step)
        return self.replace(network=new_network, rng=new_rng), info

    # --------------------------------------------------------- action sampling
    @jax.jit
    def sample_actions_normal(self, observations, temperature=1, seed=None):
        """Single flow draw — the default for critic-free BC."""
        action_seed, _ = jax.random.split(seed)
        action_shape = (
            *observations.shape[: -len(self.config['ob_dims'])],
            self.config['action_dim'],
        )
        e = self.sample_noise(action_seed, action_shape)
        batch_size = observations.shape[0]
        t = jnp.ones((batch_size, 1))
        if self.config['encoder'] is not None:
            encoded_obs = self.network.select('actor_bc_flow_encoder')(observations)
            actions = self.network.select('actor_bc_flow')(encoded_obs, e, t, is_encoded=True)
        else:
            actions = self.network.select('actor_bc_flow')(observations, e, t)
        return jnp.clip(actions, -1, 1)

    @jax.jit
    def sample_actions_mean(self, observations, temperature=1, seed=None, num_candidates=None):
        """Average of N flow draws — smoother actions, still critic-free."""
        action_seed, _ = jax.random.split(seed)
        if self.config['encoder'] is not None and observations.ndim == 3:
            observations = observations[None, :]
        batch_size = observations.shape[0]
        action_dim = self.config['action_dim']
        num_candidates = num_candidates or self.config.get('num_candidates', 4)

        candidate_seeds = jax.random.split(action_seed, num_candidates)
        noise_fn = lambda s: self.sample_noise(s, (batch_size, action_dim))
        all_noise = jax.vmap(noise_fn)(candidate_seeds)
        t_expanded = jnp.ones((num_candidates, batch_size, 1))

        if self.config['encoder'] is not None:
            encoded_obs = self.network.select('actor_bc_flow_encoder')(observations)
            if encoded_obs.ndim == 1:
                encoded_obs = encoded_obs[None, :]
            elif encoded_obs.ndim == 3:
                encoded_obs = encoded_obs[:, -1, :]
            obs_exp = jnp.tile(encoded_obs[None, :, :], (num_candidates, 1, 1))
            gen = lambda o, n, t: self.network.select('actor_bc_flow')(o, n, t, is_encoded=True)
        else:
            obs_exp = jnp.tile(observations[None, :, :], (num_candidates, 1, 1))
            gen = lambda o, n, t: self.network.select('actor_bc_flow')(o, n, t)

        candidate_actions = jax.vmap(gen)(obs_exp, all_noise, t_expanded)
        if candidate_actions.ndim == 4:
            candidate_actions = candidate_actions.squeeze(2)
        return jnp.clip(jnp.mean(candidate_actions, axis=0), -1, 1)

    @jax.jit
    def sample_actions(self, observations, temperature=1, seed=None, num_candidates=None):
        action_mode = self.config.get('action_mode', 'normal')
        if action_mode == 'mean':
            return self.sample_actions_mean(observations, temperature, seed, num_candidates)
        elif action_mode == 'normal':
            return self.sample_actions_normal(observations, temperature, seed)
        raise ValueError(
            f"Unknown action_mode {action_mode!r}. MeanFlowBC supports 'normal' "
            f"and 'mean' (best-of-N needs a critic and is unavailable here).")

    # ----------------------------------------------------------------- create
    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)
        batch_size = ex_observations.shape[0]
        ex_t = jnp.ones((batch_size, 1))
        ob_dims = ex_observations.shape[1:]
        action_dim = ex_actions.shape[-1]

        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['actor_bc_flow'] = encoder_module()

        actor_bc_flow_def = MFDiT_SIM(
            hidden_dim=config['actor_hidden_dims'],
            depth=config['actor_depth'],
            num_heads=config['actor_num_heads'],
            output_dim=action_dim,
            encoder=encoders.get('actor_bc_flow'),
            tanh_squash=config['tanh_squash'],
            use_output_layernorm=config['use_output_layernorm'],
        )

        network_info = dict(
            actor_bc_flow=(actor_bc_flow_def, (ex_observations, ex_actions, ex_t)),
        )
        if encoders.get('actor_bc_flow') is not None:
            network_info['actor_bc_flow_encoder'] = (
                encoders.get('actor_bc_flow'), (ex_observations,))
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}
        network_def = ModuleDict(networks)

        # Phase-aware (warmup + cosine) LR schedule over the offline budget.
        base_lr = config['lr']
        from absl import flags
        FLAGS = flags.FLAGS
        pretrain_steps = FLAGS.offline_steps * FLAGS.pretrain_factor
        offline_steps = FLAGS.offline_steps
        offline_end_step = pretrain_steps + offline_steps
        config['pretrain_plus_offline_steps'] = offline_end_step
        min_lr = base_lr * config.get('lr_min_ratio', 0.05)
        warmup_steps = int(offline_end_step * 0.05)

        def lr_schedule(step):
            if warmup_steps > 0:
                warmup_lr = (step / warmup_steps) * base_lr
                cos = (step - warmup_steps) / jnp.maximum(offline_end_step - warmup_steps, 1)
                cos = jnp.clip(cos, 0.0, 1.0)
                cosine_lr = min_lr + (base_lr - min_lr) * 0.5 * (1 + jnp.cos(jnp.pi * cos))
                return jnp.where(step <= warmup_steps, warmup_lr, cosine_lr)
            cos = jnp.clip(step / jnp.maximum(offline_end_step, 1), 0.0, 1.0)
            return min_lr + (base_lr - min_lr) * 0.5 * (1 + jnp.cos(jnp.pi * cos))

        config['actor_lr_schedule'] = lr_schedule

        network_tx = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(learning_rate=lr_schedule),
        )

        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        if 'metric' not in config:
            config['metric'] = lambda x: jnp.mean(x ** 2)
        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))

    # ------------------------------------------------------------- flow utils
    def adaptive_l2_loss(self, error, t, gamma=None, c=None, mode="normal"):
        gamma = gamma if gamma is not None else self.config.get('adaptive_gamma', 0.5)
        c = c if c is not None else self.config.get('adaptive_c', 1e-3)
        delta_sq = jnp.maximum(jnp.mean(error ** 2, axis=-1), 1e-12)
        p = 1.0 - gamma
        denom = jnp.maximum(jnp.power(delta_sq + c, p), 1e-12)
        w = jnp.clip(1.0 / denom, 1e-6, 1e6)
        loss = delta_sq
        if mode != "normal":
            w = w * (t * (1.0 - t) + 0.75).squeeze(-1)
        return jnp.mean(jax.lax.stop_gradient(w) * loss)

    def sample_discrete_t(self, rng, batch_size, time_steps=100):
        t_rng, t_con_rng = jax.random.split(rng)
        time_values = jnp.linspace(1 / time_steps, 1.0, time_steps)
        t_idx = jax.random.randint(t_rng, (batch_size,), 0, time_steps)
        t_con_idx = jax.random.randint(t_con_rng, (batch_size,), 0, time_steps)
        return time_values[t_idx].reshape(-1, 1), time_values[t_con_idx].reshape(-1, 1)

    # --------------------------------------------------------------- param log
    def get_param_count(self):
        params = self.network.params
        if hasattr(params, 'unfreeze'):
            params = params.unfreeze()
        counts = {}
        for module_name, module_params in params.items():
            leaves = jax.tree_util.tree_leaves(module_params)
            counts[module_name] = sum(p.size for p in leaves)
        counts['total'] = sum(p.size for p in jax.tree_util.tree_leaves(params))
        return counts

    def print_param_stats(self):
        counts = self.get_param_count()
        print("Network Parameter Statistics:")
        print("-" * 50)
        for module_name, count in counts.items():
            if module_name != 'total':
                print(f"{module_name}: {count:,} parameters ({count * 4 / (1024**2):.2f} MB)")
        print("-" * 50)
        print(f"Total parameters: {counts['total']:,} "
              f"({counts['total'] * 4 / (1024**2):.2f} MB)")


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='meanflow_bc',
            ob_dims=ml_collections.config_dict.placeholder(list),
            action_dim=ml_collections.config_dict.placeholder(int),
            encoder=ml_collections.config_dict.placeholder(str),

            # noise / flow
            sigma=1.0,
            consistency_alpha=0.0,
            batch_size=256,
            noise_type="gaussian",

            # actor (flow) network
            lr=1e-4,
            lr_min_ratio=0.1,
            actor_hidden_dims=256,
            actor_depth=3,
            actor_num_heads=2,
            actor_layer_norm=False,
            tanh_squash=False,
            use_output_layernorm=False,
            time_steps=10000,

            # meanflow loss
            adaptive_gamma=0.8,
            adaptive_c=1e-4,
            bound_loss_weight=1.0,

            # action sampling (critic-free): 'normal' (single draw) or 'mean'
            action_mode="normal",
            num_candidates=4,  # only used when action_mode='mean'
        )
    )
    return config
