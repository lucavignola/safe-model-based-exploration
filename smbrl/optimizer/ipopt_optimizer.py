from functools import partial
from typing import Generic, Tuple, NamedTuple

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
from cyipopt import minimize_ipopt
from jax import jit, vmap, grad
from jax.nn import relu
from jaxtyping import Float, Array, Key, Scalar
from mbpo.optimizers.base_optimizer import BaseOptimizer
from mbpo.systems.dynamics.base_dynamics import DynamicsParams
from mbpo.systems.rewards.base_rewards import RewardParams
from mbpo.utils.optimizer_utils import rollout_actions
from mbpo.utils.type_aliases import OptimizerState

from smbrl.optimizer.icem import AbstractCost


class IPOPTParams(NamedTuple):
    """
    num_particles: int = 10
    u_min: float | chex.Array = minimal value for action
    u_max: float | chex.Array = maximal value for action
    warm_start: bool = If we shift the action sequence for one and repeat the last action at initialization

    """
    num_particles: int = 10
    u_min: float | chex.Array = -1.0
    u_max: float | chex.Array = 1.0
    warm_start: bool = True
    lambda_constraint: float = 1e7


@chex.dataclass
class IPOPTOptimizerState(OptimizerState, Generic[DynamicsParams, RewardParams]):
    best_sequence: chex.Array
    best_reward: chex.Array

    @property
    def action(self):
        return self.best_sequence[0]


class IPOPTOptimizer(BaseOptimizer, Generic[DynamicsParams, RewardParams]):

    def __init__(self,
                 horizon: int,
                 action_dim: int,
                 key: chex.PRNGKey = jax.random.PRNGKey(0),
                 opt_params: IPOPTParams = IPOPTParams(),
                 cost_fn: AbstractCost | None = None,
                 use_optimism: bool = True,
                 use_pessimism: bool = True,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.horizon = horizon
        self.opt_params = opt_params
        self.key = key
        self.opt_dim = (horizon,) + (action_dim,)
        self.action_dim = action_dim
        self.horizon = horizon
        self.cost_fn = cost_fn
        if use_optimism:
            self.summarize_raw_samples = jnp.max
        else:
            self.summarize_raw_samples = jnp.mean
        if use_pessimism:
            self.summarize_cost_samples = jnp.max
        else:
            self.summarize_cost_samples = jnp.mean

        self.obj_grad = jit(grad(self.objective, argnums=0))

    def init(self, key: chex.Array) -> IPOPTOptimizerState:
        assert self.system is not None, "iCem optimizer requires system to be defined."
        init_key, dummy_buffer_key, key = jax.random.split(key, 3)
        system_params = self.system.init_params(init_key)
        dummy_buffer_state = self.dummy_true_buffer_state(dummy_buffer_key)
        return IPOPTOptimizerState(
            true_buffer_state=dummy_buffer_state,
            system_params=system_params,
            best_sequence=jnp.zeros(self.opt_dim),
            best_reward=jnp.zeros(1).squeeze(),
            key=key,
        )

    @staticmethod
    def pack_optimization_vector(predicted_actions):
        return predicted_actions.reshape(-1)

    def unpack_optimization_vector(self, x):
        actions = x.reshape(self.horizon, self.system.u_dim)
        return actions

    @partial(jit, static_argnums=0)
    def objective(self,
                  seq: Float[Array, 'horizon action_dim'],
                  key: Key[Array, '2'],
                  initial_state: Float[Array, 'observation_dim'],
                  opt_state: IPOPTOptimizerState) -> Scalar:
        seq = seq.reshape(self.horizon, self.system.u_dim)

        def optimize_fn(init_state: Float[Array, 'observation_dim'], rng: Key[Array, '2']):
            system_params = opt_state.system_params.replace(key=rng)
            return rollout_actions(system=self.system,
                                   system_params=system_params,
                                   init_state=init_state,
                                   horizon=self.horizon,
                                   actions=seq,
                                   )

        particles_rng = jr.split(key, self.opt_params.num_particles)
        transitions = jax.vmap(optimize_fn, in_axes=(None, 0))(initial_state, particles_rng)
        cost = 0

        # We summarize cost with mean or max (if optimism is true)
        reward = self.summarize_raw_samples(jnp.mean(transitions.reward, axis=-1))
        if self.cost_fn is not None:
            cost = vmap(self.cost_fn)(transitions.observation, transitions.action)
            assert cost.shape == (self.opt_params.num_particles,)
            # We summarize cost with mean or max (if pessimism is true)
            cost = self.summarize_cost_samples(cost)
        return - (reward - self.opt_params.lambda_constraint * relu(cost))

    def optimize(
            self,
            initial_state: Float[Array, 'observation_dim'],
            opt_state: IPOPTOptimizerState,
    ) -> IPOPTOptimizerState:
        actions = opt_state.best_sequence
        if self.opt_params.warm_start:
            actions = actions.at[:-1].set(opt_state.best_sequence[1:])
            actions = actions.at[-1].set(opt_state.best_sequence[-1])
        x = self.pack_optimization_vector(actions)

        optimizer_key, key = jax.random.split(opt_state.key, 2)
        new_opt_state = opt_state.replace(key=key)

        bnds = [(-1, 1) for _ in range(self.horizon * self.system.u_dim)]
        out = minimize_ipopt(self.objective, jac=self.obj_grad, x0=x, options={'disp': 0, 'maxiter': 100},
                             args=(key, initial_state, opt_state), bounds=bnds)
        new_actions = jnp.array(out.x)
        new_actions = self.unpack_optimization_vector(new_actions)
        new_reward = jnp.array(out.fun).item()
        new_opt_state = new_opt_state.replace(best_sequence=new_actions, best_reward=new_reward)
        return new_opt_state

    def act(self,
            obs: chex.Array,
            opt_state: IPOPTOptimizerState,
            evaluate: bool = True) -> Tuple[
        Float[Array, 'action_dim'], IPOPTOptimizerState]:
        new_opt_state = self.optimize(initial_state=obs, opt_state=opt_state)
        return new_opt_state.action, new_opt_state
