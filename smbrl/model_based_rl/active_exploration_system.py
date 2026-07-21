from dataclasses import field
from typing import Generic, Tuple

import chex
import jax.numpy as jnp
import jax.random
import jax.random as jr
from bsm.statistical_model import StatisticalModel
from bsm.utils.type_aliases import ModelState
from distrax import Distribution, Normal
from jaxtyping import Float, Array, Scalar
from mbpo.systems.base_systems import SystemParams, SystemState, System
from mbpo.systems.dynamics.base_dynamics import Dynamics
from mbpo.systems.dynamics.base_dynamics import DynamicsParams as DummyDynamicsParams
from mbpo.systems.rewards.base_rewards import Reward, RewardParams

from smbrl.dynamics_models.gp_sampling import (
    RFFPosteriorState,
    evaluate_rff_posterior,
)


@chex.dataclass
class DynamicsParams(Generic[ModelState, DummyDynamicsParams]):
    key: chex.PRNGKey
    model_state: ModelState
    posterior_path_state: RFFPosteriorState | None = None
    sample_index: chex.Array = field(
        default_factory=lambda: jnp.asarray(0, dtype=jnp.int32)
    )


class ExplorationDynamics(Dynamics, Generic[ModelState]):
    def __init__(self,
                 x_dim: int,
                 u_dim: int,
                 model: StatisticalModel,
                 use_log: bool = True,
                 scale_with_aleatoric_std: bool = True,
                 aleatoric_noise_in_prediction: bool = True,
                 predict_difference: bool = True,
                 gp_sampling_method: str = "marginal",
                 rff_path_scale: float | None = None,
                 ):
        Dynamics.__init__(self, x_dim=x_dim, u_dim=u_dim)
        self.model = model
        self.use_log = use_log
        self.scale_with_aleatoric_std = scale_with_aleatoric_std
        self.aleatoric_noise_in_prediction = aleatoric_noise_in_prediction
        self.predict_difference = predict_difference
        if gp_sampling_method not in {"marginal", "rff"}:
            raise ValueError(
                "gp_sampling_method must be one of {'marginal', 'rff'}, "
                f"got {gp_sampling_method!r}."
            )
        self.gp_sampling_method = gp_sampling_method
        if rff_path_scale is not None and rff_path_scale < 0:
            raise ValueError(
                f"rff_path_scale must be non-negative, got {rff_path_scale}."
            )
        self.rff_path_scale = rff_path_scale

    def init_params(self, key: chex.PRNGKey) -> DynamicsParams:
        param_key, model_state_key = jr.split(key, 2)
        model_state = self.model.init(model_state_key)
        return DynamicsParams(
            key=param_key,
            model_state=model_state,
            posterior_path_state=None,
            sample_index=jnp.asarray(0, dtype=jnp.int32),
        )

    def get_intrinsic_reward(self,
                             epistemic_std: Float[Array, '... observation_dim'],
                             aleatoric_std: Float[Array, '... observation_dim']) -> Scalar:
        if self.scale_with_aleatoric_std:
            # sigma^2_ep / sigma^2_al
            intrinsic_reward = jnp.square(epistemic_std / jnp.clip(aleatoric_std, 1e-4, None))
        else:
            # sigma^2_ep
            intrinsic_reward = jnp.square(epistemic_std)
        if self.use_log:
            # use log transform
            intrinsic_reward = jnp.log(1 + intrinsic_reward)
        # sum over the state axis
        return jnp.sum(intrinsic_reward, axis=0)

    def next_state(self,
                   x: chex.Array,
                   u: chex.Array,
                   dynamics_params: DynamicsParams) -> Tuple[Distribution, DynamicsParams]:
        assert x.shape == (self.x_dim,) and u.shape == (self.u_dim,)
        # Create state-action pair
        z = jnp.concatenate([x, u])
        next_key, key_sample_x_next = jr.split(dynamics_params.key)
        pred = self.model(z, dynamics_params.model_state)
        epistemic_std, aleatoric_std = pred.epistemic_std, pred.aleatoric_std
        beta = pred.statistical_model_state.beta

        if self.gp_sampling_method == "marginal":
            model_prediction = (
                pred.mean
                + beta
                * epistemic_std
                * jr.normal(key=key_sample_x_next, shape=pred.mean.shape)
            )
        else:
            if dynamics_params.posterior_path_state is None:
                raise ValueError(
                    "RFF dynamics require an episode posterior-path state."
                )
            posterior_path_value = evaluate_rff_posterior(
                model=self.model,
                model_state=dynamics_params.model_state,
                posterior_state=dynamics_params.posterior_path_state,
                input_value=z,
                path_index=dynamics_params.sample_index,
            )
            # Scale 1 is an approximate posterior draw.  A larger explicit
            # scale keeps the same global path while inflating its residual
            # around the exact GP mean (e.g. scale=beta for matching the
            # one-point variance of the marginal sampler).
            # By default, inherit the GP calibration multiplier so that the
            # coherent sampler matches the one-point variance used by the
            # existing marginal sampler.  Passing 1 explicitly gives a
            # literal approximate posterior path.
            path_scale = beta if self.rff_path_scale is None else self.rff_path_scale
            model_prediction = pred.mean + path_scale * (
                posterior_path_value - pred.mean
            )

        if self.predict_difference:
            x_next = x + model_prediction
        else:
            x_next = model_prediction

        intrinsic_reward = self.get_intrinsic_reward(epistemic_std, aleatoric_std)
        intrinsic_reward = jnp.atleast_1d(intrinsic_reward)

        if not self.aleatoric_noise_in_prediction:
            aleatoric_std = 0 * aleatoric_std
        # add intrinsic reward to the next state
        x_next_with_reward = jnp.concatenate([x_next, intrinsic_reward], axis=-1)
        aleatoric_std_with_reward = jnp.concatenate([aleatoric_std, jnp.zeros_like(intrinsic_reward)], axis=-1)
        new_dynamics_params = dynamics_params.replace(key=next_key)
        return Normal(loc=x_next_with_reward, scale=aleatoric_std_with_reward), new_dynamics_params


@chex.dataclass
class ExplorationRewardParams:
    action_cost: chex.Array | float = 0.0


class ExplorationReward(Reward, ExplorationRewardParams):
    def __init__(self, x_dim: int, u_dim: int):
        super().__init__(x_dim=x_dim, u_dim=u_dim)

    def __call__(self,
                 x: chex.Array,
                 u: chex.Array,
                 reward_params: ExplorationRewardParams,
                 x_next: chex.Array | None = None) -> Tuple[Distribution, RewardParams]:
        chex.assert_shape(x, (self.x_dim,))
        chex.assert_shape(u, (self.u_dim,))
        chex.assert_shape(x_next, (self.x_dim + 1,))
        # get intrinsic reward out
        intrinsic_reward = x_next[-1]
        total_reward = intrinsic_reward - reward_params.action_cost * jnp.sum(jnp.square(u), axis=0)
        return Normal(loc=total_reward, scale=jnp.zeros_like(total_reward)), reward_params

    def init_params(self, key: chex.PRNGKey) -> ExplorationRewardParams:
        return ExplorationRewardParams()


class ExplorationSystem(System, Generic[ModelState, RewardParams]):
    def __init__(self, dynamics: ExplorationDynamics[ModelState], reward: Reward[RewardParams] | None = None):
        if reward is None:
            reward = ExplorationReward(x_dim=dynamics.x_dim, u_dim=dynamics.u_dim)
        super().__init__(dynamics, reward)
        self.dynamics = dynamics
        self.reward = reward
        self.x_dim = dynamics.x_dim
        self.u_dim = dynamics.u_dim

    def get_reward(self,
                   x: chex.Array,
                   u: chex.Array,
                   reward_params: RewardParams,
                   x_next: chex.Array,
                   key: jax.random.PRNGKey):
        # x_next includes the next state and the intrinsic reward
        chex.assert_shape(x_next, (self.x_dim + 1,))
        if isinstance(self.reward, ExplorationReward):
            # include the intrinsic reward in x_next
            reward_dist, new_reward_params = self.reward(x, u, reward_params, x_next)
        else:
            # Check if it's an SBSRL reward that needs intrinsic reward
            from smbrl.agent.sbsrl import SBSRLReward
            if isinstance(self.reward, SBSRLReward): #TODO: I think this if can be merged with the one above
                # SBSRL needs full x_next (with intrinsic reward) like ExplorationReward
                reward_dist, new_reward_params = self.reward(x, u, reward_params, x_next)
            else:
                # ignore the last state in x_next which is the intrinsic reward
                reward_dist, new_reward_params = self.reward(x, u, reward_params, x_next[:-1])
        reward = reward_dist.sample(seed=key)
        return reward, new_reward_params

    def step(self,
             x: chex.Array,
             u: chex.Array,
             system_params: SystemParams[ModelState, RewardParams],
             ) -> SystemState:
        """

        :param x: current state of the system
        :param u: current action of the system
        :param system_params: parameters of the system
        :return: Tuple of next state, reward, updated system parameters
        """
        assert x.shape == (self.x_dim,) and u.shape == (self.u_dim,)
        x_next_dist, new_dynamics_params = self.dynamics.next_state(x, u, system_params.dynamics_params)
        next_state_key, reward_key, new_systems_key = jr.split(system_params.key, 3)
        x_next = x_next_dist.sample(seed=next_state_key)
        reward, new_reward_params = self.get_reward(x, u, system_params.reward_params, x_next, reward_key)
        new_systems_params = system_params.replace(dynamics_params=new_dynamics_params,
                                                   reward_params=new_reward_params,
                                                   key=new_systems_key)
        new_system_state = SystemState(
            x_next=x_next[:-1],
            reward=reward,
            system_params=new_systems_params,
            done=jnp.array(0.0),
        )
        return new_system_state
