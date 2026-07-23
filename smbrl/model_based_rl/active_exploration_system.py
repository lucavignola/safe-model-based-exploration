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
    RFFPriorState,
    RFFPosteriorState,
    evaluate_rff_prior,
    evaluate_rff_posterior,
)


@chex.dataclass
class DynamicsParams(Generic[ModelState, DummyDynamicsParams]):
    key: chex.PRNGKey
    model_state: ModelState
    posterior_path_state: RFFPosteriorState | RFFPriorState | None = None
    sample_index: chex.Array = field(
        default_factory=lambda: jnp.asarray(0, dtype=jnp.int32)
    )
    use_posterior_mean: chex.Array = field(
        default_factory=lambda: jnp.asarray(False)
    )


class ExplorationDynamics(Dynamics, Generic[ModelState]):
    GP_SAMPLE_TRUNCATION_MODES = {
        "none",
        "posterior",
        "prior",
        "recursive",
    }
    GP_PATH_SOURCES = {"posterior", "prior"}

    def __init__(self,
                 x_dim: int,
                 u_dim: int,
                 model: StatisticalModel,
                 use_log: bool = True,
                 scale_with_aleatoric_std: bool = True,
                 aleatoric_noise_in_prediction: bool = True,
                 predict_difference: bool = True,
                 gp_sampling_method: str = "marginal",
                 gp_path_source: str = "posterior",
                 rff_path_scale: float | None = None,
                 gp_sample_truncation: str = "none",
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
        if gp_path_source not in self.GP_PATH_SOURCES:
            raise ValueError(
                "gp_path_source must be one of "
                f"{sorted(self.GP_PATH_SOURCES)}, got {gp_path_source!r}."
            )
        if gp_path_source == "prior" and gp_sampling_method != "rff":
            raise ValueError(
                "Whole-run prior paths require gp_sampling_method='rff'."
            )
        self.gp_path_source = gp_path_source
        if rff_path_scale is not None and rff_path_scale < 0:
            raise ValueError(
                f"rff_path_scale must be non-negative, got {rff_path_scale}."
            )
        if (
                gp_path_source == "prior"
                and rff_path_scale is not None
                and rff_path_scale != 1.0
        ):
            raise ValueError(
                "Whole-run prior RFF paths are uninflated prior draws and require "
                "rff_path_scale=1."
            )
        if gp_path_source == "prior" and rff_path_scale is None:
            rff_path_scale = 1.0
        self.rff_path_scale = rff_path_scale
        if gp_sample_truncation not in self.GP_SAMPLE_TRUNCATION_MODES:
            raise ValueError(
                "gp_sample_truncation must be one of "
                f"{sorted(self.GP_SAMPLE_TRUNCATION_MODES)}, "
                f"got {gp_sample_truncation!r}."
            )
        self.gp_sample_truncation = gp_sample_truncation
        if (
                gp_sample_truncation == "recursive"
                and gp_path_source != "prior"
        ):
            raise ValueError(
                "Recursive confidence truncation requires "
                "gp_path_source='prior'."
            )

    def _prior_epistemic_std(
            self,
            z: chex.Array,
            model_state: ModelState,
    ) -> chex.Array:
        """Returns sqrt(k(z, z)) in the GP prediction's output units."""

        if (
                not hasattr(self.model, "model")
                or not hasattr(self.model.model, "m_kernel_multiple_output")
        ):
            raise TypeError(
                "Prior GP truncation requires a GPStatisticalModel-compatible "
                "model."
            )

        gp_state = model_state.model_state
        normalized_z = (
            z - gp_state.data_stats.inputs.mean
        ) / gp_state.data_stats.inputs.std
        normalized_prior_variance = self.model.model.m_kernel_multiple_output(
            normalized_z[None, :],
            normalized_z[None, :],
            gp_state.params,
        )[:, 0, 0]
        normalized_prior_std = jnp.sqrt(
            jnp.maximum(normalized_prior_variance, 0.0)
        )
        return normalized_prior_std * gp_state.data_stats.outputs.std

    def _truncate_gp_sample(
            self,
            model_prediction: chex.Array,
            posterior_mean: chex.Array,
            posterior_epistemic_std: chex.Array,
            beta: chex.Array,
            z: chex.Array,
            model_state: ModelState,
            path_state: RFFPosteriorState | RFFPriorState | None = None,
    ) -> chex.Array:
        """Projects a GP sample onto the selected beta-confidence tube."""

        if self.gp_sample_truncation == "none":
            return model_prediction
        if self.gp_sample_truncation == "recursive":
            if not isinstance(path_state, RFFPriorState):
                raise TypeError(
                    "Recursive truncation requires an RFFPriorState."
                )
            return self._recursively_truncate_prior_sample(
                model_prediction=model_prediction,
                z=z,
                prior_state=path_state,
            )
        if self.gp_sample_truncation == "posterior":
            truncation_std = posterior_epistemic_std
        else:
            truncation_std = self._prior_epistemic_std(z, model_state)

        truncation_radius = beta * truncation_std
        return jnp.clip(
            model_prediction,
            posterior_mean - truncation_radius,
            posterior_mean + truncation_radius,
        )

    def _initial_prior_distribution(
            self,
            z: chex.Array,
            prior_state: RFFPriorState,
    ) -> tuple[chex.Array, chex.Array]:
        """Returns the frozen online prior mean and epistemic std at ``z``."""

        if prior_state.initial_model_state is not None:
            prediction = self.model(z, prior_state.initial_model_state)
            return prediction.mean, prediction.epistemic_std

        path_state = prior_state.path_state
        normalized_z = (
            z - path_state.input_mean
        ) / path_state.input_std
        normalized_variance = self.model.model.m_kernel_multiple_output(
            normalized_z[None, :],
            normalized_z[None, :],
            path_state.kernel_params,
        )[:, 0, 0]
        prior_std = (
            jnp.sqrt(jnp.maximum(normalized_variance, 0.0))
            * path_state.output_std
        )
        return path_state.output_mean, prior_std

    def _recursively_truncate_prior_sample(
            self,
            model_prediction: chex.Array,
            z: chex.Array,
            prior_state: RFFPriorState,
    ) -> chex.Array:
        """Applies the paper's pointwise recursive clipping operation.

        Re-evaluating every immutable posterior snapshot is equivalent to
        constructing ``f_n(z)`` sequentially, while avoiding a discretization
        of the continuous state-action domain.
        """

        initial_mean, initial_std = self._initial_prior_distribution(
            z, prior_state
        )
        truncated_prediction = jnp.clip(
            model_prediction,
            initial_mean - prior_state.initial_beta * initial_std,
            initial_mean + prior_state.initial_beta * initial_std,
        )
        for confidence_model_state in prior_state.confidence_model_states:
            confidence_prediction = self.model(z, confidence_model_state)
            confidence_radius = (
                confidence_prediction.statistical_model_state.beta
                * confidence_prediction.epistemic_std
            )
            truncated_prediction = jnp.clip(
                truncated_prediction,
                confidence_prediction.mean - confidence_radius,
                confidence_prediction.mean + confidence_radius,
            )
        return truncated_prediction

    def init_params(self, key: chex.PRNGKey) -> DynamicsParams:
        param_key, model_state_key = jr.split(key, 2)
        model_state = self.model.init(model_state_key)
        return DynamicsParams(
            key=param_key,
            model_state=model_state,
            posterior_path_state=None,
            sample_index=jnp.asarray(0, dtype=jnp.int32),
            use_posterior_mean=jnp.asarray(False),
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
                + epistemic_std
                * jr.normal(key=key_sample_x_next, shape=pred.mean.shape)
            )
        else:
            if dynamics_params.posterior_path_state is None:
                raise ValueError(
                    "RFF dynamics require a path state."
                )
            if self.gp_path_source == "prior":
                if not isinstance(
                        dynamics_params.posterior_path_state,
                        RFFPriorState,
                ):
                    raise TypeError(
                        "gp_path_source='prior' requires an RFFPriorState."
                    )
                model_prediction = evaluate_rff_prior(
                    model=self.model,
                    prior_state=dynamics_params.posterior_path_state,
                    input_value=z,
                    path_index=dynamics_params.sample_index,
                )
            else:
                if not isinstance(
                        dynamics_params.posterior_path_state,
                        RFFPosteriorState,
                ):
                    raise TypeError(
                        "gp_path_source='posterior' requires an "
                        "RFFPosteriorState."
                    )
                posterior_path_value = evaluate_rff_posterior(
                    model=self.model,
                    model_state=dynamics_params.model_state,
                    posterior_state=dynamics_params.posterior_path_state,
                    input_value=z,
                    path_index=dynamics_params.sample_index,
                )
                # Scale 1 is an approximate posterior draw.  A larger
                # explicit scale keeps the same global path while inflating
                # its residual around the exact GP mean.
                path_scale = (
                    beta
                    if self.rff_path_scale is None
                    else self.rff_path_scale
                )
                model_prediction = pred.mean + path_scale * (
                    posterior_path_value - pred.mean
                )

        model_prediction = self._truncate_gp_sample(
            model_prediction=model_prediction,
            posterior_mean=pred.mean,
            posterior_epistemic_std=epistemic_std,
            beta=beta,
            z=z,
            model_state=dynamics_params.model_state,
            path_state=dynamics_params.posterior_path_state,
        )
        # Problem (8) evaluates task reward under the deterministic posterior
        # mean dynamics while retaining sampled models for the safety
        # constraints. iCEM opts into this branch only for its separate reward
        # rollout; ordinary particle rollouts keep the legacy sampled dynamics.
        model_prediction = jnp.where(
            dynamics_params.use_posterior_mean,
            pred.mean,
            model_prediction,
        )

        if self.predict_difference:
            x_next = x + model_prediction
        else:
            x_next = model_prediction

        intrinsic_reward = self.get_intrinsic_reward(epistemic_std, aleatoric_std)
        intrinsic_reward = jnp.atleast_1d(intrinsic_reward)

        if not self.aleatoric_noise_in_prediction:
            aleatoric_std = 0 * aleatoric_std
        aleatoric_std = jnp.where(
            dynamics_params.use_posterior_mean,
            jnp.zeros_like(aleatoric_std),
            aleatoric_std,
        )
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
