from typing import Generic, Tuple

import chex
import jax.numpy as jnp
import jax.random
import jax.random as jr
from brax.envs.base import Env, State
from bsm.statistical_model import StatisticalModel
from bsm.utils.type_aliases import ModelState
from distrax import Distribution, Normal
from jaxtyping import Float, Array, Scalar
from mbpo.systems.base_systems import SystemParams, SystemState, System
from mbpo.systems.dynamics.base_dynamics import Dynamics
from mbpo.systems.dynamics.base_dynamics import DynamicsParams as DummyDynamicsParams
from mbpo.systems.rewards.base_rewards import Reward, RewardParams
from smbrl.envs.cartpole_lenart import CartPoleDynamicsParams
from smbrl.envs.pendulum import PendulumDynamicsParams


@chex.dataclass
class DynamicsParams(Generic[ModelState, DummyDynamicsParams]):
    key: chex.PRNGKey
    model_state: ModelState


class ExplorationDynamics(Dynamics, Generic[ModelState]):
    def __init__(self,
                 x_dim: int,
                 u_dim: int,
                 model: StatisticalModel,
                 use_log: bool = True,
                 scale_with_aleatoric_std: bool = True,
                 aleatoric_noise_in_prediction: bool = True,
                 predict_difference: bool = True,
                 use_mean_dynamics: bool = False,
                 prior_knowledge: str = "none",
                 prior_num_steps: int = 1,
                 ):
        Dynamics.__init__(self, x_dim=x_dim, u_dim=u_dim)
        self.model = model
        self.use_log = use_log
        self.scale_with_aleatoric_std = scale_with_aleatoric_std
        self.aleatoric_noise_in_prediction = aleatoric_noise_in_prediction
        self.predict_difference = predict_difference
        self.use_mean_dynamics = use_mean_dynamics
        self.prior_knowledge = prior_knowledge
        self.prior_num_steps = prior_num_steps

    def init_params(self, key: chex.PRNGKey) -> DynamicsParams:
        param_key, model_state_key = jr.split(key, 2)
        model_state = self.model.init(model_state_key)
        return DynamicsParams(key=key, model_state=model_state)

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
        if self.prior_knowledge == "none":
            pred = self.model(z, dynamics_params.model_state)
            known_action_effect = jnp.zeros_like(x)
        elif self.prior_knowledge == "pendulum":
            pred = self.model(x, dynamics_params.model_state)
            known_action_effect = self.pendulum_prior(x, u, self.predict_difference)
        elif self.prior_knowledge == "cartpole":
            pred = self.model(x, dynamics_params.model_state)
            known_action_effect = self.cartpole_prior(x, u, self.predict_difference)
        else:
            raise NotImplementedError(f'Unknown prior knowledge {self.prior_knowledge}')
        epistemic_std, aleatoric_std = pred.epistemic_std, pred.aleatoric_std
        beta = pred.statistical_model_state.beta
        x_next = x
        if self.predict_difference:
            if self.use_mean_dynamics:
                x_next += pred.mean
            else:
                x_next += pred.mean + beta * epistemic_std * jr.normal(key=key_sample_x_next, shape=pred.mean.shape)
        else:
            if self.use_mean_dynamics:
                x_next = pred.mean
            else:
                x_next = pred.mean + beta * epistemic_std * jr.normal(key=key_sample_x_next, shape=pred.mean.shape)
        x_next += known_action_effect
        intrinsic_reward = self.get_intrinsic_reward(epistemic_std, aleatoric_std)
        intrinsic_reward = jnp.atleast_1d(intrinsic_reward)

        if not self.aleatoric_noise_in_prediction:
            aleatoric_std = 0 * aleatoric_std
        # add intrinsic reward to the next state
        x_next_with_reward = jnp.concatenate([x_next, intrinsic_reward], axis=-1)
        aleatoric_std_with_reward = jnp.concatenate([aleatoric_std, jnp.zeros_like(intrinsic_reward)], axis=-1)
        new_dynamics_params = dynamics_params.replace(key=next_key)
        return Normal(loc=x_next_with_reward, scale=aleatoric_std_with_reward), new_dynamics_params

    def pendulum_prior(self, x: chex.Array, u: chex.Array, predict_difference: bool) -> chex.Array:
        return pendulum_known_action_effect(x, u, predict_difference, self.prior_num_steps)

    def cartpole_prior(self, x: chex.Array, u: chex.Array, predict_difference: bool) -> chex.Array:
        return cartpole_known_action_effect(x, u, predict_difference, self.prior_num_steps)


class HallucinatedExplorationDynamics(ExplorationDynamics[ModelState]):
    """H-UCRL dynamics with augmented controls (real action, eta).

    The hallucinated part eta has one entry per state dimension and shifts the
    model prediction inside the beta-scaled epistemic confidence interval.
    """

    def __init__(self,
                 x_dim: int,
                 u_dim: int,
                 model: StatisticalModel,
                 use_log: bool = True,
                 scale_with_aleatoric_std: bool = True,
                 aleatoric_noise_in_prediction: bool = True,
                 predict_difference: bool = True,
                 use_mean_dynamics: bool = False,
                 prior_knowledge: str = "none",
                 prior_num_steps: int = 1,
                 ):
        real_u_dim = u_dim
        super().__init__(
            x_dim=x_dim,
            u_dim=real_u_dim + x_dim,
            model=model,
            use_log=use_log,
            scale_with_aleatoric_std=scale_with_aleatoric_std,
            aleatoric_noise_in_prediction=aleatoric_noise_in_prediction,
            predict_difference=predict_difference,
            use_mean_dynamics=use_mean_dynamics,
            prior_knowledge=prior_knowledge,
            prior_num_steps=prior_num_steps,
        )
        self.real_u_dim = real_u_dim

    def split_action(self, u: chex.Array) -> tuple[chex.Array, chex.Array]:
        real_action = u[..., :self.real_u_dim]
        hallucinated_action = jnp.clip(u[..., self.real_u_dim:], -1.0, 1.0)
        return real_action, hallucinated_action

    def next_state(self,
                   x: chex.Array,
                   u: chex.Array,
                   dynamics_params: DynamicsParams) -> Tuple[Distribution, DynamicsParams]:
        assert x.shape == (self.x_dim,) and u.shape == (self.u_dim,)
        real_action, hallucinated_action = self.split_action(u)
        z = jnp.concatenate([x, real_action])
        next_key, _ = jr.split(dynamics_params.key)
        if self.prior_knowledge == "none":
            pred = self.model(z, dynamics_params.model_state)
            known_action_effect = jnp.zeros_like(x)
        elif self.prior_knowledge == "pendulum":
            pred = self.model(x, dynamics_params.model_state)
            known_action_effect = self.pendulum_prior(x, real_action, self.predict_difference)
        elif self.prior_knowledge == "cartpole":
            pred = self.model(x, dynamics_params.model_state)
            known_action_effect = self.cartpole_prior(x, real_action, self.predict_difference)
        else:
            raise NotImplementedError(f'Unknown prior knowledge {self.prior_knowledge}')
        epistemic_std, aleatoric_std = pred.epistemic_std, pred.aleatoric_std
        beta = pred.statistical_model_state.beta
        optimistic_shift = beta * epistemic_std * hallucinated_action
        if self.predict_difference:
            x_next = x + pred.mean + optimistic_shift
        else:
            x_next = pred.mean + optimistic_shift
        x_next += known_action_effect
        intrinsic_reward = self.get_intrinsic_reward(epistemic_std, aleatoric_std)
        intrinsic_reward = jnp.atleast_1d(intrinsic_reward)

        if not self.aleatoric_noise_in_prediction:
            aleatoric_std = 0 * aleatoric_std
        x_next_with_reward = jnp.concatenate([x_next, intrinsic_reward], axis=-1)
        aleatoric_std_with_reward = jnp.concatenate([aleatoric_std, jnp.zeros_like(intrinsic_reward)], axis=-1)
        new_dynamics_params = dynamics_params.replace(key=next_key)
        return Normal(loc=x_next_with_reward, scale=aleatoric_std_with_reward), new_dynamics_params


class GroundTruthExplorationDynamics(ExplorationDynamics[ModelState]):
    """Planning dynamics backed by a noise-free copy of the real environment."""

    def __init__(self,
                 env: Env,
                 model: StatisticalModel,
                 action_repeat: int = 1,
                 use_log: bool = True,
                 scale_with_aleatoric_std: bool = True,
                 ):
        super().__init__(
            x_dim=env.observation_size,
            u_dim=env.action_size,
            model=model,
            use_log=use_log,
            scale_with_aleatoric_std=scale_with_aleatoric_std,
            aleatoric_noise_in_prediction=False,
            use_mean_dynamics=True,
        )
        self.env = env
        self.action_repeat = action_repeat

    def get_intrinsic_reward(self,
                             epistemic_std: Float[Array, '... observation_dim'],
                             aleatoric_std: Float[Array, '... observation_dim']) -> Scalar:
        del aleatoric_std
        return jnp.zeros((), dtype=epistemic_std.dtype)

    def next_state(self,
                   x: chex.Array,
                   u: chex.Array,
                   dynamics_params: DynamicsParams) -> Tuple[Distribution, DynamicsParams]:
        assert x.shape == (self.x_dim,) and u.shape == (self.u_dim,)
        env_state = State(
            pipeline_state=None,
            obs=x,
            reward=jnp.zeros(()),
            done=jnp.zeros(()),
        )
        for _ in range(self.action_repeat):
            env_state = self.env.step(env_state, u)

        intrinsic_reward = jnp.zeros((1,), dtype=env_state.obs.dtype)
        x_next_with_reward = jnp.concatenate([env_state.obs, intrinsic_reward])
        next_key, _ = jr.split(dynamics_params.key)
        new_dynamics_params = dynamics_params.replace(key=next_key)
        return Normal(
            loc=x_next_with_reward,
            scale=jnp.zeros_like(x_next_with_reward),
        ), new_dynamics_params


def pendulum_deterministic_next_state(x: chex.Array,
                                      u: chex.Array,
                                      dynamics_params: PendulumDynamicsParams | None = None) -> chex.Array:
    chex.assert_shape(x, (3,))
    chex.assert_shape(u, (1,))
    if dynamics_params is None:
        dynamics_params = PendulumDynamicsParams()

    th = jnp.arctan2(x[1], x[0])
    thdot = x[-1]
    dt = dynamics_params.dt
    torque = jnp.clip(u[0], -1.0, 1.0) * dynamics_params.max_torque

    thddot = (
        3.0 * dynamics_params.g / (2.0 * dynamics_params.l) * jnp.sin(th)
        + 3.0 / (dynamics_params.m * dynamics_params.l ** 2) * torque
    )
    newthdot = thdot + thddot * dt
    newthdot = jnp.clip(newthdot, -dynamics_params.max_speed, dynamics_params.max_speed)
    newth = th + newthdot * dt
    return jnp.asarray([jnp.cos(newth), jnp.sin(newth), newthdot]).reshape(-1)

def pendulum_known_action_effect(x: chex.Array,
                                 u: chex.Array,
                                 predict_difference: bool = True,
                                 num_steps: int = 1,
                                 dynamics_params: PendulumDynamicsParams | None = None) -> chex.Array:
    action_next_state = x
    passive_next_state = x
    for _ in range(num_steps):
        action_next_state = pendulum_deterministic_next_state(action_next_state, u, dynamics_params)
        passive_next_state = pendulum_deterministic_next_state(passive_next_state, jnp.zeros_like(u), dynamics_params)
    action_effect = action_next_state - passive_next_state
    if predict_difference:
        return action_effect
    return action_effect


def cartpole_from_obs_to_state(x: chex.Array) -> chex.Array:
    chex.assert_shape(x, (5,))
    position, cos_theta, sin_theta, linear_velocity, angular_velocity = x
    angle = jnp.arctan2(sin_theta, cos_theta)
    return jnp.array([position, angle, linear_velocity, angular_velocity])


def cartpole_from_state_to_obs(x: chex.Array) -> chex.Array:
    chex.assert_shape(x, (4,))
    position, angle, linear_velocity, angular_velocity = x
    return jnp.array([position, jnp.cos(angle), jnp.sin(angle), linear_velocity, angular_velocity])


def cartpole_ode(x: chex.Array,
                 u: chex.Array,
                 dynamics_params: CartPoleDynamicsParams | None = None) -> chex.Array:
    chex.assert_shape(x, (4,))
    chex.assert_shape(u, (1,))
    if dynamics_params is None:
        dynamics_params = CartPoleDynamicsParams()

    position, theta, linear_velocity, angular_velocity = x
    del position
    force = jnp.clip(u[0], -1.0, 1.0) * dynamics_params.max_torque
    cos_theta = jnp.cos(theta)
    sin_theta = jnp.sin(theta)

    m_1 = dynamics_params.m_1
    m_c = dynamics_params.m_c
    l_1 = dynamics_params.l_1
    g = dynamics_params.g

    mass_matrix = jnp.array([
        [m_1 + m_c, m_1 * l_1 * cos_theta],
        [m_1 * l_1 * cos_theta, m_1 * l_1 * l_1],
    ])
    rhs = jnp.array([
        force + sin_theta * l_1 * m_1 * angular_velocity * angular_velocity,
        -m_1 * l_1 * g * sin_theta,
    ])
    acceleration = jnp.linalg.inv(mass_matrix).dot(rhs)
    return jnp.array([linear_velocity, angular_velocity, acceleration[0], acceleration[1]])


def cartpole_deterministic_next_state(x: chex.Array,
                                      u: chex.Array,
                                      dynamics_params: CartPoleDynamicsParams | None = None) -> chex.Array:
    chex.assert_shape(x, (5,))
    chex.assert_shape(u, (1,))
    if dynamics_params is None:
        dynamics_params = CartPoleDynamicsParams()
    compressed_state = cartpole_from_obs_to_state(x)
    next_compressed_state = compressed_state + cartpole_ode(compressed_state, u, dynamics_params) * dynamics_params.dt
    return cartpole_from_state_to_obs(next_compressed_state)


def cartpole_known_action_effect(x: chex.Array,
                                 u: chex.Array,
                                 predict_difference: bool = True,
                                 num_steps: int = 1,
                                 dynamics_params: CartPoleDynamicsParams | None = None) -> chex.Array:
    action_next_state = x
    passive_next_state = x
    for _ in range(num_steps):
        action_next_state = cartpole_deterministic_next_state(action_next_state, u, dynamics_params)
        passive_next_state = cartpole_deterministic_next_state(passive_next_state, jnp.zeros_like(u), dynamics_params)
    action_effect = action_next_state - passive_next_state
    if predict_difference:
        return action_effect
    return action_effect



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

    def get_real_action(self, u: chex.Array) -> chex.Array:
        if hasattr(self.dynamics, 'real_u_dim'):
            return u[..., :self.dynamics.real_u_dim]
        return u

    def get_reward(self,
                   x: chex.Array,
                   u: chex.Array,
                   reward_params: RewardParams,
                   x_next: chex.Array,
                   key: jax.random.PRNGKey):
        # x_next includes the next state and the intrinsic reward
        chex.assert_shape(x_next, (self.x_dim + 1,))
        real_action = self.get_real_action(u)
        if isinstance(self.reward, ExplorationReward):
            # include the intrinsic reward in x_next
            reward_dist, new_reward_params = self.reward(x, real_action, reward_params, x_next)
        else:
            # Check if it's an SBSRL reward that needs intrinsic reward
            from smbrl.agent.sbsrl import SBSRLReward
            if isinstance(self.reward, SBSRLReward): #TODO: I think this if can be merged with the one above
                # SBSRL needs full x_next (with intrinsic reward) like ExplorationReward
                reward_dist, new_reward_params = self.reward(x, real_action, reward_params, x_next)
            else:
                # ignore the last state in x_next which is the intrinsic reward
                reward_dist, new_reward_params = self.reward(x, real_action, reward_params, x_next[:-1])
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
