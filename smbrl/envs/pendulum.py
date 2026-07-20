from functools import partial

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
from brax.envs.base import State, Env
from flax import struct
from jaxtyping import Float, Array

from smbrl.utils.tolerance_reward import ToleranceReward
from smbrl.utils.experiment_utils import tolerance


@chex.dataclass
class PendulumDynamicsParams:
    max_speed: chex.Array = struct.field(default_factory=lambda: jnp.array(8.0))
    max_torque: chex.Array = struct.field(default_factory=lambda: jnp.array(2.0))
    dt: chex.Array = struct.field(default_factory=lambda: jnp.array(0.05))
    g: chex.Array = struct.field(default_factory=lambda: jnp.array(9.81))
    m: chex.Array = struct.field(default_factory=lambda: jnp.array(1.0))
    l: chex.Array = struct.field(default_factory=lambda: jnp.array(1.0))


@chex.dataclass
class PendulumRewardParams:
    control_cost: chex.Array = struct.field(default_factory=lambda: jnp.array(0.02))
    angle_cost: chex.Array = struct.field(default_factory=lambda: jnp.array(1.0))
    target_angle: chex.Array = struct.field(default_factory=lambda: jnp.array(0.0))


def sparse_reward_function(theta, omega, u, action_cost, lower_bound: float = 0.5):
    reward = (
        tolerance(jnp.cos(theta), (lower_bound, 1.0), 0.1)
        * tolerance(omega, (-0.5, 0.5), 0.5)
        - action_cost * (1 - tolerance(u, (-0.5, 0.5), 0.1))
    )
    return reward


class PendulumEnv(Env):
    def __init__(self,
                 reward_source: str = 'gym',
                 add_process_noise: bool = False,
                 margin_factor: float = 10.0,
                 process_noise_scale: Float[Array, "physical_state_dim"] | float | None = None,
                 action_cost: float = 0.0,
                 sparse_reward_lower_bound: float = 0.5):
        if not -1.0 <= sparse_reward_lower_bound <= 1.0:
            raise ValueError('sparse_reward_lower_bound must lie in [-1, 1].')

        if process_noise_scale is None:
            process_noise_scale = jnp.zeros(2)
        else:
            process_noise_scale = jnp.asarray(process_noise_scale)
            if process_noise_scale.ndim == 0:
                process_noise_scale = jnp.full((2,), process_noise_scale)
            if process_noise_scale.shape != (2,):
                raise ValueError(
                    'process_noise_scale must be a scalar or have shape (2,) '
                    'for (theta, angular_velocity).'
                )

        self.dynamics_params = PendulumDynamicsParams()
        self.reward_params = PendulumRewardParams()
        bound = 0.1
        value_at_margin = 0.1
        margin_factor = margin_factor
        self.reward_source = reward_source  # 'dm-control' or 'gym'
        self.tolerance_reward = ToleranceReward(bounds=(0.0, bound),
                                                margin=margin_factor * bound,
                                                value_at_margin=value_at_margin,
                                                sigmoid='long_tail')
        self.add_process_noise = add_process_noise
        self.process_noise_scale = process_noise_scale
        self.action_cost = action_cost
        self.sparse_reward_lower_bound = sparse_reward_lower_bound

    def reset(self,
              rng: jax.Array) -> State:
        info = {'process_noise_key': rng} if self.add_process_noise else {}
        state = State(pipeline_state=None,
                      obs=jnp.array([-1.0, 0.0, 0.0]),
                      reward=jnp.array(0.0),
                      done=jnp.array(0.0),
                      info=info)
        return state

    def reward(self,
               x: Float[Array, 'observation_dim'],
               u: Float[Array, 'action_dim']) -> Float[Array, 'None']:
        theta, omega = jnp.arctan2(x[1], x[0]), x[-1]
        target_angle = self.reward_params.target_angle
        diff_th = theta - target_angle
        diff_th = ((diff_th + jnp.pi) % (2 * jnp.pi)) - jnp.pi
        reward = -(self.reward_params.angle_cost * diff_th ** 2 +
                   0.1 * omega ** 2) - self.reward_params.control_cost * u ** 2
        reward = reward.squeeze()
        return reward

    def dm_reward(self,
                  x: Float[Array, 'observation_dim'],
                  u: Float[Array, 'action_dim']) -> Float[Array, 'None']:
        theta, omega = jnp.arctan2(x[1], x[0]), x[-1]
        target_angle = self.reward_params.target_angle
        diff_th = theta - target_angle
        diff_th = ((diff_th + jnp.pi) % (2 * jnp.pi)) - jnp.pi
        reward = self.tolerance_reward(jnp.sqrt(self.reward_params.angle_cost * diff_th ** 2 +
                                                0.1 * omega ** 2)) - self.reward_params.control_cost * u ** 2
        reward = reward.squeeze()
        return reward

    def sparse_reward(self,
                      x: Float[Array, 'observation_dim'],
                      u: Float[Array, 'action_dim']) -> Float[Array, 'None']:
        theta, omega = jnp.arctan2(x[1], x[0]), x[-1]
        target_angle = self.reward_params.target_angle
        diff_th = theta - target_angle
        diff_th = ((diff_th + jnp.pi) % (2 * jnp.pi)) - jnp.pi
        reward = sparse_reward_function(
            theta,
            omega,
            u,
            self.action_cost,
            lower_bound=self.sparse_reward_lower_bound,
        )
        reward = reward.squeeze()
        return reward

    @partial(jax.jit, static_argnums=0)
    def step(self,
             state: State,
             action: jax.Array) -> State:
        x = state.obs
        chex.assert_shape(x, (self.observation_size,))
        chex.assert_shape(action, (self.action_size,))
        th = jnp.arctan2(x[1], x[0])
        thdot = x[-1]
        dt = self.dynamics_params.dt
        x_compressed = jnp.array([th, thdot])
        dx = self.ode(x_compressed, action)
        newth = th + dx[0] * dt
        newthdot = thdot + dx[-1] * dt
        newthdot = jnp.clip(newthdot, -self.dynamics_params.max_speed, self.dynamics_params.max_speed)
        info = state.info
        if self.add_process_noise:
            key = state.info['process_noise_key']
            key, subkey = jax.random.split(key)
            physical_noise = self.process_noise_scale * jr.normal(key=subkey, shape=(2,))
            newth = newth + physical_noise[0]
            newthdot = jnp.clip(
                newthdot + physical_noise[1],
                -self.dynamics_params.max_speed,
                self.dynamics_params.max_speed,
            )
            info = {**state.info, 'process_noise_key': key}
        next_obs = jnp.asarray([jnp.cos(newth), jnp.sin(newth), newthdot]).reshape(-1)
        if self.reward_source == 'gym':
            next_reward = self.reward(x, action)
        elif self.reward_source == 'dm-control':
            next_reward = self.dm_reward(x, action)
        elif self.reward_source == 'sparse':
            next_reward = self.sparse_reward(x, action)
        else:
            raise NotImplementedError(f'Unknown reward source {self.reward_source}')

        next_state = State(pipeline_state=state.pipeline_state,
                           obs=next_obs,
                           reward=next_reward,
                           done=state.done,
                           metrics=state.metrics,
                           info=info)
        return next_state

    def ode(self, x_compressed: chex.Array, u: chex.Array) -> chex.Array:
        chex.assert_shape(x_compressed, (self.observation_size - 1,))
        chex.assert_shape(u, (self.action_size,))
        thdot = x_compressed[-1]
        th = x_compressed[0]

        g = self.dynamics_params.g
        m = self.dynamics_params.m
        l = self.dynamics_params.l
        dt = self.dynamics_params.dt
        u = jnp.clip(u, -1, 1) * self.dynamics_params.max_torque
        newthddot = (3 * g / (2 * l) * jnp.sin(th) + 3.0 / (m * l ** 2) * u)
        newthdot = thdot + newthddot * dt
        newthdot = jnp.clip(newthdot, -self.dynamics_params.max_speed, self.dynamics_params.max_speed)
        return jnp.asarray([newthdot, newthddot])

    @property
    def dt(self):
        return self.dynamics_params.dt

    @property
    def observation_size(self) -> int:
        return 3

    @property
    def action_size(self) -> int:
        return 1

    def backend(self) -> str:
        return 'positional'
