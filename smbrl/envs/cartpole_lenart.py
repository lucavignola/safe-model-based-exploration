from functools import partial

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
from brax.envs.base import State, Env
from flax import struct
from jaxtyping import Float, Array, Scalar

from smbrl.utils.tolerance_reward import ToleranceReward


@chex.dataclass
class CartPoleDynamicsParams:
    max_torque: chex.Array = struct.field(default_factory=lambda: jnp.array(10.0))
    dt: chex.Array = struct.field(default_factory=lambda: jnp.array(0.05))
    g: chex.Array = struct.field(default_factory=lambda: jnp.array(9.81))
    m_1: chex.Array = struct.field(default_factory=lambda: jnp.array(1.0))
    m_c: chex.Array = struct.field(default_factory=lambda: jnp.array(1.0))
    l_1: chex.Array = struct.field(default_factory=lambda: jnp.array(1.0))


@chex.dataclass
class PendulumRewardParams:
    control_cost: chex.Array = struct.field(default_factory=lambda: jnp.array(0.01))
    angle_cost: chex.Array = struct.field(default_factory=lambda: jnp.array(1.0))
    pos_cost: chex.Array = struct.field(default_factory=lambda: jnp.array(1.0))
    vel_cost: chex.Array = struct.field(default_factory=lambda: jnp.array(0.1))
    target_angle: chex.Array = struct.field(default_factory=lambda: jnp.array(jnp.pi))


class CartPoleEnv(Env):
    def __init__(self,
                 reward_source: str = 'gym',
                 init_angle: float = 0.0,
                 ):
        self.dynamics_params = CartPoleDynamicsParams()
        self.reward_params = PendulumRewardParams()
        self.init_angle = init_angle
        # bound = 0.1
        # value_at_margin = 0.1
        # margin_factor = 10
        self.reward_source = reward_source  # 'dm-control' or 'gym'
        # self.tolerance_reward = ToleranceReward(bounds=(0.0, bound),
        #                                         margin=margin_factor * bound,
        #                                         value_at_margin=value_at_margin,
        #                                         sigmoid='long_tail')

    def reset(self,
              rng: jax.Array) -> State:
        state = State(pipeline_state=None,
                      obs=jnp.array([0.0, jnp.cos(self.init_angle), jnp.sin(self.init_angle), 0.0, 0.0]),
                      reward=jnp.array(0.0),
                      done=jnp.array(0.0), )
        return state

    @staticmethod
    def angle_to_cos_sin_representation(angle: Scalar) -> Float[Array, '2']:
        return jnp.array([jnp.cos(angle), jnp.sin(angle)])

    @staticmethod
    def cos_sin_to_angle_representation(cos_sin_angle: Float[Array, '2']) -> Scalar:
        return jnp.arctan2(cos_sin_angle[1], cos_sin_angle[0])

    def from_state_to_obs(self, obs: Float[Array, '4']) -> Float[Array, '5']:
        assert obs.shape == (4,)
        position, angle, linear_velocity, angular_velocity = obs[0], obs[1], obs[2], obs[3]
        cos_sin_angle = self.angle_to_cos_sin_representation(angle)
        return jnp.array([position, cos_sin_angle[0], cos_sin_angle[1], linear_velocity, angular_velocity])

    def from_obs_to_state(self, state: Float[Array, '5']) -> Float[Array, '4']:
        assert state.shape == (5,)
        position, cos, sin, linear_velocity, angular_velocity = state[0], state[1], state[2], state[3], state[4]
        angle = self.cos_sin_to_angle_representation(jnp.array([cos, sin]))
        return jnp.array([position, angle, linear_velocity, angular_velocity])

    def reward(self,
               x: Float[Array, 'observation_dim'],
               u: Float[Array, 'action_dim']) -> Float[Array, 'None']:
        x_compressed = self.from_obs_to_state(x)
        position, angle = x_compressed[0], x_compressed[1]
        linear_velocity, angular_velocity = x_compressed[2], x_compressed[3]

        target_angle = self.reward_params.target_angle
        diff_th = angle - target_angle
        diff_th = ((diff_th + jnp.pi) % (2 * jnp.pi)) - jnp.pi
        reward = -(self.reward_params.angle_cost * diff_th ** 2 + self.reward_params.pos_cost * position ** 2 +
                   self.reward_params.vel_cost * (
                           linear_velocity ** 2 + angular_velocity ** 2)) - self.reward_params.control_cost * u[
                     0] ** 2
        reward = reward.squeeze()
        return reward

    # def dm_reward(self,
    #               x: Float[Array, 'observation_dim'],
    #               u: Float[Array, 'action_dim']) -> Float[Array, 'None']:
    #     theta, omega = jnp.arctan2(x[1], x[0]), x[-1]
    #     target_angle = self.reward_params.target_angle
    #     diff_th = theta - target_angle
    #     diff_th = ((diff_th + jnp.pi) % (2 * jnp.pi)) - jnp.pi
    #     reward = self.tolerance_reward(jnp.sqrt(self.reward_params.angle_cost * diff_th ** 2 +
    #                                             0.1 * omega ** 2)) - self.reward_params.control_cost * u ** 2
    #     reward = reward.squeeze()
    #     return reward

    @partial(jax.jit, static_argnums=0)
    def step(self,
             state: State,
             action: jax.Array) -> State:
        x = state.obs
        chex.assert_shape(x, (self.observation_size,))
        chex.assert_shape(action, (self.action_size,))

        dt = self.dynamics_params.dt
        x_compressed = self.from_obs_to_state(x)

        dx = self.ode(x_compressed, action)

        next_x_compressed = x_compressed + dx * dt
        next_obs = self.from_state_to_obs(next_x_compressed)

        if self.reward_source == 'gym':
            next_reward = self.reward(x, action)
        elif self.reward_source == 'dm-control':
            raise NotImplementedError(f'{self.reward_source} not implemented')
        else:
            raise NotImplementedError(f'Unknown reward source {self.reward_source}')

        next_state = State(pipeline_state=state.pipeline_state,
                           obs=next_obs,
                           reward=next_reward,
                           done=state.done,
                           metrics=state.metrics,
                           info=state.info)
        return next_state

    def ode(self,
            state: Float[Array, '4'],
            action: Float[Array, '1'], ) -> Float[Array, '4']:

        g = self.dynamics_params.g
        m_1 = self.dynamics_params.m_1
        m_c = self.dynamics_params.m_c
        l_1 = self.dynamics_params.l_1

        u = jnp.clip(action, -1, 1) * self.dynamics_params.max_torque

        x, theta1, x_dot, theta1_dot = state

        cos_theta1 = jnp.cos(theta1)
        sin_theta1 = jnp.sin(theta1)

        A = jnp.array([
            [m_1 + m_c, m_1 * l_1 * cos_theta1],
            [m_1 * l_1 * cos_theta1, m_1 * l_1 * l_1]
        ])
        b = jnp.array([
            u[0] + sin_theta1 * l_1 * m_1 * theta1_dot * theta1_dot,
            -m_1 * l_1 * g * sin_theta1
        ])

        der = jnp.linalg.inv(A).dot(b)

        return jnp.array([x_dot, theta1_dot, der[0], der[1]])

    @property
    def dt(self):
        return self.dynamics_params.dt

    @property
    def observation_size(self) -> int:
        return 5

    @property
    def action_size(self) -> int:
        return 1

    def backend(self) -> str:
        return 'positional'


if __name__ == '__main__':
    from jax import jit
    import matplotlib.pyplot as plt

    jax.config.update("jax_enable_x64", True)

    env = CartPoleEnv(init_angle=jnp.pi)

    state = env.reset(jr.PRNGKey(0))
    action = jnp.array([0.0, ])

    obs = []
    rewards = []
    # step_fn = env.step
    step_fn = jit(env.step)
    for i in range(100):
        state = step_fn(state, action)
        obs.append(state.obs)
        rewards.append(state.reward)

    obs = jnp.array(obs)
    for i in range(env.observation_size):
        plt.plot(obs[:, i], label=f'State {i}')
    plt.legend()
    plt.show()
