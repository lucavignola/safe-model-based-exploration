import time
from typing import Tuple

import chex
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from brax.envs.base import State
from distrax import Distribution
from distrax import Normal
from jaxtyping import Float, Array, Scalar
from mbpo.systems import DynamicsParams, RewardParams
from mbpo.systems.base_systems import System, SystemParams, SystemState
from mbpo.systems.dynamics.base_dynamics import Dynamics
from mbpo.systems.rewards.base_rewards import Reward

from smbrl.envs.pendulum import PendulumEnv
from smbrl.optimizer.icem import iCemTO, iCemParams, AbstractCost


class DummyDynamics(Dynamics):
    def __init__(self, x_dim, u_dim):
        super().__init__(x_dim=x_dim, u_dim=u_dim)

    def next_state(self,
                   x: chex.Array,
                   u: chex.Array,
                   dynamics_params: DynamicsParams) -> Tuple[Distribution, DynamicsParams]:
        return Normal(0, 0.01), dynamics_params

    def init_params(self, key: chex.PRNGKey) -> DynamicsParams:
        return 0


class DummyReward(Reward):
    def __init__(self, x_dim, u_dim):
        super().__init__(x_dim, u_dim)

    def init_params(self, key: chex.PRNGKey) -> RewardParams:
        return 0

    def __call__(self,
                 x: chex.Array,
                 u: chex.Array,
                 reward_params: RewardParams,
                 x_next: chex.Array | None = None) -> Tuple[Distribution, RewardParams]:
        return Normal(0, 0.01), reward_params


class PendulumSystem(System):
    def __init__(self):
        super().__init__(dynamics=DummyDynamics(x_dim=3, u_dim=1),
                         reward=DummyReward(x_dim=3, u_dim=1))
        self.brax_env = PendulumEnv()

    def step(self,
             x: chex.Array,
             u: chex.Array,
             system_params: SystemParams[DynamicsParams, RewardParams],
             ) -> SystemState:
        """

        :param x: current state of the system
        :param u: current action of the system
        :param system_params: parameters of the system
        :return: Tuple of next state, reward, updated system parameters
        """
        state = State(pipeline_state=None,
                      obs=x,
                      reward=jnp.array(0.0),
                      done=jnp.array(0.0), )

        next_state = self.brax_env.step(state, u)
        next_system_state = SystemState(x_next=next_state.obs,
                                        reward=next_state.reward,
                                        system_params=system_params,
                                        done=next_state.done)

        return next_system_state


class ActionRepeatWrapper(System):
    def __init__(self,
                 action_repeat: int,
                 system: System):
        super().__init__(dynamics=system.dynamics, reward=system.reward)
        self.action_repeat = action_repeat
        self.system = system

    def step(self,
             x: chex.Array,
             u: chex.Array,
             system_params: SystemParams[DynamicsParams, RewardParams],
             ) -> SystemState:
        total_reward = 0.0
        for _ in range(action_repeat):
            sys_state = self.system.step(x, u, system_params)
            x, reward, system_params = sys_state.x_next, sys_state.reward, sys_state.system_params
            total_reward += reward
        sys_state = sys_state.replace(reward=total_reward)
        return sys_state


class VelocityBound(AbstractCost):

    def __init__(self, *args, max_abs_velocity: float = 4.0, violation_eps=1e-3, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_abs_velocity = max_abs_velocity
        self.violation_eps = violation_eps

    def __call__(self,
                 states: Float[Array, 'horizon observation_dim'],
                 actions: Float[Array, 'horizon action_dim'],
                 ) -> Scalar:
        angular_velocity = states[:, -1]
        trajectory_constraint = jnp.maximum(jnp.abs(angular_velocity) - self.max_abs_velocity, -self.violation_eps)
        assert trajectory_constraint.shape == (self.horizon,)
        return jnp.mean(trajectory_constraint)


if __name__ == '__main__':

    action_repeat = 2
    horizon = 20
    safe_exploration = True

    cost_fn = None
    if safe_exploration:
        cost_fn = VelocityBound(horizon=horizon,
                                max_abs_velocity=6.0,
                                violation_eps=0.0, )

    optimizer = iCemTO(
        horizon=horizon,
        action_dim=1,
        key=jr.PRNGKey(0),
        opt_params=iCemParams(exponent=2.0,
                              num_samples=500,
                              alpha=0.2,
                              num_steps=5,
                              num_particles=1, ),
        system=ActionRepeatWrapper(action_repeat=action_repeat, system=PendulumSystem()),
        cost_fn=cost_fn,
    )

    system = PendulumSystem()

    optimizer_state = optimizer.init(key=jr.PRNGKey(1))
    system_params = system.init_params(key=jr.PRNGKey(2))
    obs = jnp.array([-1.0, 0.0, 0.0])

    all_obs = []
    all_actions = []
    all_rewards = []

    times = []

    for i in range(100 // action_repeat):
        start_time = time.time()
        action, optimizer_state = optimizer.act(obs, optimizer_state)
        for _ in range(action_repeat):
            sys_state = system.step(obs, action, system_params)
            obs, reward, system_params = sys_state.x_next, sys_state.reward, sys_state.system_params
        all_obs.append(obs)
        all_actions.append(action)
        all_rewards.append(reward)
        end_time = time.time()
        times.append(end_time - start_time)

    fig, axs = plt.subplots(1, 4, figsize=(8, 2))
    axs[0].plot(all_obs)
    axs[0].set_title('Observation')
    axs[1].plot(all_actions)
    axs[1].set_title('Action')
    axs[2].plot(all_rewards)
    axs[2].set_title('Reward')
    axs[3].plot(times[2:])
    axs[3].set_title('Time')
    plt.tight_layout()
    plt.show()

    print(f'Maximal velocity value: {jnp.max(jnp.stack(all_obs)[:, -1])}')
    print(f'Minimal velocity value: {jnp.min(jnp.stack(all_obs)[:, -1])}')

    import numpy as np

    total_reward = np.sum(np.array(all_rewards))
    print(f'Total reward: {total_reward}')