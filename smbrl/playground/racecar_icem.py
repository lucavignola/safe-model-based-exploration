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

from smbrl.envs.racecar import RCCar
from smbrl.optimizer.icem import iCemTO, iCemParams, AbstractCost

DT = 0.03
MARGIN_FACTOR = 20


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


class RaceCarSystem(System):
    def __init__(self):
        super().__init__(dynamics=DummyDynamics(x_dim=5, u_dim=1),
                         reward=DummyReward(x_dim=5, u_dim=1))
        self.brax_env = RCCar(dt=DT, margin_factor=MARGIN_FACTOR)

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


class RadiusBound(AbstractCost):

    def __init__(self, *args, max_radius: float = 2.0, violation_eps=1e-3, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_radius = max_radius
        self.violation_eps = violation_eps

    def __call__(self,
                 states: Float[Array, 'horizon observation_dim'],
                 actions: Float[Array, 'horizon action_dim'],
                 ) -> Scalar:
        dist_to_origin = jnp.linalg.norm(states[:, :2], ord=jnp.inf, axis=-1)

        trajectory_constraint = jnp.maximum(dist_to_origin - (self.max_radius - self.violation_eps), 0.0)
        assert trajectory_constraint.shape == (self.horizon,)
        return jnp.mean(trajectory_constraint)


class RadiusBoundBinary(AbstractCost):

    def __init__(self, *args, max_radius: float = 2.0, violation_eps=1e-3, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_radius = max_radius
        self.violation_eps = violation_eps

    def __call__(self,
                 states: Float[Array, 'horizon observation_dim'],
                 actions: Float[Array, 'horizon action_dim'],
                 ) -> Scalar:
        dist_to_origin = jnp.linalg.norm(states[:, :2], ord=jnp.inf, axis=-1)

        trajectory_constraint = dist_to_origin > (self.max_radius - self.violation_eps)
        assert trajectory_constraint.shape == (self.horizon,)
        return jnp.mean(trajectory_constraint)


if __name__ == '__main__':

    action_repeat = 2
    horizon = 20
    optimizer = iCemTO(
        horizon=horizon,
        action_dim=2,
        key=jr.PRNGKey(0),
        opt_params=iCemParams(exponent=2.0,
                              num_samples=500,
                              alpha=0.2,
                              num_steps=5,
                              num_particles=1, ),
        system=ActionRepeatWrapper(action_repeat=action_repeat, system=RaceCarSystem()),
        cost_fn=RadiusBoundBinary(horizon=horizon),
    )

    system = ActionRepeatWrapper(action_repeat=action_repeat, system=RaceCarSystem())

    optimizer_state = optimizer.init(key=jr.PRNGKey(1))
    system_params = system.init_params(key=jr.PRNGKey(2))
    obs = RCCar(dt=DT, margin_factor=MARGIN_FACTOR).reset(jr.PRNGKey(0)).obs

    all_obs = []
    all_actions = []
    all_rewards = []

    times = []
    first_times = time.time()
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

    print(f'Total time {time.time() - first_times:.2f}')

    fig, axs = plt.subplots(1, 4, figsize=(8, 2))
    all_obs = jnp.array(all_obs)
    for i in range(len(all_obs[0])):
        axs[0].plot(all_obs[:, i], label=f'State {i}')
    axs[0].set_title('Observation')
    axs[0].legend(fontsize=5)

    axs[1].plot(all_actions)
    axs[1].set_title('Action')
    axs[2].plot(all_rewards)
    axs[2].set_title('Reward')
    axs[3].plot(times[2:])
    axs[3].set_title('Time')
    plt.tight_layout()
    plt.show()

    print(f'Maximal x position: {jnp.max(jnp.stack(all_obs)[:, 0])}')
    print(f'Minimal x position: {jnp.min(jnp.stack(all_obs)[:, 0])}')

    from smbrl.envs.racecar import plot_rc_trajectory

    all_actions = jnp.stack(all_actions)
    plot_rc_trajectory(all_obs, all_actions, encode_angle=True)
