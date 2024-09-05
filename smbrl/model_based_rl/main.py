import os.path
import pickle
from typing import Tuple, NamedTuple, List

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
import matplotlib.pyplot as plt
from brax.envs import Env as BraxEnv
from brax.envs import State
from bsm.bayesian_regression.bayesian_regression_model import BayesianRegressionModel
from bsm.utils.type_aliases import ModelState
from bsm.utils.normalization import Data
from distrax import Distribution, Normal
from flax import struct
from jaxtyping import Key, Array, PyTree, Float
from mbpo.systems.rewards.base_rewards import Reward, RewardParams
from optax import Schedule, constant_schedule

from smbrl.model_based_rl.active_exploration_system import ExplorationSystem, ExplorationReward, ExplorationDynamics
from smbrl.optimizer.icem import iCemParams, iCemTO, AbstractCost
from smbrl.utils.plot_3d_trajectory import create_3d_trajectory_plot
from smbrl.utils.utils import create_folder, ExplorationTrajectory

jax.config.update("jax_enable_x64", True)


class Task(NamedTuple):
    reward: Reward
    name: str


class ModelBasedAgent:
    def __init__(self,
                 env: BraxEnv,
                 model: BayesianRegressionModel,
                 episode_length: int,
                 action_repeat: int,
                 cost_fn: AbstractCost,
                 test_tasks: List[Task],
                 predict_difference: bool = True,
                 num_training_steps: Schedule = constant_schedule(1000),
                 icem_horizon: int = 20,
                 icem_params: iCemParams = iCemParams(),
                 ):
        self.env = env
        self.model = model
        self.episode_length = episode_length
        self.action_repeat = action_repeat
        self.cost_fn = cost_fn
        self.test_tasks = test_tasks

        self.predict_difference = predict_difference
        self.num_training_steps = num_training_steps
        self.icem_horizon = icem_horizon
        self.icem_params = icem_params

    def train_dynamics_model(self,
                             model_state: ModelState,
                             data: Data,
                             episode_idx: int) -> ModelState:
        model_state = self.model.fit_model(data=data,
                                           num_training_steps=self.num_training_steps(episode_idx),
                                           model_state=model_state)
        return model_state

    def test_a_task(self,
                    model_state: ModelState,
                    key: Key[Array, '2'],
                    task: Task,
                    ) -> Tuple[State, Float[Array, '... action_dim']]:
        exploration_dynamics = ExplorationDynamics(x_dim=self.env.observation_size,
                                                   u_dim=self.env.action_size,
                                                   model=self.model,
                                                   )
        learned_system = ExplorationSystem(
            dynamics=exploration_dynamics,
            reward=task.reward,
        )
        key, subkey = jr.split(key)

        optimizer = iCemTO(
            horizon=self.icem_horizon,
            action_dim=self.env.action_size,
            key=subkey,
            opt_params=self.icem_params,
            system=learned_system,
            cost_fn=self.cost_fn,
        )

        key, subkey = jr.split(key)
        optimizer_state = optimizer.init(key=subkey)

        dynamics_params = optimizer_state.system_params.dynamics_params.replace(model_state=model_state)
        system_params = optimizer_state.system_params.replace(dynamics_params=dynamics_params)
        optimizer_state = optimizer_state.replace(system_params=system_params)

        env_state = self.env.reset(rng=key)

        collected_states = [env_state]
        actions = []

        for i in range(self.episode_length):
            action, optimizer_state = optimizer.act(env_state.obs, optimizer_state)
            for _ in range(self.action_repeat):
                env_state = self.env.step(env_state, action)
            collected_states.append(env_state)
            actions.append(action)

        collected_states = jt.map(lambda *xs: jnp.stack(xs), *collected_states)
        actions = jt.map(lambda *xs: jnp.stack(xs), *actions)
        self.plot_trajectories(
            states=collected_states.obs,
            actions=actions,
            rewards=collected_states.reward,
            title=task.name
        )
        return collected_states, actions

    def simulate_on_true_env(self,
                             model_state: ModelState,
                             key: Key[Array, '2'], ) -> Tuple[
        PyTree[Array, 'episode_length ...'], Float[Array, 'episode_length action_dim'], Float[
            Array, 'episode_length 1']]:
        exploration_reward = ExplorationReward(x_dim=self.env.observation_size,
                                               u_dim=self.env.action_size, )
        exploration_dynamics = ExplorationDynamics(x_dim=self.env.observation_size,
                                                   u_dim=self.env.action_size,
                                                   model=self.model,
                                                   )
        learned_system = ExplorationSystem(
            dynamics=exploration_dynamics,
            reward=exploration_reward,
        )
        key, subkey = jr.split(key)

        optimizer = iCemTO(
            horizon=self.icem_horizon,
            action_dim=self.env.action_size,
            key=subkey,
            opt_params=self.icem_params,
            system=learned_system,
            cost_fn=self.cost_fn,
        )

        key, subkey = jr.split(key)
        optimizer_state = optimizer.init(key=subkey)

        dynamics_params = optimizer_state.system_params.dynamics_params.replace(model_state=model_state)
        system_params = optimizer_state.system_params.replace(dynamics_params=dynamics_params)
        optimizer_state = optimizer_state.replace(system_params=system_params)

        env_state = self.env.reset(rng=key)

        collected_states = [env_state]
        actions = []
        extrinsic_rewards = []
        # TODO: Should implement treatment of done flags
        for i in range(self.episode_length):
            action, optimizer_state = optimizer.act(env_state.obs, optimizer_state)
            for _ in range(self.action_repeat):
                env_state = self.env.step(env_state, action)
            # Calculate extrinsic reward
            z = jnp.concatenate([env_state.obs, action])
            dist_f, dist_y = self.model.posterior(z, model_state)
            epistemic_std, aleatoric_std = dist_f.stddev(), dist_y.aleatoric_std()
            extrinsic_reward = learned_system.dynamics.get_intrinsic_reward(epistemic_std=epistemic_std,
                                                                            aleatoric_std=aleatoric_std)
            extrinsic_rewards.append(extrinsic_reward)
            collected_states.append(env_state)
            actions.append(action)

        collected_states = jt.map(lambda *xs: jnp.stack(xs), *collected_states)
        actions = jt.map(lambda *xs: jnp.stack(xs), *actions)
        extrinsic_rewards = jt.map(lambda *xs: jnp.stack(xs), *extrinsic_rewards)
        return collected_states, actions, extrinsic_rewards

    def from_collected_transitions_to_data(self,
                                           collected_states: PyTree[Array, 'episode_length ...'],
                                           actions: Float[Array, 'episode_length action_dim']) -> Data:
        # TODO: Isn't this wrong, if we have a done flag in collected_states?
        states = collected_states.obs[:-1]
        next_states = collected_states.obs[1:]
        inputs = jnp.concatenate([states, actions], axis=-1)
        if self.predict_difference:
            outputs = next_states - states
        else:
            outputs = next_states
        return Data(inputs=inputs, outputs=outputs)

    def plot_trajectories(self,
                          states: Float[Array, 'episode_length observation_dim'],
                          actions: Float[Array, 'episode_length action_dim'],
                          rewards: Float[Array, 'episode_length 1'],
                          title: str = 'Exploration trajectory'):
        fig, axs = plt.subplots(1, 3, figsize=(8, 2))
        fig.suptitle(title)
        axs[0].plot(states)
        axs[0].set_title('Observation')
        axs[1].plot(actions)
        axs[1].set_title('Action')
        axs[2].plot(rewards[1:])
        axs[2].set_title('Reward')
        plt.tight_layout()
        plt.show()

        print(title + ': ' + f'Maximal velocity value: {jnp.max(jnp.stack(states)[:, -1])}')
        print(title + ': ' + f'Minimal velocity value: {jnp.min(jnp.stack(states)[:, -1])}')

    def do_episode(self,
                   model_state: ModelState,
                   episode_idx: int,
                   data: Data,
                   key: Key[Array, '2'],
                   plotting: bool = True,
                   folder_name: str = 'experiment_2024'
                   ) -> (ModelState, Data):
        if episode_idx > 0:
            # If we collected some data already then we train dynamics model and the policy
            print(f'Start of dynamics training')
            model_state = self.train_dynamics_model(model_state=model_state,
                                                    data=data,
                                                    episode_idx=episode_idx)

        # We collect new data with the current policy
        print(f'Start of data collection')
        exploration_states, exploration_actions, exploration_rewards = self.simulate_on_true_env(
            model_state=model_state,
            key=key)

        task_outputs = []
        for task in self.test_tasks:
            task_outputs.append(self.test_a_task(model_state=model_state, key=key, task=task))

        if plotting:
            self.plot_trajectories(exploration_states.obs, exploration_actions, exploration_rewards,
                                   title=f'Exploration trajectory Episode {episode_idx}')

            th = jnp.arctan2(exploration_states.obs[:, 1], exploration_states.obs[:, 0])
            omega = exploration_states.obs[:, 2]
            u = exploration_actions[:, 0]
            trajectory = jnp.stack([th[:-1], omega[:-1], u], axis=-1)
            create_3d_trajectory_plot(trajectory)
        new_data = self.from_collected_transitions_to_data(exploration_states, exploration_actions)
        data = Data(inputs=jnp.concatenate([data.inputs, new_data.inputs]),
                    outputs=jnp.concatenate([data.outputs, new_data.outputs]), )

        # We save everything with pickle
        folder_name = os.path.join(folder_name, f'episode_{episode_idx}')
        create_folder(folder_name)

        # Saving data to a pickle file
        with open(os.path.join(folder_name, 'data.pkl'), 'wb') as file:
            pickle.dump(data, file)

        with open(os.path.join(folder_name, 'model_state.pkl'), 'wb') as file:
            pickle.dump(model_state, file)

        with open(os.path.join(folder_name, 'exploration_trajectory.pkl'), 'wb') as file:
            pickle.dump(ExplorationTrajectory(states=exploration_states, actions=exploration_actions,
                                              rewards=exploration_rewards, ), file)

        with open(os.path.join(folder_name, 'task_outputs.pkl'), 'wb') as file:
            pickle.dump(task_outputs, file)

        return model_state, data

    def run_episodes(self,
                     num_episodes: int,
                     key: Key[Array, '2'] = jr.PRNGKey(0),
                     model_state: ModelState | None = None,
                     data: Data | None = None,
                     folder_name: str = 'experiment_2024') -> (ModelState, Data):
        create_folder(folder_name)
        if data is None:
            data = Data(inputs=jnp.zeros(shape=(0, self.env.observation_size + self.env.action_size)),
                        outputs=jnp.zeros(shape=(0, self.env.observation_size)))

        for episode_idx in range(num_episodes):
            key, subkey = jr.split(key)
            print(f'Starting with Episode {episode_idx}')
            model_state, data = self.do_episode(model_state=model_state,
                                                episode_idx=episode_idx,
                                                data=data,
                                                key=subkey,
                                                folder_name=folder_name)
            print(f'End of Episode {episode_idx}')
        return model_state, data


if __name__ == '__main__':
    from smbrl.envs.pendulum import PendulumEnv
    from smbrl.dynamics_models.gps import ARD
    from smbrl.playground.pendulum_icem import VelocityBound
    from bsm.bayesian_regression.gaussian_processes.gaussian_processes import GaussianProcess

    env = PendulumEnv()

    model = GaussianProcess(
        kernel=ARD(input_dim=env.observation_size + env.action_size),
        input_dim=env.observation_size + env.action_size,
        output_dim=env.observation_size,
        output_stds=1e-3 * jnp.ones(shape=(env.observation_size,)),
        logging_wandb=False)

    icem_horizon = 20


    @chex.dataclass
    class PendulumRewardParams:
        control_cost: chex.Array = struct.field(default_factory=lambda: jnp.array(0.02))
        angle_cost: chex.Array = struct.field(default_factory=lambda: jnp.array(1.0))
        target_angle: chex.Array = struct.field(default_factory=lambda: jnp.array(0.0))


    class PendulumSwingUp(Reward):
        def __init__(self):
            super().__init__(x_dim=3, u_dim=1)

        def __call__(self,
                     x: chex.Array,
                     u: chex.Array,
                     reward_params: PendulumRewardParams,
                     x_next: chex.Array | None = None) -> Tuple[Distribution, RewardParams]:
            chex.assert_shape(x, (self.x_dim,))
            chex.assert_shape(u, (self.u_dim,))
            chex.assert_shape(x_next, (self.x_dim,))
            # get intrinsic reward out
            theta, omega = jnp.arctan2(x[1], x[0]), x[-1]
            target_angle = reward_params.target_angle
            diff_th = theta - target_angle
            diff_th = ((diff_th + jnp.pi) % (2 * jnp.pi)) - jnp.pi
            reward = -(reward_params.angle_cost * diff_th ** 2 +
                       0.1 * omega ** 2) - reward_params.control_cost * u ** 2
            reward = reward.squeeze()
            return Normal(loc=reward, scale=jnp.zeros_like(reward)), reward_params

        def init_params(self, key: chex.PRNGKey) -> PendulumRewardParams:
            return PendulumRewardParams()


    icem_params = iCemParams(
        num_particles=10,
        num_samples=500,
        alpha=0.2,
        num_steps=5,
        exponent=2,
        lambda_constraint=1e6
    )

    agent = ModelBasedAgent(
        env=PendulumEnv(),
        model=model,
        episode_length=50,
        action_repeat=2,
        # cost_fn=None,
        cost_fn=VelocityBound(horizon=icem_horizon,
                              max_abs_velocity=6.0 - 10 ** (-3),
                              violation_eps=1e-3, ),
        test_tasks=[Task(reward=PendulumSwingUp(), name='Swing up')],
        predict_difference=True,
        num_training_steps=constant_schedule(1000),
        icem_horizon=icem_horizon,
        icem_params=icem_params
    )

    model_state = model.init(jr.PRNGKey(0))
    agent.run_episodes(num_episodes=20,
                       key=jr.PRNGKey(0),
                       model_state=model_state,
                       folder_name='Cost30Aug2024'
                       )
