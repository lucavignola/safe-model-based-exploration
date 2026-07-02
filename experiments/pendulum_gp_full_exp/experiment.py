import argparse
import os
import sys

import numpy as np

from smbrl.utils.experiment_utils import Logger, hash_dict, tolerance
from smbrl.envs.pendulum import sparse_reward_function


def experiment(
        project_name: str = 'ActSafeTest',
        alg_name: str = 'ActSafe',
        entity_name: str = 'lvignola-eth-z-rich',
        exp_hash: str = '42',
        num_offline_data: int = 100,
        seed: int = 0,
        num_particles: int = 10,
        num_samples: int = 500,
        alpha: float = 0.2,
        num_steps: int = 5,
        exponent: int = 2,
        lambda_constraint: float = 1e6,
        icem_horizon: int = 20,
        episode_length: int = 50,
        action_repeat: int = 2,
        max_abs_velocity: float = 6.0,
        action_cost: float = 0.0,
        sparse_task: bool = False,
        num_training_steps: int = 1_000,
        env_margin_factor: float = 10.0,
        process_noise_scale: float = 1e-3,
        reward_source: str = 'gym',
        use_optimism: bool = True,
        use_pessimism: bool = True,
        log_wandb: bool = True,
        logs_dir: str = 'runs',
        num_gpus: int = 0,
        function_norm: float = 1.0,
        num_elites: int = 50,
        violation_eps: float = 0.1,
        beta: float = 3.0,
        lambda_sigma: float = 1.0,
        uncertainty_eps: float = 1.0,
        uncertainty_decay_factor: float = 10.0,
        uncertainty_decay_mode: str = 'linear',
        uncertainty_constraint_threshold: float = 50.0,
        default_task_index: int = 0,
        actsafe_index: int = -1,
        wandb_notes: str = None,
        use_mean_dynamics: bool = False,
        aleatoric_noise_in_prediction: bool = True,
        prior_knowledge: str = "none",
):
    if num_gpus == 0:
        import os
        os.environ['JAX_PLATFORMS'] = 'cpu'

    import jax
    import jax.random as jr
    import jax.numpy as jnp
    import chex
    import wandb
    from smbrl.agent.actsafe import ActSafeAgent, SafeHUCRL, Task
    from smbrl.agent.sbsrl import SBSRLAgent
    from flax import struct
    from distrax import Distribution, Normal
    from typing import Tuple
    from brax.envs.base import State
    from optax import constant_schedule
    from bsm.utils.normalization import Data
    from mbpo.systems.rewards.base_rewards import Reward, RewardParams
    from smbrl.optimizer.icem import iCemParams
    from smbrl.envs.pendulum import PendulumEnv
    from smbrl.model_based_rl.active_exploration_system import pendulum_known_action_effect
    from smbrl.playground.pendulum_icem import VelocityBound
    from bsm.statistical_model import GPStatisticalModel
    from smbrl.dynamics_models.gps import ARD

    from mbrl.utils.offline_data import OfflineData

    env = PendulumEnv()
    if prior_knowledge not in ("none", "pendulum"):
        raise NotImplementedError(f'Unknown prior knowledge {prior_knowledge}')
    model_input_dim = env.observation_size + env.action_size if prior_knowledge == "none" else env.observation_size

    key = jr.PRNGKey(seed)
    key, offline_data_key = jr.split(key, 2)

    class PendulumOfflineData(OfflineData):
        def __init__(self, max_velocity: float, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.max_velocity = max_velocity

        def _sample_states(self,
                           key: chex.PRNGKey,
                           num_samples: int):
            key_angle, key_angular_velocity = jr.split(key)
            angles = jr.uniform(key_angle, shape=(num_samples,), minval=-jnp.pi, maxval=jnp.pi)
            cos, sin = jnp.cos(angles), jnp.sin(angles)
            angular_velocity = jr.uniform(key_angular_velocity, shape=(num_samples,),
                                          minval=-max_abs_velocity,
                                          maxval=max_abs_velocity)
            return jnp.stack([cos, sin, angular_velocity], axis=-1)

        def _sample_actions(self,
                            key: chex.PRNGKey,
                            num_samples: int):
            actions = jr.uniform(key, shape=(num_samples, 1), minval=-1, maxval=1)
            return actions

    if num_offline_data > 0:
        offline_data_gen = PendulumOfflineData(env=env, max_velocity=max_abs_velocity)
        if prior_knowledge == "none":
            transitions = offline_data_gen.sample_transitions(key=offline_data_key,
                                                              num_samples=num_offline_data)
            offline_inputs = jnp.concatenate([transitions.observation, transitions.action], axis=-1)
            offline_outputs = transitions.next_observation - transitions.observation
        elif prior_knowledge == "pendulum":
            offline_state_key, offline_action_key = jr.split(offline_data_key)
            offline_states = offline_data_gen.sample_states(key=offline_state_key,
                                                            num_samples=num_offline_data)
            offline_actions = offline_data_gen.sample_actions(key=offline_action_key,
                                                              num_samples=num_offline_data)
            offline_brax_state = State(pipeline_state=jnp.zeros(shape=(num_offline_data,)),
                                       obs=offline_states,
                                       reward=jnp.zeros(shape=(num_offline_data,)),
                                       done=jnp.zeros(shape=(num_offline_data,)))
            for _ in range(action_repeat):
                offline_brax_state = jax.vmap(env.step)(offline_brax_state, offline_actions)
            offline_inputs = offline_states
            known_action_effect = jax.vmap(
                pendulum_known_action_effect,
                in_axes=(0, 0, None, None),
            )(offline_states, offline_actions, True, action_repeat)
            offline_outputs = offline_brax_state.obs - offline_states - known_action_effect
        else:
            raise NotImplementedError(f'Unknown prior knowledge {prior_knowledge}')
        offline_data = Data(inputs=offline_inputs, outputs=offline_outputs)
    else:
        offline_data = None

    configs = dict(
        alg_name=alg_name,
        num_offline_data=num_offline_data,
        seed=seed,
        num_particles=num_particles,
        num_samples=num_samples,
        alpha=alpha,
        num_steps=num_steps,
        exponent=exponent,
        lambda_constraint=lambda_constraint,
        icem_horizon=icem_horizon,
        episode_length=episode_length,
        action_repeat=action_repeat,
        max_abs_velocity=max_abs_velocity,
        action_cost=action_cost,
        sparse_task=sparse_task,
        num_training_steps=num_training_steps,
        env_margin_factor=env_margin_factor,
        process_noise_scale=process_noise_scale,
        reward_source=reward_source,
        use_optimism=use_optimism,
        use_pessimism=use_pessimism,
        num_gpus=num_gpus,
        function_norm=function_norm,
        num_elites=num_elites,
        violation_eps=violation_eps,
        beta=beta,
        lambda_sigma=lambda_sigma,
        uncertainty_eps=uncertainty_eps,
        uncertainty_decay_factor=uncertainty_decay_factor,
        uncertainty_decay_mode=uncertainty_decay_mode,
        uncertainty_constraint_threshold=uncertainty_constraint_threshold,
        default_task_index=default_task_index,
        actsafe_index=actsafe_index,
        wandb_notes=wandb_notes,  # Add to config for visibility
        use_mean_dynamics=use_mean_dynamics,
        aleatoric_noise_in_prediction=aleatoric_noise_in_prediction,
        prior_knowledge=prior_knowledge,
    )

    model = GPStatisticalModel(
        kernel=ARD(input_dim=model_input_dim, length_scale=0.1),
        input_dim=model_input_dim,
        output_dim=env.observation_size,
        output_stds=1e-3 * jnp.ones(shape=(env.observation_size,)),
        logging_wandb=log_wandb,
        beta=jnp.ones(3) * beta,
        num_training_steps=constant_schedule(num_training_steps),
        lr_rate=1e-2,
        weight_decay=1e-3,
    )

    @chex.dataclass
    class PendulumRewardParams:
        control_cost: chex.Array = struct.field(default_factory=lambda: jnp.array(0.02))
        angle_cost: chex.Array = struct.field(default_factory=lambda: jnp.array(1.0))
        target_angle: chex.Array = struct.field(default_factory=lambda: jnp.array(0.0))

    class PendulumReward(Reward):
        def __init__(self, target_angle: float = 0.0, action_cost: float = 0.0, sparse_task: bool = False):
            super().__init__(x_dim=3, u_dim=1)
            self.target_angle = jnp.array(target_angle)
            self.action_cost = action_cost
            self.sparse_task = sparse_task

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
            if self.sparse_task:
                reward = sparse_reward_function(theta, omega, u, self.action_cost)
            else:
                reward = -(reward_params.angle_cost * diff_th ** 2 + 0.1 * omega ** 2) - reward_params.control_cost * u ** 2
            reward = reward.squeeze()
            return Normal(loc=reward, scale=jnp.zeros_like(reward)), reward_params

        def init_params(self, key: chex.PRNGKey) -> PendulumRewardParams:
            default_reward_params = PendulumRewardParams()
            return default_reward_params.replace(target_angle=self.target_angle)

    if alg_name == 'SafeHUCRL':
        alg = SafeHUCRL
    elif alg_name == 'ActSafe':
        alg = ActSafeAgent
    elif alg_name == 'HUCRL':
        alg = SafeHUCRL
        lambda_constraint = 0.0
    elif alg_name == 'OPAX':
        alg = ActSafeAgent
        lambda_constraint = 0.0
    elif alg_name == 'SBSRL':
        alg = SBSRLAgent
    else:
        raise NotImplementedError

    icem_params = iCemParams(
        num_particles=num_particles,
        num_samples=num_samples,
        num_elites=num_elites,
        alpha=alpha,
        num_steps=num_steps,
        exponent=exponent,
        lambda_constraint=lambda_constraint,
    )

    cost_fn = VelocityBound(horizon=icem_horizon,
                            max_abs_velocity=max_abs_velocity,
                            violation_eps=violation_eps, )

    true_env_process_noise_scale = process_noise_scale * jnp.ones(env.observation_size)
    true_env = PendulumEnv(margin_factor=env_margin_factor,
                           reward_source=reward_source,
                           add_process_noise=True,
                           process_noise_scale=true_env_process_noise_scale,
                           action_cost=action_cost,)

    # Create agent with appropriate parameters
    agent_kwargs = {
        'env': true_env,
        'model': model,
        'episode_length': episode_length,
        'action_repeat': action_repeat,
        'cost_fn': cost_fn,
        'test_tasks': [
            #Task(reward=PendulumReward(target_angle=jnp.pi), name='Keep down', env=env),
            Task(reward=PendulumReward(action_cost=action_cost, sparse_task=sparse_task), name='Swing up', env=true_env),
        ],
        'predict_difference': True,
        'num_training_steps': constant_schedule(num_training_steps),
        'icem_horizon': icem_horizon,
        'icem_params': icem_params,
        'log_to_wandb': log_wandb,
        'use_pessimism': use_pessimism,
        'use_optimism': use_optimism,
        'use_mean_dynamics': use_mean_dynamics,
        'aleatoric_noise_in_prediction': aleatoric_noise_in_prediction,
        'prior_knowledge': prior_knowledge,
    }

    # Add SBSRL-specific parameters if needed
    if alg_name == 'SBSRL':
        agent_kwargs.update({
            'action_cost': 0.0,  # Already included in the reward function
            'lambda_sigma': lambda_sigma,
            'uncertainty_eps': uncertainty_eps,
            'uncertainty_decay_factor': uncertainty_decay_factor,
            'uncertainty_decay_mode': uncertainty_decay_mode,
            'uncertainty_constraint_threshold': uncertainty_constraint_threshold,
            'default_task_index': default_task_index,
        })
    elif alg_name == 'ActSafe':
        agent_kwargs.update({
            'actsafe_index': actsafe_index,
            'actsafe_task_index': default_task_index,
        })

    agent = alg(**agent_kwargs)

    if log_wandb:
        import wandb

        # Setup wandb environment for Euler if needed
        if logs_dir.startswith('/cluster/scratch/'):
            import os
            if not os.getenv('WANDB_CACHE_DIR'):
                os.environ['WANDB_CACHE_DIR'] = '/cluster/scratch/lvignola/wandb'
                os.environ['WANDB_CONFIG_DIR'] = '/cluster/scratch/lvignola/wandb/config'
                os.environ['WANDB_DATA_DIR'] = '/cluster/scratch/lvignola/wandb/data'

        wandb_kwargs = {
            'project': project_name,
            'entity': entity_name,
            'config': configs,
            'resume': 'allow',  # Allow resuming if run exists
        }
        if wandb_notes:
            wandb_kwargs['notes'] = wandb_notes
            wandb_kwargs['tags'] = [wandb_notes]  # Also add as tags for easier filtering

        # Only set dir if not on cluster to avoid permission issues
        if not logs_dir.startswith('/cluster/scratch/'):
            wandb_kwargs['dir'] = logs_dir

        wandb.init(**wandb_kwargs)

    model_state = model.init(jr.PRNGKey(seed))
    # if num_offline_data > 0:
    #     print('collecting offline data')
    #     offline_data_gen = PendulumOfflineData(env=env, max_velocity=max_abs_velocity)
    #     offline_data_key, key = jr.split(key)
    #     offline_data = offline_data_gen.sample_transitions(key=offline_data_key,
    #                                                        num_samples=num_offline_data)
    #     offline_data = Data(inputs=jnp.concatenate([offline_data.observation, offline_data.action], axis=-1),
    #                         outputs=offline_data.next_observation - offline_data.observation,
    #                         )
    #     print('model state before update: ', model_state)
    #     updated_model_state = model.update(stats_model_state=model_state, data=offline_data)
    #     new_ms = model_state.model_state.replace(
    #         data_stats=updated_model_state.model_state.data_stats,
    #         params=updated_model_state.model_state.params,
    #     )
    #     model_state = model_state.replace(
    #         beta=updated_model_state.beta,
    #         model_state=new_ms,
    #     )
    #    print('model state after update: ', model_state)

    agent.run_episodes(num_episodes=15,
                       key=key,
                       model_state=model_state,
                       folder_name=f'{logs_dir}/{alg_name}/{exp_hash}/',
                       data=offline_data,
                       )
    wandb.finish()


def main(args):
    """"""
    from pprint import pprint
    print(args)
    """ generate experiment hash and set up redirect of output streams """
    exp_hash = hash_dict(args.__dict__)
    if args.exp_result_folder is not None:
        os.makedirs(args.exp_result_folder, exist_ok=True)
        log_file_path = os.path.join(args.exp_result_folder, '%s.log ' % exp_hash)
        logger = Logger(log_file_path)
        sys.stdout = logger
        sys.stderr = logger

    pprint(args.__dict__)
    print('\n ------------------------------------ \n')

    """ Experiment core """
    np.random.seed(args.seed)

    experiment(
        project_name=args.project_name,
        entity_name=args.entity_name,
        alg_name=args.alg_name,
        action_repeat=args.action_repeat,
        num_offline_data=args.num_offline_data,
        num_particles=args.num_particles,
        num_samples=args.num_samples,
        alpha=args.alpha,
        num_steps=args.num_steps,
        exponent=args.exponent,
        lambda_constraint=args.lambda_constraint,
        icem_horizon=args.icem_horizon,
        episode_length=args.episode_length,
        max_abs_velocity=args.max_abs_velocity,
        num_training_steps=args.num_training_steps,
        env_margin_factor=args.env_margin_factor,
        process_noise_scale=args.process_noise_scale,
        reward_source=args.reward_source,
        use_optimism=bool(args.use_optimism),
        use_pessimism=bool(args.use_pessimism),
        log_wandb=bool(args.log_wandb),
        seed=args.seed,
        logs_dir=args.logs_dir,
        num_gpus=args.num_gpus,
        exp_hash=exp_hash,
        function_norm=args.function_norm,
        num_elites=args.num_elites,
        violation_eps=args.violation_eps,
        beta=args.beta,
        lambda_sigma=args.lambda_sigma,
        action_cost=args.action_cost,
        sparse_task=args.sparse_task,
        uncertainty_eps=args.uncertainty_eps,
        uncertainty_decay_factor=args.uncertainty_decay_factor,
        uncertainty_decay_mode=args.uncertainty_decay_mode,
        uncertainty_constraint_threshold=args.uncertainty_constraint_threshold,
        default_task_index=args.default_task_index,
        actsafe_index=args.actsafe_index,
        wandb_notes=args.wandb_notes,
        use_mean_dynamics=args.use_mean_dynamics,
        aleatoric_noise_in_prediction=args.aleatoric_noise_in_prediction,
        prior_knowledge=args.prior_knowledge,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MTTest')

    # general experiment args
    parser.add_argument('--logs_dir', type=str, default='logs')
    parser.add_argument('--project_name', type=str, default='ActSafeTest')
    parser.add_argument('--alg_name', type=str, default='ActSafe')
    parser.add_argument('--entity_name', type=str, default='lvignola-eth-z-rich')
    parser.add_argument('--num_offline_data', type=int, default=100)
    parser.add_argument('--num_particles', type=int, default=10)
    parser.add_argument('--num_samples', type=int, default=500)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--num_steps', type=int, default=5)
    parser.add_argument('--exponent', type=float, default=0.2)
    parser.add_argument('--lambda_constraint', type=float, default=1e6)
    parser.add_argument('--icem_horizon', type=int, default=20)
    parser.add_argument('--episode_length', type=int, default=50)
    parser.add_argument('--action_repeat', type=int, default=2)
    parser.add_argument('--max_abs_velocity', type=float, default=6.0)
    parser.add_argument('--action_cost', type=float, default=0.0)
    parser.add_argument('--num_training_steps', type=int, default=1_000)
    parser.add_argument('--env_margin_factor', type=float, default=10.0)
    parser.add_argument('--process_noise_scale', type=float, default=1e-3)
    parser.add_argument('--reward_source', type=str, default='gym')
    parser.add_argument('--use_optimism', type=int, default=1)
    parser.add_argument('--use_pessimism', type=int, default=1)
    parser.add_argument('--log_wandb', type=int, default=1)
    parser.add_argument('--num_gpus', type=int, default=0)
    parser.add_argument('--function_norm', type=float, default=1.0)
    parser.add_argument('--num_elites', type=int, default=100)
    parser.add_argument('--violation_eps', type=float, default=0.1)
    parser.add_argument('--beta', type=float, default=3.0)
    parser.add_argument('--use_mean_dynamics', action='store_true')
    parser.add_argument('--aleatoric_noise_in_prediction', action='store_true')
    parser.add_argument('--sparse_task', action='store_true')
    parser.add_argument('--prior_knowledge', type=str, default='none', choices=['none', 'pendulum'])

    # SBSRL-specific parameters
    parser.add_argument('--lambda_sigma', type=float, default=1.0, help='Weight for exploration penalty in SBSRL')
    parser.add_argument('--uncertainty_eps', type=float, default=1.0, help='Uncertainty threshold for SBSRL')
    parser.add_argument('--uncertainty_decay_factor', type=float, default=10.0, help='Divide SBSRL uncertainty threshold by this factor each episode')
    parser.add_argument('--uncertainty_decay_mode', type=str, default='linear', choices=['linear', 'log_sigma_eps'], help='How SBSRL eps_sigma decays over episodes')
    parser.add_argument('--uncertainty_constraint_threshold', type=float, default=50.0, help='Disable SBSRL uncertainty constraint when mean relu(eps-intrinsic) exceeds this threshold')
    parser.add_argument('--default_task_index', type=int, default=0, help='Which task reward to use as extrinsic component in SBSRL')
    parser.add_argument('--actsafe_index', type=int, default=-1, help='Episode index from which ActSafe switches to task-reward exploitation (-1 disables)')
    parser.add_argument('--wandb_notes', type=str, default=None, help='Notes for wandb run grouping')

    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--exp_result_folder', type=str, default=None)

    args = parser.parse_args()
    main(args)


def run_sbsrl(lambda_sigma=1.0, uncertainty_eps=1.0, default_task_index=0, seed=0, episode_length=5, num_particles=2, num_samples=10, num_elites=5, **kwargs):
    """
    Simple function to run SBSRL experiment with default parameters
    """
    return experiment(
        alg_name='SBSRL',
        lambda_sigma=lambda_sigma,
        uncertainty_eps=uncertainty_eps,
        default_task_index=default_task_index,
        wandb_notes=args.wandb_notes,
        seed=seed,
        episode_length=episode_length,
        num_particles=num_particles,
        num_samples=num_samples,
        num_elites=num_elites,
        log_wandb=False,  # Default to false for testing
        **kwargs
    )
