import argparse
import sys

import numpy as np

from smbrl.utils.experiment_utils import Logger, hash_dict


def experiment(
        project_name: str = 'ActSafeTest',
        alg_name: str = 'ActSafe',
        entity_name: str = 'lvignola-eth-z-rich',
        exp_hash: str = '42',
        seed: int = 0,
        num_particles: int = 10,
        num_samples: int = 500,
        alpha: float = 0.2,
        num_steps: int = 5,
        exponent: int = 2,
        lambda_constraint: float = 1e6,
        constraint_mode: str = 'penalty',
        constraint_tolerance: float = 1e-6,
        constraint_failure_mode: str = 'recovery',
        reward_dynamics_source: str = 'particles',
        icem_horizon: int = 20,
        episode_length: int = 50,
        num_episodes: int = 10,
        action_repeat: int = 2,
        max_position: float = 0.5,
        action_cost: float = 0.0,
        num_training_steps: int = 1_000,
        use_optimism: bool = True,
        use_pessimism: bool = True,
        log_wandb: bool = True,
        logs_dir: str = 'runs',
        num_gpus: int = 0,
        function_norm: float = 1.0,
        num_elites: int = 50,
        beta: float = 3.0,
        use_precomputed_kernel_params: bool = False,
        use_function_norms: bool = False,
        num_offline_data: int = 0,
        num_safe_offline_data: int = 1,
        violation_eps: float = 0.1,
        optimizer: str = 'icem',
        lambda_sigma: float = 0.0,
        uncertainty_eps: float = 100.0,
        uncertainty_decay_factor: float = 10.0,
        uncertainty_decay_mode: str = 'linear',
        uncertainty_constraint_threshold: float = 10.0,
        uncertainty_scale_with_beta: bool = False,
        default_task_index: int = 0,
        actsafe_index: int = -1,
        wandb_notes: str = None,
        num_traj: int = 0,
        gp_sampling_method: str = 'marginal',
        num_rff_features: int = 512,
        rff_path_scale: float | None = None,
        gp_sample_truncation: str = 'none',
        aleatoric_noise_in_prediction: bool = True,
        gp_prior_condition_on_initial_data: bool = False,
        gp_hyperparameter_update: str = 'model_default',
        gp_path_source: str = 'posterior',
        gp_beta_mode: str = 'fixed',
        confidence_delta: float = 0.05,
        information_gain_bound: str = 'diagonal',
        rkhs_norm_safety_factor: float = 1.0,
        num_evaluation_trajectories: int = 1,
        log_gp_diagnostics: bool = False,
):
    if rff_path_scale is None:
        rff_path_scale = 1.0
    if num_gpus == 0:
        import os
        os.environ['JAX_PLATFORMS'] = 'cpu'

    import jax.random as jr
    import jax.numpy as jnp
    import chex
    import wandb
    from smbrl.agent.actsafe import ActSafeAgent, SafeHUCRL
    from smbrl.agent.sbsrl import SBSRLAgent
    from flax import struct
    from distrax import Distribution, Normal
    from typing import Tuple
    from optax import constant_schedule
    from mbpo.systems.rewards.base_rewards import Reward, RewardParams
    from smbrl.optimizer.icem import iCemParams
    from smbrl.envs.cartpole_lenart import CartPoleEnv, CartPoleOfflineData, CartPoleTrajectoryOfflineData
    from smbrl.playground.cartpole_icem import PositionBound
    from bsm.statistical_model import GPStatisticalModel
    from smbrl.dynamics_models.gps import ARD
    from smbrl.dynamics_models.gp_confidence import TheoremGPStatisticalModel
    from jaxtyping import Float, Array, Scalar
    from bsm.utils import Data, Stats, DataStats
    from smbrl.agent.actsafe import Task

    configs = dict(
        alg_name=alg_name,
        seed=seed,
        num_particles=num_particles,
        num_samples=num_samples,
        alpha=alpha,
        num_steps=num_steps,
        exponent=exponent,
        lambda_constraint=lambda_constraint,
        constraint_mode=constraint_mode,
        constraint_tolerance=constraint_tolerance,
        constraint_failure_mode=constraint_failure_mode,
        reward_dynamics_source=reward_dynamics_source,
        icem_horizon=icem_horizon,
        episode_length=episode_length,
        num_episodes=num_episodes,
        action_repeat=action_repeat,
        max_position=max_position,
        action_cost=action_cost,
        num_training_steps=num_training_steps,
        use_optimism=use_optimism,
        use_pessimism=use_pessimism,
        num_gpus=num_gpus,
        function_norm=function_norm,
        num_elites=num_elites,
        violation_eps=violation_eps,
        beta=beta,
        use_precomputed_kernel_params=use_precomputed_kernel_params,
        use_function_norms=use_function_norms,
        num_offline_data=num_offline_data,
        num_safe_offline_data=num_safe_offline_data,
        optimizer=optimizer,
        lambda_sigma=lambda_sigma,
        uncertainty_eps=uncertainty_eps,
        uncertainty_decay_factor=uncertainty_decay_factor,
        uncertainty_decay_mode=uncertainty_decay_mode,
        uncertainty_constraint_threshold=uncertainty_constraint_threshold,
        uncertainty_scale_with_beta=uncertainty_scale_with_beta,
        default_task_index=default_task_index,
        actsafe_index=actsafe_index,
        gp_sampling_method=gp_sampling_method,
        num_rff_features=num_rff_features,
        rff_path_scale=rff_path_scale,
        gp_sample_truncation=gp_sample_truncation,
        aleatoric_noise_in_prediction=aleatoric_noise_in_prediction,
        gp_prior_condition_on_initial_data=(
            gp_prior_condition_on_initial_data
        ),
        gp_hyperparameter_update=gp_hyperparameter_update,
        gp_path_source=gp_path_source,
        gp_beta_mode=gp_beta_mode,
        confidence_delta=confidence_delta,
        information_gain_bound=information_gain_bound,
        rkhs_norm_safety_factor=rkhs_norm_safety_factor,
        num_evaluation_trajectories=num_evaluation_trajectories,
        log_gp_diagnostics=log_gp_diagnostics,
        wandb_notes=wandb_notes  # Add to config for visibility
    )
    configs['kernel_lifecycle_theory_aligned'] = (
        gp_hyperparameter_update != 'every_episode'
    )
    import jax
    jax.config.update("jax_enable_x64", True)

    precomputed_kernel_params = {
        'pseudo_length_scale': jnp.array([[7.16197382, 6.65598727, 1.27592871, 5.13755356, 4.53211409,
                                           9.09040351],
                                          [8.19072527, 2.16344693, 1.40387256, 9.7920551, 2.02477876,
                                           7.42569151],
                                          [10.34339361, 1.85744069, 1.41517779, 9.59343932, 2.2050838,
                                           7.67097848],
                                          [13.89958062, 2.77055121, 0.257797, 11.71776589, 0.99092663,
                                           10.44102166],
                                          [11.03042704, 0.98221761, 0.17153512, 10.007944, 1.01396213,
                                           10.06233002]], dtype=jnp.float64)}
    precomputed_normalization_stats = DataStats(
        inputs=Stats(mean=jnp.array([0.00055799, 0.0285231, -0.00933083, -0.09942926, 0.08258638, -0.00132751],
                                    dtype=jnp.float64),
                     std=jnp.array([0.28729099, 0.70630461, 0.70727364, 2.27893656, 4.6260925, 0.55621416],
                                   dtype=jnp.float64)),
        outputs=Stats(
            mean=jnp.array([-0.00929227, 0.01625088, -0.01068899, 0.02509439, -0.01438436],
                           dtype=jnp.float64),
            std=jnp.array([0.22896878, 0.31788133, 0.32660905, 1.29298522, 1.11480673],
                          dtype=jnp.float64)))
    precomputed_function_norms = jnp.array([14.77733678, 13.75797717, 13.80373648, 16.40662952, 17.46610356],
                                           dtype=jnp.float64)

    key = jr.PRNGKey(seed)
    key, key_offline_data = jr.split(key)

    # Choose data collection method based on num_traj parameter
    if num_traj == 0:
        if not 0 <= num_safe_offline_data <= num_offline_data:
            raise ValueError(
                "num_safe_offline_data must lie between zero and "
                f"num_offline_data={num_offline_data}, got "
                f"{num_safe_offline_data}."
            )
        # Keep the total D0 size fixed while optionally replacing uniformly
        # random points with a local design around the known safe equilibrium.
        offline_data_sampler = CartPoleOfflineData(action_repeat=action_repeat,
                                                   predict_difference=True)
        key_random_data, key_safe_data = jr.split(key_offline_data)
        num_random_offline_data = (
            num_offline_data - num_safe_offline_data
        )
        random_offline_data = offline_data_sampler.sample(
            key=key_random_data,
            num_samples=num_random_offline_data,
            max_abs_lin_position=1.0,
            max_abs_ang_velocity=5.0,
            max_abs_lin_velocity=5.0,
        )
        offline_data_parts = []
        if num_safe_offline_data > 0:
            # Keep one exact transition at the stable downward equilibrium in
            # the data passed to model.update().  Setting model_state.history
            # alone is insufficient because GP training replaces that history
            # with this offline dataset.
            equilibrium_input = jnp.array([0., 1., 0., 0., 0., 0.])
            equilibrium_output = offline_data_sampler.dynamics_fn(equilibrium_input)
            offline_data_parts.append(Data(
                inputs=equilibrium_input[None, :],
                outputs=equilibrium_output[None, :],
            ))
        if num_safe_offline_data > 1:
            offline_data_parts.append(
                offline_data_sampler.sample_near_downward_equilibrium(
                    key=key_safe_data,
                    num_samples=num_safe_offline_data - 1,
                )
            )
        offline_data_parts.append(random_offline_data)
        offline_data = Data(
            inputs=jnp.concatenate(
                [part.inputs for part in offline_data_parts], axis=0
            ),
            outputs=jnp.concatenate(
                [part.outputs for part in offline_data_parts], axis=0
            ),
        )
    else:
        # Use trajectory-based data collection
        offline_data_traj = CartPoleTrajectoryOfflineData(action_repeat=1)
        offline_data = offline_data_traj.sample(
            key,
            num_samples=500,
            num_trajectories=num_traj,
            trajectory_length=50
        )

    env = CartPoleEnv()

    if (
            use_precomputed_kernel_params
            and gp_hyperparameter_update == 'model_default'
    ):
        num_training_steps = constant_schedule(0)
    else:
        num_training_steps = constant_schedule(num_training_steps)

    if gp_beta_mode == 'theorem':
        if use_function_norms:
            # These values are finite-design simulator estimates, not
            # certified continuous-domain RKHS upper bounds.  The theorem
            # assumes one common B, so use the largest output-wise estimate
            # for every output rather than a less conservative vector bound.
            rkhs_norm_bound = (
                jnp.ones(env.observation_size)
                * jnp.max(precomputed_function_norms)
                * rkhs_norm_safety_factor
            )
            configs['rkhs_norm_bound_source'] = (
                'max_empirical_finite_design_simulator'
            )
        else:
            # An explicit prior assumption supplied by the experimenter.
            rkhs_norm_bound = (
                jnp.ones(env.observation_size)
                * function_norm
                * rkhs_norm_safety_factor
            )
            configs['rkhs_norm_bound_source'] = 'explicit_assumption'
        configs['rkhs_norm_bound'] = np.asarray(rkhs_norm_bound).tolist()
        model = TheoremGPStatisticalModel(
            kernel=ARD(input_dim=env.observation_size + env.action_size),
            input_dim=env.observation_size + env.action_size,
            output_dim=env.observation_size,
            output_stds=1e-3 * jnp.ones(shape=(env.observation_size,)),
            logging_wandb=log_wandb,
            f_norm_bound=rkhs_norm_bound,
            delta=confidence_delta,
            information_gain_bound=information_gain_bound,
            fixed_kernel_params=True,
            normalization_stats=precomputed_normalization_stats,
            num_training_steps=num_training_steps,
        )
    elif gp_beta_mode == 'bsm' or use_function_norms:
        if use_function_norms:
            rkhs_norm_bound = precomputed_function_norms * beta
        else:
            rkhs_norm_bound = jnp.ones(env.observation_size) * function_norm
        model = GPStatisticalModel(
            kernel=ARD(input_dim=env.observation_size + env.action_size),
            input_dim=env.observation_size + env.action_size,
            output_dim=env.observation_size,
            output_stds=1e-3 * jnp.ones(shape=(env.observation_size,)),
            logging_wandb=log_wandb,
            beta=None,
            f_norm_bound=rkhs_norm_bound,
            delta=confidence_delta,
            fixed_kernel_params=use_precomputed_kernel_params,
            normalization_stats=precomputed_normalization_stats,
            num_training_steps=num_training_steps,
        )
    else:
        model = GPStatisticalModel(
            kernel=ARD(input_dim=env.observation_size + env.action_size),
            input_dim=env.observation_size + env.action_size,
            output_dim=env.observation_size,
            output_stds=1e-3 * jnp.ones(shape=(env.observation_size,)),
            logging_wandb=log_wandb,
            beta=jnp.ones(shape=(env.observation_size,)) * beta,
            fixed_kernel_params=use_precomputed_kernel_params,
            normalization_stats=precomputed_normalization_stats,
            num_training_steps=num_training_steps,
        )

    @chex.dataclass
    class CartPoleRewardParams:
        control_cost: chex.Array = struct.field(default_factory=lambda: jnp.array(0.01))
        angle_cost: chex.Array = struct.field(default_factory=lambda: jnp.array(1.0))
        pos_cost: chex.Array = struct.field(default_factory=lambda: jnp.array(1.0))
        vel_cost: chex.Array = struct.field(default_factory=lambda: jnp.array(0.1))
        target_angle: chex.Array = struct.field(default_factory=lambda: jnp.array(jnp.pi))

    class CartPoleReward(Reward):
        def __init__(self, target_angle: float = jnp.pi):
            super().__init__(x_dim=5, u_dim=1)
            self.target_angle = jnp.array(target_angle)

        @staticmethod
        def cos_sin_to_angle_representation(cos_sin_angle: Float[Array, '2']) -> Scalar:
            return jnp.arctan2(cos_sin_angle[1], cos_sin_angle[0])

        def from_obs_to_state(self, state: Float[Array, '5']) -> Float[Array, '4']:
            assert state.shape == (5,)
            position, cos, sin, linear_velocity, angular_velocity = state[0], state[1], state[2], state[3], state[4]
            angle = self.cos_sin_to_angle_representation(jnp.array([cos, sin]))
            return jnp.array([position, angle, linear_velocity, angular_velocity])

        def __call__(self,
                     x: chex.Array,
                     u: chex.Array,
                     reward_params: CartPoleRewardParams,
                     x_next: chex.Array | None = None) -> Tuple[Distribution, RewardParams]:
            chex.assert_shape(x, (self.x_dim,))
            chex.assert_shape(u, (self.u_dim,))
            chex.assert_shape(x_next, (self.x_dim,))
            # get intrinsic reward out
            x_compressed = self.from_obs_to_state(x)
            position, angle = x_compressed[0], x_compressed[1]
            linear_velocity, angular_velocity = x_compressed[2], x_compressed[3]

            target_angle = reward_params.target_angle
            diff_th = angle - target_angle
            diff_th = ((diff_th + jnp.pi) % (2 * jnp.pi)) - jnp.pi
            reward = -(reward_params.angle_cost * diff_th ** 2 + reward_params.pos_cost * position ** 2 +
                       reward_params.vel_cost * (
                           linear_velocity ** 2 + angular_velocity ** 2)) - reward_params.control_cost * u[0] ** 2
            reward = reward.squeeze()
            return Normal(loc=reward, scale=jnp.zeros_like(reward)), reward_params

        def init_params(self, key: chex.PRNGKey) -> CartPoleRewardParams:
            default_reward_params = CartPoleRewardParams()
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
        constraint_mode=constraint_mode,
        constraint_tolerance=constraint_tolerance,
        reward_dynamics_source=reward_dynamics_source,
    )

    cost_fn = PositionBound(horizon=icem_horizon,
                                  max_position=max_position,
                                  violation_eps=violation_eps, )

    # Create agent with appropriate parameters
    agent_kwargs = {
        'env': CartPoleEnv(),
        'model': model,
        'episode_length': episode_length,
        'action_repeat': action_repeat,
        'cost_fn': cost_fn,
        'test_tasks': [#Task(reward=CartPoleReward(target_angle=0.0), name='Keep down', env=env),
                       Task(reward=CartPoleReward(target_angle=jnp.pi), name='Swing up', env=env),
                      ],
        'predict_difference': True,
        'num_training_steps': num_training_steps,
        'icem_horizon': icem_horizon,
        'icem_params': icem_params,
        'log_to_wandb': log_wandb,
        'use_pessimism': use_pessimism,
        'use_optimism': use_optimism,
        'optimizer': optimizer,
        'gp_sampling_method': gp_sampling_method,
        'num_rff_features': num_rff_features,
        'rff_path_scale': rff_path_scale,
        'gp_sample_truncation': gp_sample_truncation,
        'aleatoric_noise_in_prediction': aleatoric_noise_in_prediction,
        'gp_prior_condition_on_initial_data':
            gp_prior_condition_on_initial_data,
        'gp_hyperparameter_update': gp_hyperparameter_update,
        'gp_path_source': gp_path_source,
        'constraint_failure_mode': constraint_failure_mode,
        'num_evaluation_trajectories': num_evaluation_trajectories,
        'log_gp_diagnostics': log_gp_diagnostics,
    }

    # Add SBSRL-specific parameters if needed
    if alg_name == 'SBSRL':
        agent_kwargs.update({
            'action_cost': action_cost,
            'lambda_sigma': lambda_sigma,
            'uncertainty_eps': uncertainty_eps,
            'uncertainty_decay_factor': uncertainty_decay_factor,
            'uncertainty_decay_mode': uncertainty_decay_mode,
            'uncertainty_constraint_threshold': uncertainty_constraint_threshold,
            'uncertainty_scale_with_beta': uncertainty_scale_with_beta,
            'default_task_index': default_task_index,
        })
    elif alg_name == 'ActSafe':
        agent_kwargs.update({
            'actsafe_index': actsafe_index,
            'actsafe_task_index': default_task_index,
        })

    agent = alg(**agent_kwargs)

    model_state = model.init(jr.PRNGKey(seed))

    # Here we set the model_state of the GP to the right one, we must ensure that we don't do
    # any GP training afterward
    if use_precomputed_kernel_params:
        model_state.model_state.params = precomputed_kernel_params
        model_state.model_state.data_stats = precomputed_normalization_stats

    # Here we need to take care of the first datapoint!!
    if gp_beta_mode != 'theorem':
        model_state.model_state.history = Data(
            inputs=jnp.array([[0., 1.0, 0., 0., 0., 0.]]),
            outputs=jnp.array([[0., 0., 0., 0., 0.]]),
        )

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
    agent.run_episodes(num_episodes=num_episodes,
                       key=key,
                       model_state=model_state,
                       folder_name=f'{logs_dir}/{alg_name}/{exp_hash}/',
                       data=offline_data,
                       )

    wandb.finish()


def main(args):
    """"""
    from pprint import pprint
    import os
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
        num_particles=args.num_particles,
        num_samples=args.num_samples,
        alpha=args.alpha,
        num_steps=args.num_steps,
        exponent=args.exponent,
        lambda_constraint=args.lambda_constraint,
        constraint_mode=args.constraint_mode,
        constraint_tolerance=args.constraint_tolerance,
        constraint_failure_mode=args.constraint_failure_mode,
        reward_dynamics_source=args.reward_dynamics_source,
        icem_horizon=args.icem_horizon,
        episode_length=args.episode_length,
        num_episodes=args.num_episodes,
        max_position=args.max_position,
        action_cost=args.action_cost,
        num_training_steps=args.num_training_steps,
        use_optimism=bool(args.use_optimism),
        use_pessimism=bool(args.use_pessimism),
        log_wandb=bool(args.log_wandb),
        seed=args.seed,
        logs_dir=args.logs_dir,
        num_gpus=args.num_gpus,
        exp_hash=exp_hash,
        function_norm=args.function_norm,
        num_elites=args.num_elites,
        beta=args.beta,
        use_precomputed_kernel_params=bool(args.use_precomputed_kernel_params),
        use_function_norms=bool(args.use_function_norms),
        num_offline_data=args.num_offline_data,
        num_safe_offline_data=args.num_safe_offline_data,
        violation_eps=args.violation_eps,
        optimizer=args.optimizer,
        lambda_sigma=args.lambda_sigma,
        uncertainty_eps=args.uncertainty_eps,
        uncertainty_decay_factor=args.uncertainty_decay_factor,
        uncertainty_decay_mode=args.uncertainty_decay_mode,
        uncertainty_constraint_threshold=args.uncertainty_constraint_threshold,
        uncertainty_scale_with_beta=bool(
            args.uncertainty_scale_with_beta
        ),
        default_task_index=args.default_task_index,
        actsafe_index=args.actsafe_index,
        wandb_notes=args.wandb_notes,
        num_traj=args.num_traj,
        gp_sampling_method=args.gp_sampling_method,
        num_rff_features=args.num_rff_features,
        rff_path_scale=args.rff_path_scale,
        gp_sample_truncation=args.gp_sample_truncation,
        aleatoric_noise_in_prediction=bool(
            args.aleatoric_noise_in_prediction
        ),
        gp_prior_condition_on_initial_data=bool(
            args.gp_prior_condition_on_initial_data
        ),
        gp_hyperparameter_update=args.gp_hyperparameter_update,
        gp_path_source=args.gp_path_source,
        gp_beta_mode=args.gp_beta_mode,
        confidence_delta=args.confidence_delta,
        information_gain_bound=args.information_gain_bound,
        rkhs_norm_safety_factor=args.rkhs_norm_safety_factor,
        num_evaluation_trajectories=args.num_evaluation_trajectories,
        log_gp_diagnostics=bool(args.log_gp_diagnostics),
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MTTest')

    # general experiment args
    parser.add_argument('--logs_dir', type=str, default='logs')
    parser.add_argument('--project_name', type=str, default='ActSafeTest')
    parser.add_argument('--alg_name', type=str, default='ActSafe')
    parser.add_argument('--entity_name', type=str, default='lvignola-eth-z-rich')
    parser.add_argument('--num_particles', type=int, default=10)
    parser.add_argument('--num_samples', type=int, default=500)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--num_steps', type=int, default=5)
    parser.add_argument('--exponent', type=float, default=1.0)
    parser.add_argument('--lambda_constraint', type=float, default=1e8)
    parser.add_argument(
        '--constraint_mode',
        choices=['penalty', 'hard'],
        default='penalty',
        help='Penalty ranking or feasibility-first hard constraint ranking.',
    )
    parser.add_argument(
        '--constraint_tolerance',
        type=float,
        default=1e-6,
        help='Numerical feasibility tolerance used in hard mode.',
    )
    parser.add_argument(
        '--constraint_failure_mode',
        choices=['recovery', 'raise'],
        default='recovery',
        help=(
            'Raise before executing an infeasible sequence, or execute the '
            'least-violating candidate and report recovery diagnostics.'
        ),
    )
    parser.add_argument(
        '--reward_dynamics_source',
        choices=['particles', 'posterior_mean'],
        default='particles',
        help=(
            'Evaluate reward over sampled particles (legacy) or in one '
            'deterministic posterior-mean dynamics rollout.'
        ),
    )
    parser.add_argument('--icem_horizon', type=int, default=50)
    parser.add_argument('--episode_length', type=int, default=50)
    parser.add_argument('--num_episodes', type=int, default=10)
    parser.add_argument('--action_repeat', type=int, default=2)
    parser.add_argument('--max_position', type=float, default=1.5)
    parser.add_argument('--action_cost', type=float, default=0.0)
    parser.add_argument('--num_training_steps', type=int, default=1_000)
    parser.add_argument('--use_optimism', type=int, default=1)
    parser.add_argument('--use_pessimism', type=int, default=1)
    parser.add_argument('--log_wandb', type=int, default=1)
    parser.add_argument('--num_gpus', type=int, default=0)
    parser.add_argument('--function_norm', type=float, default=1.0)
    parser.add_argument('--num_elites', type=int, default=50)
    parser.add_argument('--beta', type=float, default=2.0)
    parser.add_argument('--use_precomputed_kernel_params', type=int, default=0)
    parser.add_argument('--use_function_norms', type=int, default=0)
    parser.add_argument('--num_offline_data', type=int, default=10)
    parser.add_argument(
        '--num_safe_offline_data',
        type=int,
        default=1,
        help=(
            'Number of D0 points reserved for the known safe downward region, '
            'including one exact equilibrium transition.'
        ),
    )
    parser.add_argument('--violation_eps', type=float, default=0.1)
    parser.add_argument('--optimizer', type=str, default='icem')

    # SBSRL-specific parameters
    parser.add_argument('--lambda_sigma', type=float, default=0)
    parser.add_argument('--uncertainty_eps', type=float, default=300)
    parser.add_argument('--uncertainty_decay_factor', type=float, default=10.0)
    parser.add_argument('--uncertainty_decay_mode', type=str, default='linear', choices=['linear', 'log_sigma_eps'])
    parser.add_argument('--uncertainty_constraint_threshold', type=float, default=50.0)
    parser.add_argument(
        '--uncertainty_scale_with_beta',
        type=int,
        choices=[0, 1],
        default=0,
        help='Use d_sigma^n = scheduled_d_sigma^n / max_j beta_n,j.',
    )
    parser.add_argument('--default_task_index', type=int, default=0)
    parser.add_argument('--actsafe_index', type=int, default=-1)
    parser.add_argument('--wandb_notes', type=str, default=None, help='Notes for wandb run grouping')
    parser.add_argument('--num_traj', type=int, default=0, help='Number of trajectories for trajectory-based data collection. 0=use uniform grid sampling')
    parser.add_argument('--gp_sampling_method', type=str, default='marginal',
                        choices=['marginal', 'rff'],
                        help='Epistemic dynamics sampler used inside iCEM')
    parser.add_argument('--num_rff_features', type=int, default=512,
                        help='Number of spectral frequencies per GP output in RFF mode')
    parser.add_argument('--rff_path_scale', type=float, default=None,
                        help='Scale of RFF posterior residuals (defaults to 1, an uninflated approximate posterior path)')
    parser.add_argument(
        '--gp_sample_truncation',
        type=str,
        default='none',
        choices=['none', 'posterior', 'prior', 'recursive'],
        help=(
            'Projection applied to GP samples: none, posterior '
            '(|f-mu| <= beta*sigma_n), or prior '
            '(|f-mu| <= beta*sqrt(k(z,z)))'
        ),
    )
    parser.add_argument(
        '--aleatoric_noise_in_prediction',
        type=int,
        choices=[0, 1],
        default=1,
        help=(
            'Whether planning rollouts sample the GP likelihood scale as '
            'stepwise process noise.'
        ),
    )
    parser.add_argument(
        '--gp_prior_condition_on_initial_data',
        type=int,
        choices=[0, 1],
        default=0,
        help=(
            'Sample whole-run prior paths after conditioning on D0 (1), '
            'or from the unconditioned GP prior (0).'
        ),
    )
    parser.add_argument(
        '--gp_hyperparameter_update',
        choices=['model_default', 'freeze_after_d0', 'every_episode'],
        default='model_default',
        help=(
            'model_default preserves the model construction; '
            'freeze_after_d0 fits on D0 once and then freezes; '
            'every_episode optimizes GP hyperparameters after every data '
            'update while keeping any sampled RFF paths fixed.'
        ),
    )
    parser.add_argument(
        '--gp_path_source',
        type=str,
        default='posterior',
        choices=['posterior', 'prior'],
        help='Resample posterior RFF paths each episode or retain prior paths.',
    )
    parser.add_argument(
        '--gp_beta_mode',
        type=str,
        default='fixed',
        choices=['fixed', 'bsm', 'theorem'],
        help=(
            'fixed uses --beta; bsm uses the library beta=None rule; theorem '
            'uses the SBSRL coefficient with a fixed GP.'
        ),
    )
    parser.add_argument('--confidence_delta', type=float, default=0.05)
    parser.add_argument(
        '--information_gain_bound',
        choices=['diagonal', 'observed'],
        default='diagonal',
        help=(
            'diagonal is the conservative maximum-information-gain bound; '
            'observed is a non-certified diagnostic ablation.'
        ),
    )
    parser.add_argument(
        '--rkhs_norm_safety_factor',
        type=float,
        default=1.0,
        help=(
            'Multiplier on the explicit/finite-design B value; finite-design '
            'estimates remain non-certified for every finite multiplier.'
        ),
    )
    parser.add_argument(
        '--num_evaluation_trajectories',
        type=int,
        default=1,
        help='Number of independent true-environment rollouts per task evaluation.',
    )
    parser.add_argument(
        '--log_gp_diagnostics',
        type=int,
        choices=[0, 1],
        default=0,
        help='Compute the comparatively expensive visited-path GP diagnostics.',
    )

    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--exp_result_folder', type=str, default=None)

    args = parser.parse_args()
    main(args)
