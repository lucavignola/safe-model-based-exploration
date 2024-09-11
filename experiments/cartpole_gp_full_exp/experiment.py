import numpy as np
import os
import sys
import argparse
from smbrl.utils.experiment_utils import Logger, hash_dict


def experiment(
        project_name: str = 'ActSafeTest',
        alg_name: str = 'ActSafe',
        entity_name: str = 'sukhijab',
        exp_hash: str = '42',
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
        max_position: float = 0.5,
        num_training_steps: int = 1_000,
        use_optimism: bool = True,
        use_pessimism: bool = True,
        log_wandb: bool = True,
        logs_dir: str = 'runs',
        num_gpus: int = 0,
):
    if num_gpus == 0:
        import os
        os.environ['JAX_PLATFORMS'] = 'cpu'

    import jax.numpy as jnp
    import jax.random as jr
    import chex
    import wandb
    from smbrl.agent.actsafe import ActSafeAgent, SafeHUCRL, Task
    from flax import struct
    from distrax import Distribution, Normal
    from typing import Tuple
    from optax import constant_schedule
    from mbpo.systems.rewards.base_rewards import Reward, RewardParams
    from smbrl.optimizer.icem import iCemParams
    from smbrl.envs.cartpole_lenart import CartPoleEnv
    from smbrl.playground.cartpole_lenart_icem import PositionBound
    from bsm.bayesian_regression.gaussian_processes import GaussianProcess
    from smbrl.dynamics_models.gps import ARD
    from jaxtyping import Float, Array, Scalar

    configs = dict(
        alg_name=alg_name,
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
        max_position=max_position,
        num_training_steps=num_training_steps,
        use_optimism=use_optimism,
        use_pessimism=use_pessimism,
        num_gpus=num_gpus
    )

    key = jr.PRNGKey(seed)

    offline_data = None

    env = CartPoleEnv()

    model = GaussianProcess(
        kernel=ARD(input_dim=env.observation_size + env.action_size),
        input_dim=env.observation_size + env.action_size,
        output_dim=env.observation_size,
        output_stds=1e-3 * jnp.ones(shape=(env.observation_size,)),
        logging_wandb=log_wandb)

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
                               linear_velocity ** 2 + angular_velocity ** 2)) - reward_params.control_cost * u[
                         0] ** 2
            reward = reward.squeeze()
            return Normal(loc=reward, scale=jnp.zeros_like(reward)), reward_params

        def init_params(self, key: chex.PRNGKey) -> CartPoleRewardParams:
            default_reward_params = CartPoleRewardParams()
            return default_reward_params.replace(target_angle=self.target_angle)

    icem_params = iCemParams(
        num_particles=num_particles,
        num_samples=num_samples,
        alpha=alpha,
        num_steps=num_steps,
        exponent=exponent,
        lambda_constraint=lambda_constraint,
    )
    if alg_name == 'SafeHUCRL':
        alg = SafeHUCRL
    elif alg_name == 'ActSafe':
        alg = ActSafeAgent
    else:
        raise NotImplementedError

    agent = alg(
        env=CartPoleEnv(),
        model=model,
        episode_length=episode_length,
        action_repeat=action_repeat,
        cost_fn=PositionBound(horizon=icem_horizon,
                              max_position=max_position,
                              violation_eps=0.0, ),
        test_tasks=[Task(reward=CartPoleReward(target_angle=jnp.pi), name='Swing up', env=env),
                    Task(reward=CartPoleReward(target_angle=0.0), name='Keep down', env=env),
                    ],
        predict_difference=True,
        num_training_steps=constant_schedule(num_training_steps),
        icem_horizon=icem_horizon,
        icem_params=icem_params,
        log_to_wandb=log_wandb,
        use_pessimism=use_pessimism,
        use_optimism=use_optimism,
    )

    model_state = model.init(jr.PRNGKey(seed))
    if log_wandb:
        wandb.init(project=project_name,
                   config=configs,
                   entity=entity_name,
                   )
    agent.run_episodes(num_episodes=20,
                       key=key,
                       model_state=model_state,
                       folder_name=f'{alg_name}/{exp_hash}/{logs_dir}/',
                       data=offline_data,
                       )


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
        num_particles=args.num_particles,
        num_samples=args.num_samples,
        alpha=args.alpha,
        num_steps=args.num_steps,
        exponent=args.exponent,
        lambda_constraint=args.lambda_constraint,
        icem_horizon=args.icem_horizon,
        episode_length=args.episode_length,
        max_position=args.max_position,
        num_training_steps=args.num_training_steps,
        use_optimism=bool(args.use_optimism),
        use_pessimism=bool(args.use_pessimism),
        log_wandb=bool(args.log_wandb),
        seed=args.seed,
        logs_dir=args.logs_dir,
        num_gpus=args.num_gpus,
        exp_hash=exp_hash,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MTTest')

    # general experiment args
    parser.add_argument('--logs_dir', type=str, default='logs')
    parser.add_argument('--project_name', type=str, default='ActSafeTest')
    parser.add_argument('--alg_name', type=str, default='ActSafe')
    parser.add_argument('--entity_name', type=str, default='trevenl')
    parser.add_argument('--num_particles', type=int, default=10)
    parser.add_argument('--num_samples', type=int, default=500)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--num_steps', type=int, default=5)
    parser.add_argument('--exponent', type=float, default=0.2)
    parser.add_argument('--lambda_constraint', type=float, default=1e6)
    parser.add_argument('--icem_horizon', type=int, default=20)
    parser.add_argument('--episode_length', type=int, default=50)
    parser.add_argument('--action_repeat', type=int, default=2)
    parser.add_argument('--max_position', type=float, default=0.5)
    parser.add_argument('--num_training_steps', type=int, default=1_000)
    parser.add_argument('--use_optimism', type=int, default=1)
    parser.add_argument('--use_pessimism', type=int, default=1)
    parser.add_argument('--log_wandb', type=int, default=1)
    parser.add_argument('--num_gpus', type=int, default=0)

    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--exp_result_folder', type=str, default=None)

    args = parser.parse_args()
    main(args)
