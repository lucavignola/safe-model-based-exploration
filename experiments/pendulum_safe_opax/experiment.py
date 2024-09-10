ENTITY = 'trevenl'


def experiment(project_name: str,
               seed: int,
               save_to_wandb: bool,
               safe_exploration: bool,
               iCem_alpha: float = 0.2):
    config = dict(seed=seed,
                  save_to_wandb=save_to_wandb,
                  safe_exploration=safe_exploration,
                  iCem_alpha=iCem_alpha)

    import os
    os.environ['JAX_PLATFORMS'] = 'cpu'

    from typing import Tuple

    import jax
    import jax.random as jr
    import wandb
    from distrax import Distribution, Normal
    from mbpo.systems.rewards.base_rewards import Reward, RewardParams
    from optax import constant_schedule

    from smbrl.optimizer.icem import iCemParams
    from smbrl.model_based_rl.main import Task, ModelBasedAgent

    jax.config.update("jax_enable_x64", True)

    import jax.numpy as jnp
    import chex
    from flax import struct
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
        logging_wandb=save_to_wandb)

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
        alpha=iCem_alpha,
        num_steps=5,
        exponent=2,
        lambda_constraint=1e6
    )

    cost_fn = None
    if safe_exploration:
        cost_fn = VelocityBound(horizon=icem_horizon,
                                max_abs_velocity=6.0 - 10 ** (-3),
                                violation_eps=1e-3, )
    agent = ModelBasedAgent(
        env=PendulumEnv(),
        model=model,
        episode_length=50,
        action_repeat=2,
        cost_fn=cost_fn,
        test_tasks=[Task(reward=PendulumSwingUp(), name='Swing up')],
        predict_difference=True,
        num_training_steps=constant_schedule(1000),
        icem_horizon=icem_horizon,
        icem_params=icem_params
    )

    wandb.init(
        project=project_name,
        dir='/cluster/scratch/' + ENTITY,
        config=config,
    )

    key = jr.PRNGKey(seed)

    key, subkey = jr.split(key)
    model_state = model.init(subkey)
    agent.run_episodes(num_episodes=16,
                       key=key,
                       model_state=model_state,
                       folder_name='EulerFolder',
                       save_to_wandb=save_to_wandb
                       )


def main(args):
    experiment(project_name=args.project_name,
               seed=args.seed,
               save_to_wandb=bool(args.save_to_wandb),
               safe_exploration=bool(args.safe_exploration),
               iCem_alpha=args.iCem_alpha
               )


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name', type=str, default='project_name')
    parser.add_argument('--seed', type=int, default=20)
    parser.add_argument('--save_to_wandb', type=int, default=1)
    parser.add_argument('--safe_exploration', type=int, default=1)
    parser.add_argument('--iCem_alpha', type=float, default=0.2)

    args = parser.parse_args()
    main(args)
