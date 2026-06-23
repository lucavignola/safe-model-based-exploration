# SBSRL: Safe State-Based Reinforcement Learning
# Implementation based on ActSafe with proper reward handling following the established architecture

from typing import Tuple

import chex
import jax.numpy as jnp
import jax.random as jr
from jax.nn import relu

from smbrl.agent.actsafe import SafeModelBasedAgent
from smbrl.mbpo_stubs import Reward
from tensorflow_probability.substrates.jax import distributions as tfd
from tensorflow_probability.substrates.jax.distributions import Normal


@chex.dataclass
class SBSRLRewardParams:
    """Parameters for SBSRL reward computation"""
    action_cost: chex.Array | float = 0.0
    control_cost: chex.Array | float = 0.0
    extrinsic_task_index: int = 0  # Which task to use for extrinsic penalty

def _sigmoids(x, value_at_1, sigmoid):
    """Returns 1 when `x` == 0, between 0 and 1 otherwise.

    Args:
      x: A scalar or numpy array.
      value_at_1: A float between 0 and 1 specifying the output when `x` == 1.
      sigmoid: String, choice of sigmoid type.

    Returns:
      A numpy array with values between 0.0 and 1.0.

    Raises:
      ValueError: If not 0 < `value_at_1` < 1, except for `linear`, `cosine` and
        `quadratic` sigmoids which allow `value_at_1` == 0.
      ValueError: If `sigmoid` is of an unknown type.
    """
    if sigmoid in ("cosine", "linear", "quadratic"):
        if not 0 <= value_at_1 < 1:
            raise ValueError(
                "`value_at_1` must be nonnegative and smaller than 1, "
                "got {}.".format(value_at_1)
            )
    else:
        if not 0 < value_at_1 < 1:
            raise ValueError(
                "`value_at_1` must be strictly between 0 and 1, " "got {}.".format(
                    value_at_1
                )
            )

    if sigmoid == "gaussian":
        scale = jnp.sqrt(-2 * jnp.log(value_at_1))
        return jnp.exp(-0.5 * (x * scale) ** 2)
    else:
        raise ValueError("Unknown sigmoid type {!r}.".format(sigmoid))

def tolerance(
    x,
    bounds=(0.0, 0.0),
    margin=0.0,
    sigmoid="gaussian",
    value_at_margin=0.1,
):
    """Returns 1 when `x` falls inside the bounds, between 0 and 1 otherwise.

    Args:
      x: A scalar or numpy array.
      bounds: A tuple of floats specifying inclusive `(lower, upper)` bounds for
        the target interval. These can be infinite if the interval is unbounded
        at one or both ends, or they can be equal to one another if the target
        value is exact.
      margin: Float. Parameter that controls how steeply the output decreases as
        `x` moves out-of-bounds.
        * If `margin == 0` then the output will be 0 for all values of `x`
          outside of `bounds`.
        * If `margin > 0` then the output will decrease sigmoidally with
          increasing distance from the nearest bound.
      sigmoid: String, choice of sigmoid type. Valid values are: 'gaussian',
         'linear', 'hyperbolic', 'long_tail', 'cosine', 'tanh_squared'.
      value_at_margin: A float between 0 and 1 specifying the output value when
        the distance from `x` to the nearest bound is equal to `margin`. Ignored
        if `margin == 0`.

    Returns:
      A float or numpy array with values between 0.0 and 1.0.

    Raises:
      ValueError: If `bounds[0] > bounds[1]`.
      ValueError: If `margin` is negative.
    """
    lower, upper = bounds
    if lower > upper:
        raise ValueError("Lower bound must be <= upper bound.")
    if margin < 0:
        raise ValueError("`margin` must be non-negative.")

    in_bounds = jnp.logical_and(lower <= x, x <= upper)
    if margin == 0:
        value = jnp.where(in_bounds, 1.0, 0.0)
    else:
        d = jnp.where(x < lower, lower - x, x - upper) / margin
        value = jnp.where(in_bounds, 1.0, _sigmoids(d, value_at_margin, sigmoid))

    return float(value) if jnp.isscalar(x) else value


class SBSRLReward(Reward, SBSRLRewardParams):
    """SBSRL reward: extrinsic_reward - λ_σ * relu(ε_σ - intrinsic_reward) - action_cost - control_cost * ||u||^2
    """

    def __init__(self, x_dim: int, u_dim: int, extrinsic_reward_fn, extrinsic_task_index: int = 0, lambda_sigma: float = 1.0, eps_sigma: float = 1.0, action_cost: float = 0.0):
        super().__init__()  # Call Reward's init (no parameters)
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.extrinsic_reward_fn = extrinsic_reward_fn  # Task reward function
        self.extrinsic_task_index = extrinsic_task_index
        self.lambda_sigma = lambda_sigma  # Weight for exploration penalty
        self.eps_sigma = eps_sigma  # Uncertainty threshold
        self.action_cost = action_cost

        # Initialize extrinsic reward parameters
        self.extrinsic_reward_params = self.extrinsic_reward_fn.init_params(jr.PRNGKey(0))

    def __call__(self,
                 x: chex.Array,
                 u: chex.Array,
                 reward_params: SBSRLRewardParams,
                 x_next: chex.Array | None = None) -> Tuple[tfd.Distribution, SBSRLRewardParams]:
        chex.assert_shape(x, (self.x_dim,))
        chex.assert_shape(u, (self.u_dim,))
        chex.assert_shape(x_next, (self.x_dim + 1,))  # State + intrinsic reward from ExplorationDynamics

        # Extract intrinsic reward from augmented state (added by ExplorationDynamics)
        intrinsic_reward = x_next[-1]
        pure_next_state = x_next[:-1]

        # Compute extrinsic reward using task reward with proper parameters
        extrinsic_reward_dist, _ = self.extrinsic_reward_fn(x, u, self.extrinsic_reward_params, pure_next_state)
        extrinsic_reward = extrinsic_reward_dist.mean()

        # SBSRL formulation: extrinsic_reward - λ_σ * relu(ε_σ - intrinsic_reward) - action_cost
        # Note: control_cost is already applied by the task reward (PendulumReward/CartPoleReward),
        # so we don't apply it again here to avoid double-counting
        total_reward = (
            extrinsic_reward
            - self.lambda_sigma * relu(self.eps_sigma - intrinsic_reward)  # Exploration penalty
            - reward_params.action_cost * (1 - tolerance(u, (-0.1, 0.1), 0.1))[0]
            # - self.control_cost * jnp.sum(jnp.square(u), axis=0)  # Commented: task reward already applies this
        )

        return Normal(loc=total_reward, scale=jnp.zeros_like(total_reward)), reward_params

    def init_params(self, key: chex.PRNGKey) -> SBSRLRewardParams:
        return SBSRLRewardParams(
            action_cost=self.action_cost,
            control_cost=0.0,
            extrinsic_task_index=self.extrinsic_task_index,
        )


class SBSRLAgent(SafeModelBasedAgent):
    """SBSRL Agent that inherits from SafeModelBasedAgent

    Key behaviors:
    - Training (train_task_index == -1): Use SBSRL reward (extrinsic - intrinsic penalty)
    - Evaluation (train_task_index >= 0): Use pure task reward (no exploration penalty)
    - Follows exact same pattern as ActSafe for proper integration
    """

    def __init__(self, default_task_index: int = 0, lambda_sigma: float = 1.0, uncertainty_eps: float = 1.0,
                 uncertainty_decay_factor: float = 10.0,
                 uncertainty_decay_mode: str = 'linear',
                 uncertainty_constraint_threshold: float = 50.0,
                 action_cost: float = 0.0,
                 *args, **kwargs):
        # Remove SBSRL-specific parameters from kwargs before passing to parent
        sbsrl_kwargs = {
            'default_task_index': default_task_index,
            'lambda_sigma': lambda_sigma,
            'uncertainty_eps': uncertainty_eps,
            'uncertainty_decay_factor': uncertainty_decay_factor,
            'uncertainty_decay_mode': uncertainty_decay_mode,
            'uncertainty_constraint_threshold': uncertainty_constraint_threshold,
            'action_cost': action_cost,
        }

        # Remove any SBSRL-specific parameters from kwargs that weren't already removed
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in sbsrl_kwargs}

        super().__init__(*args, **filtered_kwargs)

        # Set train_task_index to -1 for exploration like ActSafe
        self.train_task_index = -1
        self.default_task_index = default_task_index
        self.lambda_sigma = lambda_sigma
        self.initial_uncertainty_eps = uncertainty_eps
        self.uncertainty_eps = uncertainty_eps
        self.uncertainty_decay_factor = uncertainty_decay_factor
        self.uncertainty_decay_mode = uncertainty_decay_mode
        self.uncertainty_constraint_threshold = uncertainty_constraint_threshold
        self.action_cost = action_cost
        self.uncertainty_constraint_enabled = True
        self.latest_uncertainty_penalty_mean = 0.0
        self._sbsrl_reward: SBSRLReward | None = None
        self.enable_additional_exploration_optimizer = True

    def get_episode_wandb_metrics(self, episode_idx: int) -> dict:
        return {
            'sbsrl/eps_sigma': float(self.uncertainty_eps),
            'sbsrl/uncertainty_penalty_mean': float(self.latest_uncertainty_penalty_mean),
            'sbsrl/uncertainty_constraint_enabled': float(self.uncertainty_constraint_enabled),
        }

    def on_exploration_rollout_end(self,
                                   episode_idx: int,
                                   intrinsic_rewards: chex.Array,
                                   extrinsic_rewards: chex.Array) -> None:
        if self.train_task_index != -1:
            return
        penalty = relu(self.uncertainty_eps - intrinsic_rewards)
        self.latest_uncertainty_penalty_mean = float(jnp.mean(penalty))
        if self.uncertainty_constraint_enabled and (
                self.latest_uncertainty_penalty_mean > self.uncertainty_constraint_threshold):
            self.uncertainty_eps = 0.0
            self.uncertainty_constraint_enabled = False
            if self._sbsrl_reward is not None:
                self._sbsrl_reward.eps_sigma = self.uncertainty_eps

    def get_train_rewards(self) -> Reward:
        """Return appropriate reward based on training vs evaluation mode

        This mirrors ActSafe's get_train_rewards exactly:
        - Training: exploration-based reward
        - Evaluation: pure task reward
        """
        if self.train_task_index == -1:
            # Training: use SBSRL reward
            if self._sbsrl_reward is None:
                extrinsic_reward_fn = self.test_tasks[self.default_task_index].reward
                self._sbsrl_reward = SBSRLReward(
                    x_dim=self.env.observation_size,
                    u_dim=self.env.action_size,
                    extrinsic_reward_fn=extrinsic_reward_fn,
                    extrinsic_task_index=self.default_task_index,
                    lambda_sigma=self.lambda_sigma,
                    eps_sigma=self.uncertainty_eps,
                    action_cost=self.action_cost,
                )
            self._sbsrl_reward.eps_sigma = self.uncertainty_eps
            return self._sbsrl_reward
        else:
            # Evaluation: use pure extrinsic task reward (no exploration penalty)
            return self.test_tasks[self.train_task_index].reward

    def on_episode_end(self, episode_idx: int) -> None:
        if self.train_task_index == -1:
            if self.uncertainty_constraint_enabled:
                if self.uncertainty_decay_mode == 'linear':
                    self.uncertainty_eps = self.uncertainty_eps / self.uncertainty_decay_factor
                elif self.uncertainty_decay_mode == 'log_sigma_eps':
                    self.uncertainty_eps = self.initial_uncertainty_eps / (1.0 + jnp.log(episode_idx + 2.0))
                else:
                    raise ValueError(f'Unknown uncertainty_decay_mode {self.uncertainty_decay_mode}')
            else:
                self.uncertainty_eps = 0.0
            if self._sbsrl_reward is not None:
                self._sbsrl_reward.eps_sigma = self.uncertainty_eps
            if self.log_to_wandb:
                import wandb
                wandb.log({
                    'episode_idx': episode_idx,
                    'sbsrl/eps_sigma_after_decay': float(self.uncertainty_eps),
                })
