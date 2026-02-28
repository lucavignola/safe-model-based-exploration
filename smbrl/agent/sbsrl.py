# SBSRL: Safe State-Based Reinforcement Learning
# Implementation based on ActSafe with proper reward handling following the established architecture

import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float
from typing import NamedTuple, Tuple, Union, Optional
import chex
from chex import dataclass
import jax
from jax.nn import relu

from smbrl.agent.actsafe import SafeModelBasedAgent
from smbrl.model_based_rl.active_exploration_system import (
    ExplorationDynamics,
    ExplorationReward, 
    ExplorationRewardParams,
    ExplorationSystem
)
from smbrl.mbpo_stubs import Reward, RewardParams
from tensorflow_probability.substrates.jax import distributions as tfd
from tensorflow_probability.substrates.jax.distributions import Normal


@chex.dataclass
class SBSRLRewardParams:
    """Parameters for SBSRL reward computation"""
    action_cost: chex.Array | float = 0.0
    extrinsic_task_index: int = 0  # Which task to use for extrinsic penalty
    

class SBSRLReward(Reward, SBSRLRewardParams):
    """SBSRL reward: extrinsic_reward - λ_σ * relu(ε_σ - intrinsic_reward) - action_cost * ||u||^2
    """
    
    def __init__(self, x_dim: int, u_dim: int, extrinsic_reward_fn, extrinsic_task_index: int = 0, lambda_sigma: float = 1.0, eps_sigma: float = 1.0):
        super().__init__()  # Call Reward's init (no parameters)
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.extrinsic_reward_fn = extrinsic_reward_fn  # Task reward function
        self.extrinsic_task_index = extrinsic_task_index
        self.lambda_sigma = lambda_sigma  # Weight for exploration penalty
        self.eps_sigma = eps_sigma  # Uncertainty threshold
        
        # Initialize extrinsic reward parameters
        import jax.random as jr
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
        extrinsic_reward = extrinsic_reward_dist.mean()  # Use mean for base reward
        
        # SBSRL formulation: extrinsic_reward - λ_σ * relu(ε_σ - intrinsic_reward) - action_cost
        total_reward = (
            extrinsic_reward 
            - self.lambda_sigma * relu(self.eps_sigma - intrinsic_reward)  # Exploration penalty
            - reward_params.action_cost * jnp.sum(jnp.square(u), axis=0)
        )
        
        return Normal(loc=total_reward, scale=jnp.zeros_like(total_reward)), reward_params
        
    def init_params(self, key: chex.PRNGKey) -> SBSRLRewardParams:
        return SBSRLRewardParams(extrinsic_task_index=self.extrinsic_task_index)


class SBSRLAgent(SafeModelBasedAgent):
    """SBSRL Agent that inherits from SafeModelBasedAgent
    
    Key behaviors:
    - Training (train_task_index == -1): Use SBSRL reward (extrinsic - intrinsic penalty)
    - Evaluation (train_task_index >= 0): Use pure task reward (no exploration penalty)
    - Follows exact same pattern as ActSafe for proper integration
    """
    
    def __init__(self, default_task_index: int = 0, lambda_sigma: float = 1.0, uncertainty_eps: float = 1.0, *args, **kwargs):
        # Remove SBSRL-specific parameters from kwargs before passing to parent
        sbsrl_kwargs = {
            'default_task_index': default_task_index,
            'lambda_sigma': lambda_sigma,
            'uncertainty_eps': uncertainty_eps
        }
        
        # Remove any SBSRL-specific parameters from kwargs that weren't already removed
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in sbsrl_kwargs}
        
        super().__init__(*args, **filtered_kwargs)
        
        # Set train_task_index to -1 for exploration like ActSafe
        self.train_task_index = -1
        self.default_task_index = default_task_index  # Which task to use for extrinsic penalty
        self.lambda_sigma = lambda_sigma  # Weight for exploration penalty (not used in reward, kept for compatibility)
        self.uncertainty_eps = uncertainty_eps  # Uncertainty threshold (not used in reward, kept for compatibility)
        
    def get_train_rewards(self) -> Reward:
        """Return appropriate reward based on training vs evaluation mode
        
        This mirrors ActSafe's get_train_rewards exactly:
        - Training: exploration-based reward
        - Evaluation: pure task reward
        """
        if self.train_task_index == -1:
            # Training: use SBSRL reward (intrinsic - extrinsic penalty)
            extrinsic_reward_fn = self.test_tasks[self.default_task_index].reward
            return SBSRLReward(
                x_dim=self.env.observation_size,
                u_dim=self.env.action_size,
                extrinsic_reward_fn=extrinsic_reward_fn,
                extrinsic_task_index=self.default_task_index,
                lambda_sigma=self.lambda_sigma,
                eps_sigma=self.uncertainty_eps
            )
        else:
            # Evaluation: use pure extrinsic task reward (no exploration penalty)
            return self.test_tasks[self.train_task_index].reward