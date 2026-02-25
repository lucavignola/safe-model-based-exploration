"""
Minimal stubs for mbpo dependencies to avoid brax.v1 import issues.
This allows running GP experiments without the full mbpo installation.
"""

from typing import TypeVar, Generic, Any
from abc import ABC, abstractmethod
import chex
from jaxtyping import Float, Array

# Type variables for generic typing
DynamicsParams_T = TypeVar('DynamicsParams_T')
RewardParams_T = TypeVar('RewardParams_T')

# Type aliases
class OptimizerState(Generic[DynamicsParams_T, RewardParams_T]):
    """Base optimizer state with generic type parameters."""
    pass

class BaseOptimizer(ABC):
    """Base optimizer interface stub."""
    
    @abstractmethod
    def init(self, key: chex.Array):
        """Initialize optimizer state."""
        pass
    
    def update(self, *args, **kwargs):
        """Update optimizer state.""" 
        pass
    
    def dummy_true_buffer_state(self, key: chex.Array):
        """Return dummy buffer state - implementation depends on specific optimizer."""
        # Return a simple empty buffer state for now
        return None

# Base parameter classes
class DynamicsParams:
    """Base dynamics parameters stub."""
    pass

class RewardParams:
    """Base reward parameters stub."""
    pass

class Reward(ABC):
    """Base reward interface stub."""
    pass

# Base system classes
class SystemParams:
    """Base system parameters stub."""
    pass

class SystemState:
    """Base system state stub."""
    pass

class System(ABC):
    """Base system interface stub."""
    pass

class Dynamics(ABC):
    """Base dynamics interface stub."""
    pass

def rollout_actions(system, system_params, init_state, horizon, actions):
    """Rollout actions through the system for the given horizon."""
    import jax.numpy as jnp
    from typing import NamedTuple, Any
    
    class RolloutResult(NamedTuple):
        reward: Any      # JAX Array
        state: Any       # JAX Array  
        observation: Any # JAX Array - states for observations
        action: Any      # JAX Array - actions taken
    
    current_state = init_state
    rewards = []
    states = [current_state]
    actions_taken = []
    observations = []
    
    for t in range(horizon):
        # Get action for this timestep
        action = actions[t] if actions.ndim > 1 else actions
        actions_taken.append(action)
        observations.append(current_state)
        
        # Step the system
        if hasattr(system, 'step'):
            sys_state = system.step(current_state, action, system_params)
            current_state = sys_state.x_next
            reward = sys_state.reward
            system_params = sys_state.system_params
        else:
            # Fallback implementation
            current_state = current_state  # No change
            reward = 0.0
        
        rewards.append(reward)
        states.append(current_state)
    
    # Return a NamedTuple which JAX can handle
    return RolloutResult(
        reward=jnp.array(rewards),
        state=jnp.stack(states),
        observation=jnp.stack(observations),
        action=jnp.stack(actions_taken) if actions_taken else jnp.array(actions_taken)
    )