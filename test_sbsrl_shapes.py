#!/usr/bin/env python3
"""
Test debug per verificare che SBSRLReward riceva i parametri corretti
"""

import sys
import os
sys.path.append('/Users/lucav/Documents/safe-model-based-exploration')

import jax.numpy as jnp
import jax.random as jr

def test_sbsrl_reward_shape():
    """Test che SBSRLReward riceva la shape corretta"""
    from smbrl.agent.sbsrl import SBSRLReward, SBSRLRewardParams
    
    # Mock extrinsic reward function with proper init_params
    class MockTaskReward:
        def __init__(self):
            pass
            
        def __call__(self, x, u, params, x_next):
            from tensorflow_probability.substrates.jax.distributions import Normal
            reward = -jnp.sum(jnp.square(x_next))
            return Normal(loc=reward, scale=jnp.zeros_like(reward)), params
        
        def init_params(self, key):
            return None  # Simple mock params
    
    mock_task_reward = MockTaskReward()
    
    # Create SBSRL reward
    sbsrl_reward = SBSRLReward(
        x_dim=3,  # Pendulum state
        u_dim=1, 
        extrinsic_reward_fn=mock_task_reward,
        lambda_sigma=1.0,
        eps_sigma=1.0
    )
    
    print(f"SBSRLReward created: x_dim={sbsrl_reward.x_dim}, u_dim={sbsrl_reward.u_dim}")
    
    # Test data
    x = jnp.array([1.0, 2.0, 3.0])  # Pendulum state (3D)
    u = jnp.array([0.5])  # Action (1D)
    x_next_with_intrinsic = jnp.array([1.1, 2.1, 3.1, 0.8])  # [next_state(3D), intrinsic_reward(1D)] = 4D
    params = SBSRLRewardParams()
    
    print(f"Input shapes: x={x.shape}, u={u.shape}, x_next={x_next_with_intrinsic.shape}")
    
    # Compute reward
    try:
        reward_dist, new_params = sbsrl_reward(x, u, params, x_next_with_intrinsic)
        reward = reward_dist.mean()
        print(f"✅ SBSRL reward computed successfully: {reward}")
        
        # Check components 
        intrinsic = x_next_with_intrinsic[-1]  # 0.8
        extrinsic = -jnp.sum(jnp.square(x_next_with_intrinsic[:-1]))  # Mock reward
        penalty = sbsrl_reward.lambda_sigma * jnp.maximum(0.0, sbsrl_reward.eps_sigma - intrinsic)
        
        print(f"  Intrinsic reward: {intrinsic}")
        print(f"  Extrinsic reward: {extrinsic}")
        print(f"  Penalty: {penalty}")
        print(f"  Total: {extrinsic} - {penalty} = {extrinsic - penalty}")
        
        return True
        
    except Exception as e:
        print(f"❌ SBSRL reward failed: {e}")
        return False

def test_sbsrl_detection():
    """Test che ExplorationSystem riconosca SBSRLReward"""
    from smbrl.model_based_rl.active_exploration_system import ExplorationSystem
    from smbrl.agent.sbsrl import SBSRLReward
    
    class MockTaskReward:
        def __call__(self, x, u, params, x_next):
            from tensorflow_probability.substrates.jax.distributions import Normal
            reward = -jnp.sum(jnp.square(x_next))
            return Normal(loc=reward, scale=jnp.zeros_like(reward)), params
        def init_params(self, key):
            return None
    
    sbsrl_reward = SBSRLReward(
        x_dim=3, u_dim=1, 
        extrinsic_reward_fn=MockTaskReward(),
        lambda_sigma=1.0, eps_sigma=1.0
    )
    
    print(f"SBSRLReward instance: {type(sbsrl_reward)}")
    print(f"Is SBSRLReward? {isinstance(sbsrl_reward, SBSRLReward)}")
    
if __name__ == "__main__":
    print("🔍 Testing SBSRL Reward Shape and Detection\\n")
    
    success1 = test_sbsrl_reward_shape()
    print()
    test_sbsrl_detection()
    
    if success1:
        print("\\n✅ SBSRL reward computation works correctly")
    else:
        print("\\n❌ SBSRL has issues")