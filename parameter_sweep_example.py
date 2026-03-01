#!/usr/bin/env python3
"""
Example of parameter sweep for SBSRL experiments
"""
import sys
import os
sys.path.append('/Users/lucav/Documents/safe-model-based-exploration')

from experiments.pendulum_gp_full_exp import experiment
from smbrl.utils.experiment_utils import generate_run_commands, generate_base_command, dict_permutations

# Parameter sweep configuration
sweep_config = {
    'project_name': ['PendulumSweep'],  
    'entity': ['lvignola-eth-z-rich'],
    'alg_name': ['SBSRL'],
    'seed': [0, 1, 2,3,4],  # Multiple seeds
    'lambda_sigma': [0, 100, 1000, 100000],  # Different penalty weights
    'uncertainty_eps': [500,1000],  # Different uncertainty thresholds
    'episode_length': [50],  # Different episode lengths
    'num_particles': [20],  # Fixed
    'num_training_steps': [1000],
    'log_wandb': [1],
    'wandb_notes': ['Mar01-sbsrl_sweep']
}

def main():
    command_list = []
    
    # Generate all combinations
    all_combinations = dict_permutations(sweep_config)
    print(f"Generating {len(all_combinations)} parameter combinations...")
    
    for flags in all_combinations:
        cmd = generate_base_command(experiment, flags=flags)
        command_list.append(cmd)
    
    # Launch on Euler with RTX 4090
    generate_run_commands(command_list,
                          num_cpus=10,
                          num_gpus=1,  
                          mode='euler',
                          duration='2:00:00',  # 2h per job
                          gpu_type='rtx_4090',
                          prompt=True)

if __name__ == '__main__':
    main()