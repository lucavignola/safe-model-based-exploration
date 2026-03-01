#!/usr/bin/env python3
"""
Example of parameter sweep for SBSRL experiments
"""
import sys
import os
import argparse

# Add the project root to path (works both locally and on Euler)
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, 'experiments', 'pendulum_gp_full_exp'))

# Import experiment function directly
import experiment
from smbrl.utils.experiment_utils import generate_run_commands, generate_base_command, dict_permutations

# Parameter sweep configuration
def get_sweep_config(wandb_notes):
    return {
        'project_name': ['PendulumSweep'],  
        'entity_name': ['lvignola-eth-z-rich'],  # Use entity_name not entity
        'alg_name': ['SBSRL'],
        'seed': [0, 1, 2, 3, 4],  # Multiple seeds
        'lambda_sigma': [0, 100, 1000, 100000],  # Different penalty weights
        'uncertainty_eps': [500, 1000],  # Different uncertainty thresholds
        'episode_length': [50],  # Different episode lengths
        'num_particles': [20],  # Fixed
        'num_training_steps': [1000],
        'log_wandb': [1],
        'wandb_notes': [wandb_notes] if wandb_notes else ['Mar01-sbsrl_sweep']
    }

def main(args=None):
    if args is None:
        parser = argparse.ArgumentParser(description='Parameter sweep for SBSRL experiments')
        parser.add_argument('--wandb_notes', type=str, default=None,
                           help='Notes for wandb run grouping')
        args = parser.parse_args()
    
    # Get sweep configuration with wandb_notes
    sweep_config = get_sweep_config(args.wandb_notes)
    
    command_list = []
    
    # Generate all combinations
    all_combinations = dict_permutations(sweep_config)
    print(f"Generating {len(all_combinations)} parameter combinations...")
    
    for flags in all_combinations:
        # Use experiment module directly (not the function)
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