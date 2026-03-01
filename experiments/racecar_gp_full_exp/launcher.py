import experiment
from smbrl.utils.experiment_utils import generate_run_commands, generate_base_command, dict_permutations
import argparse

PROJECT_NAME = 'Racecar26Sept15_30_GPU'
ENTITY = 'lvignola-eth-z-rich'
NUM_GPUS = 1

# Hardware configurations matching working Hydra setup
HARDWARE_CONFIGS = {
    '4090_rtx': {
        'gpu_type': 'rtx_4090',
        'cpus_per_task': 10,
        'timeout_min': 60
    },
    'rtx_a6000': {
        'gpu_type': 'rtxa6000', 
        'cpus_per_task': 8,
        'timeout_min': 120
    },
    'cpu_only': {
        'gpu_type': None,
        'cpus_per_task': 4,
        'timeout_min': 240
    }
}

_applicable_configs = {
    'project_name': [PROJECT_NAME],
    'num_training_steps': [2_000],
    'episode_length': [100],
    'action_repeat': [2],
    'seed': list(range(5)),
    'entity': [ENTITY],
    'num_gpus': [NUM_GPUS],
    'exponent': [1.0],
    'alpha': [0.2],
    'use_precomputed_kernel_params': [1],
    'icem_horizon': [20],
    'num_steps': [10],
    'violation_eps': [0.0, 0.5, 1.0],
}

_applicable_configs_actsafe = {'alg_name': ['ActSafe'], 'use_optimism': [1], 'use_pessimism': [1],
                               'beta': [3.0], 'num_particles': [10],
                               } \
                              | _applicable_configs

_applicable_configs_actsafe_mean = {'alg_name': ['ActSafe'], 'use_optimism': [1], 'use_pessimism': [0],
                                    'beta': [0.0], 'num_particles': [1],
                                    } \
                                   | _applicable_configs

_applicable_configs_sbsrl = {'alg_name': ['SBSRL'], 'use_optimism': [1], 'use_pessimism': [1],
                             'beta': [3.0], 'num_particles': [10],
                             'lambda_sigma': [0,1000],
                             'uncertainty_eps': [300],
                             'default_task_index': [1],
                             } \
                            | _applicable_configs

all_flags_combinations = dict_permutations(_applicable_configs_actsafe) \
                         + dict_permutations(_applicable_configs_actsafe_mean) \
                         + dict_permutations(_applicable_configs_sbsrl)


# _applicable_configs_actsafe_no_optimism = {'alg_name': ['ActSafe'], 'use_optimism': [0], 'use_pessimism': [1]} \
#                                           | _applicable_configs

# _applicable_configs_actsafe_no_pessimism = {'alg_name': ['ActSafe'], 'use_optimism': [0], 'use_pessimism': [0]} \
#                                            | _applicable_configs
#
# _applicable_configs_safehucrl = {'alg_name': ['SafeHUCRL'], 'use_optimism': [1], 'use_pessimism': [1]} \
#                                 | _applicable_configs

# all_flags_combinations = dict_permutations(_applicable_configs_actsafe) \
#                          + dict_permutations(_applicable_configs_actsafe_no_optimism) \
#                          + dict_permutations(_applicable_configs_actsafe_no_pessimism) \
#                          + dict_permutations(_applicable_configs_safehucrl)
#

def main(args):
    command_list = []

    logs_dir = '/cluster/scratch/'
    logs_dir += 'lvignola' + '/' + PROJECT_NAME + '/'

    for flags in all_flags_combinations:
        flags['logs_dir'] = logs_dir
        # Add wandb notes if specified
        if args.wandb_notes:
            flags['wandb_notes'] = args.wandb_notes
        cmd = generate_base_command(experiment, flags=flags)
        command_list.append(cmd)

    # submit jobs - using exact working Hydra configuration
    hw_config = HARDWARE_CONFIGS.get(args.hardware, HARDWARE_CONFIGS['4090_rtx'])
    duration_hours = hw_config['timeout_min'] // 60 if not args.long_run else 23
    duration_mins = hw_config['timeout_min'] % 60 if not args.long_run else 59
    
    generate_run_commands(command_list,
                          num_cpus=hw_config['cpus_per_task'],
                          num_gpus=NUM_GPUS if hw_config['gpu_type'] is not None else 0,
                          mode='euler',
                          duration=f'{duration_hours}:{duration_mins:02d}:00',
                          prompt=True,
                          gpu_type=hw_config['gpu_type']
                          )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hardware', type=str, default='4090_rtx', 
                       choices=['4090_rtx', 'rtx_a6000', 'cpu_only'],
                       help='hardware configuration (similar to Hydra +hardware=4090_rtx)')
    parser.add_argument('--long_run', default=False, action="store_true")
    parser.add_argument('--wandb_notes', type=str, default=None,
                       help='wandb notes for grouping runs (e.g. Mar01-racecar_sbsrl)')
    
    args = parser.parse_args()
    main(args)
