import experiment
from smbrl.utils.experiment_utils import generate_run_commands, generate_base_command, dict_permutations
import argparse

PROJECT_NAME = 'PendulumGP'
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
    'env_margin_factor': [10],
    'reward_source': ['gym'],
    'project_name': [PROJECT_NAME],
    'num_training_steps': [1_000],
    'seed': list(range(5)),
    'entity': [ENTITY],
    'num_gpus': [NUM_GPUS],
    'num_samples': [1_000],
    'lambda_constraint': [1e7],
    'icem_horizon': [20],
    'num_elites': [100],
    'num_offline_data': [0]
}

_applicable_configs_actsafe = {'alg_name': ['ActSafe'], 'use_optimism': [1], 'use_pessimism': [1],
                               'num_particles': [30],
                               'beta': [3.0],
                               } \
                              | _applicable_configs

_applicable_configs_opax = {'alg_name': ['OPAX'], 'use_optimism': [1], 'use_pessimism': [1],
                            'num_particles': [30],
                            'beta': [3.0],
                            } \
                           | _applicable_configs

_applicable_configs_sbsrl = {'alg_name': ['SBSRL'], 'use_optimism': [1], 'use_pessimism': [1],
                             'num_particles': [30],
                             'beta': [3.0],
                             'lambda_sigma': [0,1000],
                             'uncertainty_eps': [300],
                             'default_task_index': [1],
                             } \
                            | _applicable_configs

# _applicable_configs_actsafe_no_pessimism = {'alg_name': ['ActSafe'], 'use_optimism': [0], 'use_pessimism': [0],
#                                             'num_particles': [1],
#                                             'beta': [3.0],
#                                             } \
#                                            | _applicable_configs

# _applicable_configs_actsafe_no_optimism = {'alg_name': ['ActSafe'], 'use_optimism': [0], 'use_pessimism': [1]} \
#                                           | _applicable_configs
#
# _applicable_configs_actsafe_no_pessimism = {'alg_name': ['ActSafe'], 'use_optimism': [0], 'use_pessimism': [0]} \
#                                            | _applicable_configs
#
_applicable_configs_safehucrl = {'alg_name': ['SafeHUCRL'], 'use_optimism': [1], 'use_pessimism': [1],
                                 'num_particles': [30],
                                 'beta': [3.0],
                                 } \
                                | _applicable_configs

_applicable_configs_hucrl = {'alg_name': ['HUCRL'], 'use_optimism': [1], 'use_pessimism': [1],
                             'num_particles': [30],
                             'beta': [3.0],
                             } \
                            | _applicable_configs

_applicable_configs_actsafe_mean = {'alg_name': ['ActSafe'], 'use_optimism': [0], 'use_pessimism': [0],
                                    'num_particles': [1],
                                    'beta': [0.0],
                                    } \
                                   | _applicable_configs

all_flags_combinations = dict_permutations(_applicable_configs_actsafe) \
                         + dict_permutations(_applicable_configs_actsafe_mean) \
                         + dict_permutations(_applicable_configs_safehucrl) \
                         + dict_permutations(_applicable_configs_opax) \
                         + dict_permutations(_applicable_configs_sbsrl) \
                         + dict_permutations(_applicable_configs_hucrl)



# all_flags_combinations = dict_permutations(_applicable_configs_actsafe) \
#                          + dict_permutations(_applicable_configs_actsafe_no_optimism) \
#                          + dict_permutations(_applicable_configs_actsafe_no_pessimism) \
#                          + dict_permutations(_applicable_configs_safehucrl)


def main(args):
    command_list = []
    logs_dir = '../'
    if args.mode == 'euler':
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
                          mode=args.mode,
                          duration=f'{duration_hours}:{duration_mins:02d}:00',
                          prompt=True,
                          gpu_type=hw_config['gpu_type']
                          )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='euler', help='how to launch the experiments')
    parser.add_argument('--long_run', default=False, action="store_true")
    parser.add_argument('--hardware', type=str, default='4090_rtx', 
                       choices=['4090_rtx', 'rtx_a6000', 'cpu_only'],
                       help='hardware configuration (similar to Hydra +hardware=4090_rtx)')
    parser.add_argument('--wandb_notes', type=str, default=None,
                       help='wandb notes for grouping runs (e.g. Mar01-pendulum_sbsrl)')

    args = parser.parse_args()
    main(args)

    args = parser.parse_args()
    main(args)
