import experiment
from smbrl.utils.experiment_utils import generate_run_commands, generate_base_command, dict_permutations

PROJECT_NAME = 'Racecar26Sept15_30_GPU'
ENTITY = 'sukhijab'
NUM_GPUS = 1

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

def main():
    command_list = []

    logs_dir = '/cluster/scratch/'
    logs_dir += ENTITY + '/' + PROJECT_NAME + '/'

    for flags in all_flags_combinations:
        flags['logs_dir'] = logs_dir
        cmd = generate_base_command(experiment, flags=flags)
        command_list.append(cmd)

    # submit jobs
    generate_run_commands(command_list,
                          num_cpus=1,
                          num_gpus=NUM_GPUS,
                          mode='euler',
                          duration='23:59:00',
                          prompt=True,
                          mem=16000)


if __name__ == '__main__':
    main()
