import experiment
from smbrl.utils.experiment_utils import generate_run_commands, generate_base_command, dict_permutations

PROJECT_NAME = 'CartPoleGP19Sept14_30_GPU'
ENTITY = 'trevenl'
NUM_GPUS = 1

_applicable_configs = {
    'project_name': [PROJECT_NAME],
    'num_training_steps': [2_000],
    'episode_length': [50, ],
    'action_repeat': [2, ],
    'seed': list(range(5)),
    'entity': [ENTITY],
    'num_gpus': [NUM_GPUS],

    'beta': [1, 2, ],
    'use_precomputed_kernel_params': [0, ],
    'use_function_norms': [0],

    'num_offline_data': [20, ],

    'max_position': [1.0],

    'num_samples': [1000],
    'num_particles': [20, ],
    'icem_horizon': [50, ],
    'num_elites': [100],
    'num_steps': [10],
    'violation_eps': [0.3],
}

_applicable_configs_actsafe = {'alg_name': ['ActSafe'], 'use_optimism': [1], 'use_pessimism': [1]} \
                              | _applicable_configs

all_flags_combinations = dict_permutations(_applicable_configs_actsafe)


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
                          mem=32000)


if __name__ == '__main__':
    main()
