import experiment
from smbrl.utils.experiment_utils import generate_run_commands, generate_base_command, dict_permutations

PROJECT_NAME = 'PendulumGP11Sept17_00_GPU'
ENTITY = 'trevenl'
NUM_GPUS = 1

_applicable_configs = {
    'env_margin_factor': [10, ],
    'reward_source': ['gym'],
    'project_name': [PROJECT_NAME],
    'num_training_steps': [1_000],
    'seed': list(range(5)),
    'entity': [ENTITY],
    'num_gpus': [NUM_GPUS],
    'function_norm': [1.0, 2.0, 5.0, ],
    'num_particles': [20, ],
    'num_samples': [1000, ],
    'lambda_constraint': [1e7],
    'icem_horizon': [30],
    'num_elites': [100, ]
}

_applicable_configs_actsafe = {'alg_name': ['ActSafe'], 'use_optimism': [1], 'use_pessimism': [1]} \
                              | _applicable_configs

# _applicable_configs_actsafe_no_optimism = {'alg_name': ['ActSafe'], 'use_optimism': [0], 'use_pessimism': [1]} \
#                                           | _applicable_configs
#
# _applicable_configs_actsafe_no_pessimism = {'alg_name': ['ActSafe'], 'use_optimism': [0], 'use_pessimism': [0]} \
#                                            | _applicable_configs
#
# _applicable_configs_safehucrl = {'alg_name': ['SafeHUCRL'], 'use_optimism': [1], 'use_pessimism': [1]} \
#                                 | _applicable_configs

all_flags_combinations = dict_permutations(_applicable_configs_actsafe)


# all_flags_combinations = dict_permutations(_applicable_configs_actsafe) \
#                          + dict_permutations(_applicable_configs_actsafe_no_optimism) \
#                          + dict_permutations(_applicable_configs_actsafe_no_pessimism) \
#                          + dict_permutations(_applicable_configs_safehucrl)


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
