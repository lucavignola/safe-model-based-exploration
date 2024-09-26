import experiment
from smbrl.utils.experiment_utils import generate_run_commands, generate_base_command, dict_permutations

PROJECT_NAME = 'CartPoleGP26Sept14_30_GPU'
ENTITY = 'trevenl'
NUM_GPUS = 1

_applicable_configs = {
    'project_name': [PROJECT_NAME],
    'num_training_steps': [2_000],
    'episode_length': [50],
    'action_repeat': [2],
    'seed': list(range(5)),
    'entity': [ENTITY],
    'num_gpus': [NUM_GPUS],

    'beta': [3.0],
    'use_precomputed_kernel_params': [0, ],
    'use_function_norms': [0],

    'num_offline_data': [0, 10],

    'max_position': [1.5],

    'num_samples': [1000],
    'icem_horizon': [30, ],
    'num_elites': [100],
    'num_steps': [5],
    'violation_eps': [0.0, 0.25, 0.5],
}

_applicable_configs_actsafe = {'alg_name': ['ActSafe'],
                               'use_optimism': [1],
                               'use_pessimism': [1],
                               'num_particles': [30],
                               } | _applicable_configs

_applicable_configs_actsafe_no_pessimism = {'alg_name': ['ActSafe'],
                                            'use_optimism': [0],
                                            'use_pessimism': [0],
                                            'num_particles': [1]
                                            } | _applicable_configs

_applicable_configs_opax = {'alg_name': ['OPAX'],
                            'use_optimism': [1],
                            'use_pessimism': [1],
                            'num_particles': [30],
                            } | _applicable_configs

_applicable_configs_safehucrl = {'alg_name': ['SafeHUCRL'],
                                 'use_optimism': [1],
                                 'use_pessimism': [1],
                                 'num_particles': [30],
                                 } | _applicable_configs

all_flags_combinations = dict_permutations(_applicable_configs_actsafe) \
                         + dict_permutations(_applicable_configs_actsafe_no_pessimism) \
                         + dict_permutations(_applicable_configs_opax) \
                         + dict_permutations(_applicable_configs_safehucrl)


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
