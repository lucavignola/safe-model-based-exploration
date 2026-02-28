import experiment
from smbrl.utils.experiment_utils import generate_run_commands, generate_base_command, dict_permutations
import argparse

PROJECT_NAME = 'CartPoleGP'
ENTITY = 'lvignola-eth-z-rich'
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

_applicable_configs_sbsrl = {'alg_name': ['SBSRL'],
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
                         + dict_permutations(_applicable_configs_sbsrl) \
                         + dict_permutations(_applicable_configs_safehucrl)


def main(args):
    command_list = []

    logs_dir = '../'
    if args.mode == 'euler':
        logs_dir = '/cluster/scratch/'
        logs_dir += 'lvignola' + '/' + PROJECT_NAME + '/'

    for flags in all_flags_combinations:
        flags['logs_dir'] = logs_dir
        cmd = generate_base_command(experiment, flags=flags)
        command_list.append(cmd)

    # submit jobs
    num_hours = 23 if args.long_run else 3
    generate_run_commands(command_list,
                          num_cpus=args.num_cpus,
                          num_gpus=NUM_GPUS,
                          mode=args.mode,
                          duration=f'{num_hours}:59:00',
                          prompt=True,
                          mem=16000)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_cpus', type=int, default=1, help='number of cpus to use')
    parser.add_argument('--num_gpus', type=int, default=1, help='number of gpus to use')
    parser.add_argument('--mode', type=str, default='euler', help='how to launch the experiments')
    parser.add_argument('--long_run', default=False, action="store_true")

    args = parser.parse_args()
    main(args)
