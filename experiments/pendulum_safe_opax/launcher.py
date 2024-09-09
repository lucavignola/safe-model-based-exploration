import experiment
from experiments.util import generate_run_commands, generate_base_command, dict_permutations

pendulum = {'project_name': ['ExplorationPendulum13hSep092024', ],
            'seed': list(range(10)),
            'safe_exploration': [0, 1],
            'iCem_alpha': [0, 0.2, ],
            }


def main():
    command_list = []
    flags_combinations = dict_permutations(pendulum)
    for flags in flags_combinations:
        cmd = generate_base_command(experiment, flags=flags)
        command_list.append(cmd)

    # submit jobs
    generate_run_commands(command_list,
                          num_cpus=1,
                          num_gpus=0,
                          mode='euler',
                          duration='23:59:00',
                          prompt=True,
                          mem=32000)



if __name__ == '__main__':
    main()