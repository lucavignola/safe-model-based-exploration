import os
import pickle

from smbrl.utils.utils import decode_angles
from smbrl.utils.plot_episode_progression import create_plot_double
from jax import vmap

print(os.getcwd())

if __name__ == '__main__':
    experiment_name = 'NoCost28Aug2024'
    num_episodes = 15

    relative_folder_name = os.path.join('../saved_data', experiment_name)

    tasks_data = []

    for i in range(num_episodes):
        file_path = os.path.join(relative_folder_name, f'episode_{i}', 'task_outputs.pkl')
        with open(file_path, 'rb') as file:
            tasks_datum = pickle.load(file)
            tasks_data.append(tasks_datum)

    num_tasks = len(tasks_data[0])
    print(f'Number of tasks: {num_tasks}')

    unsafe_per_task_data = [[] for _ in range(num_tasks)]
    for episode in range(num_episodes):
        episode_tasks = tasks_data[episode]
        for task_idx in range(num_tasks):
            unsafe_per_task_data[task_idx].append(episode_tasks[task_idx])


    ##############################################################################
    ##############################################################################


    experiment_name = 'Cost30Aug2024'
    num_episodes = 15

    relative_folder_name = os.path.join('../saved_data', experiment_name)

    tasks_data = []

    for i in range(num_episodes):
        file_path = os.path.join(relative_folder_name, f'episode_{i}', 'task_outputs.pkl')
        with open(file_path, 'rb') as file:
            tasks_datum = pickle.load(file)
            tasks_data.append(tasks_datum)

    num_tasks = len(tasks_data[0])
    print(f'Number of tasks: {num_tasks}')

    safe_per_task_data = [[] for _ in range(num_tasks)]
    for episode in range(num_episodes):
        episode_tasks = tasks_data[episode]
        for task_idx in range(num_tasks):
            safe_per_task_data[task_idx].append(episode_tasks[task_idx])

    ##############################################################################
    ##############################################################################

    for task_idx in range(num_tasks):
        safe_trajectories = []
        unsafe_trajectories = []
        for i in range(num_episodes):
            states = safe_per_task_data[task_idx][i][0].obs
            states = vmap(decode_angles)(states)
            safe_trajectories.append(states)

            states = unsafe_per_task_data[task_idx][i][0].obs
            states = vmap(decode_angles)(states)
            unsafe_trajectories.append(states)

        create_plot_double(unsafe_trajectories, safe_trajectories, [1, 3, 7, 15],
                           save_name='pendulum_swing_up.pdf',
                           suptitle='Pendulum Zero Shot Swing Up')
