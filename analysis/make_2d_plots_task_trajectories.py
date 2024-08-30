import os
import pickle

from smbrl.utils.utils import decode_angles
from smbrl.utils.plot_2d_trajectories import plot_2d_trajectories
from jax import vmap

print(os.getcwd())

if __name__ == '__main__':
    # experiment_name = 'PendulumTesting28Aug2024'
    # num_episodes = 6
    experiment_name = 'NoCost28Aug2024'
    num_episodes = 15

    relative_folder_name = os.path.join('../smbrl/model_based_rl', experiment_name)

    tasks_data = []

    for i in range(num_episodes):
        file_path = os.path.join(relative_folder_name, f'episode_{i}', 'task_outputs.pkl')
        with open(file_path, 'rb') as file:
            tasks_datum = pickle.load(file)
            tasks_data.append(tasks_datum)

    num_tasks = len(tasks_data[0])
    print(f'Number of tasks: {num_tasks}')

    per_task_data = [[] for _ in range(num_tasks)]
    for episode in range(num_episodes):
        episode_tasks = tasks_data[episode]
        for task_idx in range(num_tasks):
            per_task_data[task_idx].append(episode_tasks[task_idx])

    for task_idx in range(num_tasks):
        trajectories = []
        for i in range(num_episodes):
            states = per_task_data[task_idx][i][0].obs
            states = vmap(decode_angles)(states)
            trajectories.append(states)

        plot_2d_trajectories(trajectories,
                             fig_title='UnSafe exploration [Zero shot = Swing Up]',
                             file_name='unsafe_exploration_zero_shot_swing_up.pdf')

