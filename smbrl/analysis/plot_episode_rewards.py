import os
import pickle

import numpy as np

from smbrl.utils.utils import decode_angles
from smbrl.utils.plot_2d_trajectories import plot_2d_trajectories
from jax import vmap
import matplotlib.pyplot as plt

LEGEND_SIZE = 20
LABEL_SIZE = 20
TICKS_SIZE = 20
TITLE_SIZE = 20

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=
r'\usepackage{amsmath}'
r'\usepackage{bm}'
r'\def\vx{{\bm{x}}}'
r'\def\vu{{\bm{u}}}'
r'\def\vf{{\bm{f}}}')

import matplotlib as mpl

mpl.rcParams['xtick.labelsize'] = TICKS_SIZE
mpl.rcParams['ytick.labelsize'] = TICKS_SIZE



if __name__ == '__main__':
    experiment_name = 'NoCost28Aug2024'
    task_titles = ['Swing Up']
    num_episodes = 16

    relative_folder_name = os.path.join('../model_based_rl', experiment_name)

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

    fig, axs = plt.subplots(1, num_tasks, sharey=True)
    axs = np.array(axs).reshape(num_tasks)

    for task_idx in range(num_tasks):
        rewards = []
        for i in range(num_episodes):
            reward = per_task_data[task_idx][i][0].reward
            rewards.append(np.sum(reward))

        axs[task_idx].plot(rewards)
        axs[task_idx].set_xlabel('Episode', fontsize=LABEL_SIZE)
        axs[task_idx].set_ylabel('Reward', fontsize=LABEL_SIZE)
        axs[task_idx].set_title(task_titles[task_idx], fontsize=TITLE_SIZE)

    fig.tight_layout()
    plt.show()

