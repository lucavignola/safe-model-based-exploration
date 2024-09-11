import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

PROJECT = 'ExplorationPendulum13hSep092024'
NUM_EPISODES = 15

data = pd.read_csv(f'{PROJECT}.csv')

# Filter data by icem alpha
iCem_alpha = 0.2
data = data.loc[data['iCem_alpha'] == iCem_alpha]

best_safe = -176.64344787597656
best_unsafe = -172.27825927734375

rewards_safe_exploration = []
rewards_unsafe_exploration = []


# We now loop over all the runs in data, for every row we compute the reward vector i.e. we go to file and go over all
# the episodes, compute the rewards and return vector of shape (NUM_EPISODES,) and add it to rewards_safe_exploration
# or rewards_unsafe_exploration depending on the parameter 'safe_exploration'

def prepare_reward_vector(run_id: str) -> np.ndarray:
    run_folder = os.path.join(PROJECT, run_id, 'saved_data')
    rewards = []
    for i in range(NUM_EPISODES):
        folder = os.path.join(run_folder, f'episode_{i}')
        file_path = os.path.join(folder, 'task_outputs.pkl')
        with open(file_path, 'rb') as file:
            tasks_datum = pickle.load(file)
        # We get the swing-up task out (at index 0) # TODO: We should name the tasks so that we can read it later
        task = tasks_datum[0]  # Task is now tuple (states: States, actions: np.ndarray)
        episode_reward = np.sum(task[0].reward)
        rewards.append(episode_reward)

    return np.array(rewards)


for index, row in data.iterrows():
    run_id, safe = row['run_id'], row['safe_exploration']
    rewards_vector = prepare_reward_vector(run_id)
    if safe:
        rewards_safe_exploration.append(rewards_vector)
    else:
        rewards_unsafe_exploration.append(rewards_vector)

rewards_safe_exploration = np.array(rewards_safe_exploration)
rewards_unsafe_exploration = np.array(rewards_unsafe_exploration)


def add_to_ax(ax, array, color, label):
    index = np.arange(len(array[0]))
    mean = np.mean(array, axis=0)
    std = np.std(array, axis=0)
    ax.plot(index, mean, color=color, label=label)
    ax.fill_between(index, mean - std, mean + std, alpha=0.2)


fig, ax = plt.subplots()

add_to_ax(ax, rewards_safe_exploration, 'blue', 'Safe')
add_to_ax(ax, rewards_unsafe_exploration, 'red', 'Unsafe')

ax.hlines(y=best_safe, xmin=0, xmax=15, linewidth=2, color='black', label='Best Safe', linestyles="--")
ax.hlines(y=best_unsafe, xmin=0, xmax=15, linewidth=2, color='black', label='Best Unsafe', linestyles="-.")

ax.set_title('Pendulum Swing-Up Task', fontsize=TITLE_SIZE)
ax.set_xlabel('Episodes', fontsize=LABEL_SIZE)
ax.set_ylabel('Rewards', fontsize=LABEL_SIZE)
ax.legend(fontsize=LEGEND_SIZE)
plt.tight_layout()
plt.savefig('rewards_swing_up_rewards.pdf')
plt.show()
