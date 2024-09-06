import os
import numpy as np
import pickle

from smbrl.utils.utils import decode_angles
from smbrl.utils.plot_episode_progression import create_plot_double
from jax import vmap

print(os.getcwd())

if __name__ == '__main__':
    experiment_name = 'NoCost28Aug2024'
    num_episodes = 15

    relative_folder_name = os.path.join('../saved_data', experiment_name)

    exploration_trajectories = []

    for i in range(num_episodes):
        file_path = os.path.join(relative_folder_name, f'episode_{i}', 'exploration_trajectory.pkl')
        with open(file_path, 'rb') as file:
            exploration_trajectory = pickle.load(file)
            exploration_trajectories.append(exploration_trajectory)


    # Now we create a numpy of all states
    unsafe_trajectories = []
    for i in range(num_episodes):
        states = exploration_trajectories[i].states.obs
        states = vmap(decode_angles)(states)
        unsafe_trajectories.append(states)

    experiment_name = 'Cost30Aug2024'
    num_episodes = 15

    relative_folder_name = os.path.join('../saved_data', experiment_name)
    exploration_trajectories = []

    for i in range(num_episodes):
        file_path = os.path.join(relative_folder_name, f'episode_{i}', 'exploration_trajectory.pkl')
        with open(file_path, 'rb') as file:
            exploration_trajectory = pickle.load(file)
            exploration_trajectories.append(exploration_trajectory)


    # Now we create a numpy of all states
    safe_trajectories = []
    for i in range(num_episodes):
        states = exploration_trajectories[i].states.obs
        states = vmap(decode_angles)(states)
        safe_trajectories.append(states)


    np.save(
        "safe_trajectories.npy",
        np.array(safe_trajectories, dtype=object),
        allow_pickle=True,
    )
    np.save(
        "unsafe_trajectories.npy",
        np.array(unsafe_trajectories, dtype=object),
        allow_pickle=True,
    )
    create_plot_double(unsafe_trajectories, safe_trajectories, [1, 3, 7, 15],
                       save_name='pendulum_exploration.pdf')
