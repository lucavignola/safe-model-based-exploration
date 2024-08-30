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

    relative_folder_name = os.path.join('../model_based_rl', experiment_name)

    exploration_trajectories = []

    for i in range(num_episodes):
        file_path = os.path.join(relative_folder_name, f'episode_{i}', 'exploration_trajectory.pkl')
        with open(file_path, 'rb') as file:
            exploration_trajectory = pickle.load(file)
            exploration_trajectories.append(exploration_trajectory)


    # Now we create a numpy of all states
    trajectories = []
    for i in range(num_episodes):
        states = exploration_trajectories[i].states.obs
        states = vmap(decode_angles)(states)
        trajectories.append(states)

    plot_2d_trajectories(trajectories,
                         fig_title='UnSafe exploration',
                         file_name='unsafe_exploration.pdf')
