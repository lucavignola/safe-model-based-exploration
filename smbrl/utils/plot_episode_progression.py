import numpy as np
import matplotlib.pyplot as plt
from jaxtyping import Float, Array
from typing import List

LEGEND_SIZE = 20
LABEL_SIZE = 20
TICKS_SIZE = 20
TITLE_SIZE = 20
SUPTITLE_SIZE = 24
LEGEND_FONT_SIZE = 20

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


def split(trajectory, threshold=np.pi):
    parts = []
    angles, angular_velocities = trajectory[:, 0], trajectory[:, 1]
    trajectory_start_index = 0
    trajectory_prefix = np.zeros(shape=(0, 2))

    for i in range(len(angles) - 1):
        if np.abs(angles[i + 1] - angles[i]) > threshold:
            sub_trajectory = trajectory[trajectory_start_index:i + 1, :]
            if np.abs(sub_trajectory[-1][0] - np.pi) < np.abs(sub_trajectory[-1][0] + np.pi):
                # Angles is positive
                trajectory_suffix = np.array([[np.pi, (angular_velocities[i + 1] + angular_velocities[i + 1]) / 2],
                                              [np.nan, np.nan]])
                sub_trajectory = np.concatenate((trajectory_prefix, sub_trajectory, trajectory_suffix), axis=0)
                parts.append(sub_trajectory)
                trajectory_prefix = np.array([[-np.pi, (angular_velocities[i + 1] + angular_velocities[i + 1]) / 2]])
            else:
                trajectory_suffix = np.array([[-np.pi, (angular_velocities[i + 1] + angular_velocities[i + 1]) / 2],
                                              [np.nan, np.nan]])
                sub_trajectory = np.concatenate((trajectory_prefix, sub_trajectory, trajectory_suffix), axis=0)
                parts.append(sub_trajectory)
                trajectory_prefix = np.array([[np.pi, (angular_velocities[i + 1] + angular_velocities[i + 1]) / 2]])
            trajectory_start_index = i + 1
    sub_trajectory = np.concatenate((trajectory_prefix, trajectory[trajectory_start_index:, :]), axis=0)
    parts.append(sub_trajectory)
    trajectory = np.concatenate(parts, axis=0)
    return trajectory


def put_trajectories_on_ax(ax,
                           trajectories,
                           add_x_label=True,
                           add_y_label=True,
                           ):
    # Plot each trajectory with increasing darkness
    for i, trajectory in enumerate(trajectories):
        color = '#1f77b4'
        trajectory = np.array(trajectory)
        trajectory = split(trajectory, threshold=np.pi)
        ax.plot(trajectory[:, 0], trajectory[:, 1], color=color, label='Trajectories', linewidth=2)

    # Define the square boundaries and plot the square
    square_x = [-np.pi, np.pi, np.pi, -np.pi, -np.pi]
    square_y = [-6, -6, 6, 6, -6]
    ax.plot(square_x, square_y, color='r', linestyle='--', label='Safe Boundary', linewidth=3)

    ax.scatter(trajectories[0][0, 0], trajectories[0][0, 1], color='red', s=100, label='Start Point')

    # Set the viewing area
    ax.set_xlim([-4, 4])
    ax.set_ylim([-9, 9])

    # Add labels and title for clarity (optional)
    if add_x_label:
        ax.set_xlabel(r'$\theta$', fontsize=LABEL_SIZE)
    if add_y_label:
        ax.set_ylabel(r'$\omega$', fontsize=LABEL_SIZE)


def create_plot(trajectories,
                episodes: List[int]):
    num_cols = len(episodes)
    fig, axs = plt.subplots(1, ncols=num_cols, figsize=(4 * num_cols, 4), sharey=True)
    for index, episode in enumerate(episodes):
        add_y_label = True if index == 0 else False
        put_trajectories_on_ax(axs[index], trajectories[:episode], add_y_label=add_y_label)
        axs[index].set_title(f'Episode {episode}', fontsize=TITLE_SIZE)

    fig.suptitle('Pendulum Active Exploration', fontsize=SUPTITLE_SIZE, y=0.98)

    handles, labels = [], []
    for ax in axs:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            handles.append(handle)
            labels.append(label)
    by_label = dict(zip(labels, handles))

    fig.legend(by_label.values(), by_label.keys(),
               ncols=3,
               loc='upper center',
               bbox_to_anchor=(0.5, 0.94),
               fontsize=LEGEND_FONT_SIZE,
               frameon=True)

    fig.tight_layout(rect=[0.0, 0.0, 1, 0.93])
    plt.show()


def create_plot_double(unsafe_trajectories,
                       safe_trajectories,
                       episodes: List[int],
                       save_name: str | None = None,
                       suptitle: str | None = None, ):
    num_cols = len(episodes)
    fig, axs = plt.subplots(2, ncols=num_cols, figsize=(4 * num_cols, 4 * 2), sharey=True, sharex=True)
    for index, episode in enumerate(episodes):
        add_y_label = True if index == 0 else False
        add_x_label = False
        put_trajectories_on_ax(axs[0, index], unsafe_trajectories[:episode], add_y_label=add_y_label,
                               add_x_label=add_x_label)
        axs[0, index].set_title(f'Episode {episode}', fontsize=TITLE_SIZE)

    for index, episode in enumerate(episodes):
        add_y_label = True if index == 0 else False
        add_x_label = True
        put_trajectories_on_ax(axs[1, index], safe_trajectories[:episode], add_y_label=add_y_label,
                               add_x_label=add_x_label)

    if suptitle:
        fig.suptitle(suptitle, fontsize=SUPTITLE_SIZE, y=0.98)
    else:
        fig.suptitle('Pendulum Active Exploration', fontsize=SUPTITLE_SIZE, y=0.98)

    handles, labels = [], []
    for ax in axs[0]:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            handles.append(handle)
            labels.append(label)
    by_label = dict(zip(labels, handles))

    fig.legend(by_label.values(), by_label.keys(),
               ncols=3,
               loc='upper center',
               bbox_to_anchor=(0.5, 0.94),
               fontsize=LEGEND_FONT_SIZE,
               frameon=False)

    # Add title for the first row
    fig.text(0.02, 0.65, 'Unsafe', ha='center', va='center', fontsize=TITLE_SIZE, rotation='vertical')
    # Add title for the second row
    fig.text(0.02, 0.27, 'Safe', ha='center', va='center', fontsize=TITLE_SIZE, rotation='vertical')

    fig.tight_layout(rect=[0.02, 0.0, 1, 0.93])
    if save_name:
        plt.savefig(save_name)
    plt.show()


if __name__ == '__main__':
    trajectories = [np.random.uniform(-np.pi, np.pi, (50, 2)) for _ in range(10)]
    create_plot_double(trajectories, trajectories, [3, 6, 10])
