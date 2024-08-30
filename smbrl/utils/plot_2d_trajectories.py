import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

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


def plot_2d_trajectories(trajectories,
                         file_name: str | None = None,
                         fig_title: str | None = None, ):
    # Set up the plot
    fig, ax = plt.subplots()

    # Create a colormap
    cmap = cm.viridis  # You can also use other colormaps like cm.plasma, cm.inferno, etc.
    normalize = mcolors.Normalize(vmin=0, vmax=len(trajectories))
    sm = cm.ScalarMappable(cmap=cmap, norm=normalize)
    sm.set_array([])

    # Plot each trajectory with increasing darkness
    for i, trajectory in enumerate(trajectories):
        color = cmap(normalize(i))  # Get color from colormap
        trajectory = np.array(trajectory)
        trajectory = split(trajectory, threshold=np.pi)

        ax.plot(trajectory[:, 0], trajectory[:, 1], color=color)

    # Define the square boundaries and plot the square
    square_x = [-np.pi, np.pi, np.pi, -np.pi, -np.pi]
    square_y = [-6, -6, 6, 6, -6]
    ax.plot(square_x, square_y, color='r', linestyle='--', label='Safe Boundary')

    ax.scatter(trajectories[0][0, 0], trajectories[0][0, 1], color='red', s=100, label='Start Point')

    # Set the viewing area
    ax.set_xlim([-4, 4])
    ax.set_ylim([-9, 9])

    # Add labels and title for clarity (optional)
    ax.set_xlabel(r'$\theta$', fontsize=LABEL_SIZE)
    ax.set_ylabel(r'$\omega$', fontsize=LABEL_SIZE)

    if fig_title:
        ax.set_title(fig_title, fontsize=TITLE_SIZE, pad=40)
    else:
        ax.set_title('Pendulum safe exploration', fontsize=TITLE_SIZE, pad=40)

    # Add a colorbar
    cbar = plt.colorbar(sm, ax=ax, ticks=range(len(trajectories)))
    cbar.set_label('Episode', fontsize=TICKS_SIZE)

    ticks = np.arange(0, len(trajectories), 3)
    tick_labels = [f'{i}' for i in ticks]

    cbar.set_ticks(ticks)
    cbar.set_ticklabels(tick_labels)

    fig.legend(ncols=2,
               loc='upper center',
               bbox_to_anchor=(0.5, 0.95),
               fontsize=LEGEND_SIZE,
               frameon=False)
    fig.tight_layout(rect=[0.0, 0.0, 1, 1])

    # Display the plot
    plt.tight_layout()

    if file_name:
        plt.savefig(file_name)

    plt.show()


if __name__ == '__main__':
    trajectories = [np.random.uniform(-np.pi, np.pi, (50, 2)) for _ in range(3)]
    plot_2d_trajectories(trajectories)
