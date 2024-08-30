import itertools
from itertools import combinations
from typing import NamedTuple, Tuple

import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Float, Array



class BoxBounds3d(NamedTuple):
    theta_bounds: Tuple[float, float] = (-np.pi, np.pi)
    omega_bounds: Tuple[float, float] = (-6.0, 6.0)
    u_bounds: Tuple[float, float] = (-1.0, 1.0)


def create_3d_trajectory_plot(spiral_data: Float[Array, 'horizon 3'],
                              box_bounds: BoxBounds3d = BoxBounds3d()
                              ):
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

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the spiral trajectory
    ax.plot(spiral_data[:, 0], spiral_data[:, 1], spiral_data[:, 2], label='3D Spiral Trajectory')

    start_point = spiral_data[0]
    ax.scatter(start_point[0], start_point[1], start_point[2], color='red', s=100, label='Start Point')
    # Define the boundaries of the box
    x_bounds = box_bounds.theta_bounds
    y_bounds = box_bounds.omega_bounds
    z_bounds = box_bounds.u_bounds

    # Plot the edges of the box
    for s, e in combinations(np.array(list(itertools.product(x_bounds, y_bounds, z_bounds))), 2):
        if np.sum(np.abs(s - e)) == x_bounds[1] - x_bounds[0] or np.sum(np.abs(s - e)) == y_bounds[1] - y_bounds[
            0] or np.sum(np.abs(s - e)) == z_bounds[1] - z_bounds[0]:
            ax.plot3D(*zip(s, e), color="gray")

    # Set labels and title
    ax.set_xlabel(r'$\theta$', fontsize=LABEL_SIZE)
    ax.set_ylabel(r'$\omega$', fontsize=LABEL_SIZE)
    ax.set_zlabel(r'u', fontsize=LABEL_SIZE)
    ax.set_title('3D Spiral Trajectory with Boundary Box', fontsize=TITLE_SIZE)

    # Show the plot
    plt.legend(fontsize=LEGEND_SIZE)
    plt.show()


if __name__ == '__main__':
    spiral_data = np.random.random((100, 3)) * [2 * np.pi, 12, 2] - [np.pi, 6, 1]
    create_3d_trajectory_plot(spiral_data, box_bounds=BoxBounds3d())
