"""Contains some functions to help with visualization."""

import numpy as np


def plot_coordinate_system(coordinate_system, axes):
    """
    Plot a coordinate system in a matplotlib 3d plot.

    :param coordinate_system: Coordinate system
    :param axes: Matplotlib axes object (output from plt.gca())
    :return: ---
    """
    p0 = coordinate_system.origin
    px = p0 + coordinate_system.orientation[:, 0]
    py = p0 + coordinate_system.orientation[:, 1]
    pz = p0 + coordinate_system.orientation[:, 2]

    axes.plot([p0[0], px[0]], [p0[1], px[1]], [p0[2], px[2]], 'r')
    axes.plot([p0[0], py[0]], [p0[1], py[1]], [p0[2], py[2]], 'g')
    axes.plot([p0[0], pz[0]], [p0[1], pz[1]], [p0[2], pz[2]], 'b')


def set_axes_equal(axes):
    """
    Adjust axis in a 3d plot to be equally scaled.

    Source code taken from the stackoverflow answer of 'karlo' in the
    following question:
    https://stackoverflow.com/questions/13685386/matplotlib-equal-unit
    -length-with-equal-aspect-ratio-z-axis-is-not-equal-to

    :param axes: Matplotlib axes object (output from plt.gca())
    """
    x_limits = axes.get_xlim3d()
    y_limits = axes.get_ylim3d()
    z_limits = axes.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    axes.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    axes.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    axes.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
