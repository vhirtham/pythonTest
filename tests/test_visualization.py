"""Test functions of the visualization package."""

import mypackage.visualization as vs
import mypackage.transformations as tf

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt


def test_plot_coordinate_system():
    cs = tf.CoordinateSystem()
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    vs.plot_coordinate_system(cs, ax)


def test_set_axes_equal():
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    vs.set_axes_equal(ax)
