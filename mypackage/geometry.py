"""Provides classes to define lines and surfaces."""

import numpy as np


# https://codereview.stackexchange.com/questions/193835
def is_row_in_array(row, array):
    """
    Check if a row (1d array) can be found inside of a 2d array.

    :param row: Row that should be checked
    :param array: 2d array
    :return: True or False
    """
    return (array == row).all(axis=1).any()


class Shape2D:
    """Defines a shape in 2 dimensions."""

    def __init__(self, x0, y0, x1, y1):
        """
        Constructor.

        :param x0: x-coordinate of first point
        :param y0: y-coordinate of first point
        :param x1: x-coordinate of second point
        :param y1: y-coordinate of second point
        """
        self._points = np.array([[x0, y0], [x1, y1]])

    def is_point_included(self, point):
        """
        Check if a point is already part of the shape.

        :param point: Point which should be checked
        :return: True or False
        """
        return is_row_in_array(point, self._points)

    def is_shape_closed(self):
        """
        Check if the shape is already closed.

        :return: True or False
        """
        return is_row_in_array(self._points[-1, :], self._points[:-1])

    def add_segment(self, x, y):
        """
        Add a new segment which is connected to previous one.

        :param x: x-coordinate of the segments end point
        :param y: y-coordinate of the segments end point
        :return: ---
        """
        point = np.array([x, y])
        if self.is_shape_closed():
            raise ValueError("Shape is already closed")

        self._points = np.vstack((self._points, point))
