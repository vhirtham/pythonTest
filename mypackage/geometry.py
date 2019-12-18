"""Provides classes to define lines and surfaces."""

import xarray as xr


class Point2D:
    """Defines a point in 2 dimensions."""

    def __init__(self, x, y):
        """
        Constructor.

        :param x: x-coordinate
        :param y: y-coordinate
        """
        self.coordinates = xr.DataArray([x, y])
