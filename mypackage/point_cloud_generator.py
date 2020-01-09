"""Contains methods and classes to generate 3d point clouds."""

import mypackage.geometry as geo
import numpy as np


class Profile:
    """Defines a 2d profile."""

    def __init__(self, shapes):
        """
        Construct profile class.

        :param: shapes: Instance or list of geo.Shape2D class(es)
        """
        self.shapes = []

        self.add_shapes(shapes)

    def add_shapes(self, shapes):
        """
        Add shapes to the profile.

        :param shapes: Instance or list of geo.Shape2D class(es)
        :return: ---
        """
        if not isinstance(shapes, list):
            shapes = [shapes]

        if not all(isinstance(shape, geo.Shape2D) for shape in shapes):
            raise TypeError(
                "Only instances or lists of Shape2d objects are accepted.")

        self.shapes += shapes

    def num_shapes(self):
        """
        Get the number of shapes of the profile.

        :return: Number of shapes
        """
        return len(self.shapes)

    def rasterize(self, raster_width):
        """
        Rasterize the profile.

        :param: raster_width: Raster width
        :return: Raster data
        """
        raster_data = np.empty([0, 3])
        for shape in self.shapes:
            raster_data = np.vstack(
                (raster_data, shape.rasterize(raster_width)))

        return raster_data
