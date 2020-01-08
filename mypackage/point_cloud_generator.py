"""Contains methods and classes to generate 3d point clouds."""

import mypackage.geometry as geo


class Profile:
    """Defines a 2d profile."""

    def __init__(self, shapes):
        """
        Construct profile class.

        :param: shapes: Instance or list of geo.Shape2D class(es)
        """
        if not isinstance(shapes, list):
            shapes = [shapes]

        if not all(isinstance(shape, geo.Shape2D) for shape in shapes):
            raise ValueError(
                "Only instances or lists of Shape2d objects are accepted.")

        self.shapes = shapes

    def add_shape(self, shape):
        """
        Add a shape to the profile.

        :param shape: Shape that should be added.
        :return: ---
        """
        if not isinstance(shape, geo.Shape2D):
            raise ValueError("Only instances Shape2d objects are accepted.")
        self.shapes.append(shape)

    def num_shapes(self):
        """
        Get the number of shapes of the profile.

        :return: Number of shapes
        """
        return len(self.shapes)
