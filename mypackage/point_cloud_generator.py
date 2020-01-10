"""Contains methods and classes to generate 3d point clouds."""

import mypackage.geometry as geo
import numpy as np
from scipy.spatial.transform import Rotation as R


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


def is_point_valid(point):
    if not isinstance(point, np.ndarray):
        return False
    if not point.size == 3:
        return False
    return True


def is_orthogonal(u, v, tolerance=1E-9):
    return np.abs(np.cross(u, v) - 1) < tolerance


class CartesianCoordinateSystem3d:
    def __init__(self, x=None, y=None, z=None, positive_orientation=True):
        if z is None:
            if x is None or y is None:
                raise Exception("Two axes need to be defined")
            if not is_orthogonal(x, y):
                raise Exception("Defined axes are not orthogonal")


def vector_to_vector_transformation(u, v):
    r = np.cross(u, v)
    w = np.sqrt(np.dot(u, u) * np.dot(v, v)) + np.dot(u, v)
    quaternion = np.concatenate((r, [w]))
    unit_quaternion = quaternion / np.linalg.norm(quaternion)

    return R.from_quat(unit_quaternion).as_dcm()


class LinearTrace:

    def __init__(self, point_start, point_end):
        point_start = np.array(point_start)
        point_end = np.array(point_end)

        if not is_point_valid(point_start):
            raise TypeError("point_start is not a valid 3d point.")
        if not is_point_valid(point_end):
            raise TypeError("point_end is not a valid 3d point.")

        self._points = np.array([point_start, point_end])
        self._direction_vector = self._points[1] - self._points[0]

    def position(self, weight):
        weight = np.clip(weight, 0, 1)
        return self._points[0] + weight * self._direction_vector

    def direction(self, weight):
        return self._direction_vector


class Section:
    """Defines a section"""

    def __init__(self, profile_start, profile_end, trace):

        if not isinstance(profile_start, Profile):
            raise TypeError("profile_start must be a Profile object.")
        if not isinstance(profile_end, Profile):
            raise TypeError("profile_end must be a Profile object.")
        self._profiles = [profile_start, profile_end]
        self.trace = trace

    def rasterize(self):

        raster_data = np.empty([0, 3])
        trace = self.trace
        for weight in np.arange(0, 1.01, 0.1):
            profile = self._profiles[0]
            profile_raster_data = profile.rasterize(0.1)

            matrix = vector_to_vector_transformation([0, 0, -1],
                                                     trace.direction(weight))
            profile_raster_data = np.matmul(profile_raster_data,
                                            np.transpose(matrix))
            profile_raster_data += trace.position(weight)

            raster_data = np.vstack((raster_data, profile_raster_data))

        return raster_data
