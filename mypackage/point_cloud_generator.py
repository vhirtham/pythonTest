"""Contains methods and classes to generate 3d point clouds."""

import mypackage.geometry as geo
import numpy as np
import copy
from scipy.spatial.transform import Rotation as R


class Profile:
    """Defines a 2d profile."""

    def __init__(self, shapes):
        """
        Construct profile class.

        :param: shapes: Instance or list of geo.Shape2D class(es)
        """
        self._shapes = []

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

        self._shapes += shapes

    def num_shapes(self):
        """
        Get the number of shapes of the profile.

        :return: Number of shapes
        """
        return len(self._shapes)

    def rasterize(self, raster_width):
        """
        Rasterize the profile.

        :param: raster_width: Raster width
        :return: Raster data
        """
        raster_data = np.empty([0, 2])
        for shape in self._shapes:
            raster_data = np.vstack(
                (raster_data, shape.rasterize(raster_width)))

        return raster_data

    @property
    def shapes(self):
        return self._shapes


def check_is_point2d(point):
    if not isinstance(point, np.ndarray):
        raise TypeError("Point data must be a numpy array")
    if not len(point.shape) == 1:
        raise ValueError("Point must be a 1d array")
    if not point.size == 2:
        raise ValueError("Point must have a size of 2")
    return True


def is_point3d(point):
    if not isinstance(point, np.ndarray):
        return False
    if not point.size == 3:
        return False
    return True


def is_orthogonal(u, v, tolerance=1E-9):
    return np.abs(np.dot(u, v)) < tolerance


class CartesianCoordinateSystem3d:
    def __init__(self, basis=None, x=None, y=None, z=None,
                 origin=np.array([0, 0, 0]),
                 positive_orientation=True):
        if positive_orientation:
            sign_orientation = 1
        else:
            sign_orientation = -1
        if basis is not None:
            x = basis[0]
            y = basis[1]
            z = basis[2]

        if x is None and y is None and z is None:
            e0 = np.array([1, 0, 0])
            e1 = np.array([0, 1, 0])
            e2 = np.array([0, 0, 1]) * sign_orientation
        elif x is not None and y is not None and z is not None:
            if not (is_orthogonal(x, y) and
                    is_orthogonal(y, z) and
                    is_orthogonal(z, x)):
                raise Exception("Basis vectors must be orthogonal")
            e0 = x / np.linalg.norm(x)
            e1 = y / np.linalg.norm(y)
            e2 = z / np.linalg.norm(z)
        else:
            if x is None:
                x = self._calcualte_third_axis(y, z) * sign_orientation
            if y is None:
                y = self._calcualte_third_axis(z, x) * sign_orientation
            if z is None:
                z = self._calcualte_third_axis(x, y) * sign_orientation

            e0 = x / np.linalg.norm(x)
            e1 = y / np.linalg.norm(y)
            e2 = z / np.linalg.norm(z)

        self._basis = [e0, e1, e2]
        self._origin = np.array(origin)

    @staticmethod
    def _calcualte_third_axis(a0, a1):
        if a0 is None or a1 is None:
            raise Exception("Two axes need to be defined")
        if not is_orthogonal(a0, a1):
            raise Exception("Defined axes are not orthogonal")
        return np.cross(a0, a1)

    @property
    def basis(self):
        return self._basis

    @property
    def origin(self):
        return self._origin

    @staticmethod
    def change_of_base_rotation(css_from, css_to):
        return np.linalg.solve(css_from.basis, css_to.basis)

    @staticmethod
    def change_of_base_translation(css_from, css_to):
        return css_from.origin - css_to.origin


def vector_to_vector_transformation(u, v):
    r = np.cross(u, v)
    w = np.sqrt(np.dot(u, u) * np.dot(v, v)) + np.dot(u, v)
    quaternion = np.concatenate((r, [w]))
    unit_quaternion = quaternion / np.linalg.norm(quaternion)

    return R.from_quat(unit_quaternion).as_dcm()


class LinearHorizontalTraceSegment:

    def __init__(self, point_start, point_end):
        point_start = np.array(point_start)
        point_end = np.array(point_end)

        check_is_point2d(point_start)
        check_is_point2d(point_end)

        self._points = np.array([point_start, point_end])
        self._direction_vector = self._points[1] - self._points[0]
        self._length = np.linalg.norm(self._direction_vector)

    def _position(self, weight):
        weight = np.clip(weight, 0, 1)
        current_position = self._points[0] + weight * self._direction_vector
        return np.append(current_position, [0])

    def length(self):
        return self._length

    def local_coordinate_system(self, weight):
        weight = np.clip(weight, 0, 1)
        direction = np.append(self._direction_vector, [0])
        return CartesianCoordinateSystem3d(y=direction, z=[0, 0, 1],
                                           origin=self._position(weight))


class Trace:
    def __init__(self,
                 segments,
                 coordinate_system=CartesianCoordinateSystem3d()):
        self._segments = segments
        self._coordinate_system = coordinate_system
        length = 0
        for segment in self._segments:
            length += segment.length()
        self._length = length

    def length(self):
        return self._length

    def local_coordinate_system(self, position):
        position = np.clip(position, 0, self._length)
        cs_base = CartesianCoordinateSystem3d()
        cs_stack = copy.deepcopy(self._coordinate_system)

        for i in range(len(self._segments)):
            segment_length = self._segments[i].length()
            if position <= segment_length:
                weight = position / segment_length
            else:
                weight = 1

            cs_local = self._segments[i].local_coordinate_system(weight)
            tra = CartesianCoordinateSystem3d.change_of_base_translation(
                cs_local, cs_base)
            rot = CartesianCoordinateSystem3d.change_of_base_rotation(
                cs_base, cs_local)
            rot_tra = CartesianCoordinateSystem3d.change_of_base_rotation(
                cs_stack, cs_base)

            tra = np.matmul(rot_tra, tra)

            basis = np.matmul(rot, cs_stack.basis)
            origin = cs_stack.origin + tra
            cs_stack = CartesianCoordinateSystem3d(basis=basis,
                                                   origin=origin)

            if position <= segment_length:
                return cs_stack
            position -= segment_length


class ProfileInterpolationLSBS:
    """Linear segment by segment profile interpolation class."""

    @staticmethod
    def interpolate(a, b, weight):
        weight = np.clip(weight, 0, 1)
        if not len(a.shapes) == len(b.shapes):
            raise Exception("Number of profile shapes do not match.")

        shapes = []
        for i in range(len(a.shapes)):
            points_a = a.shapes[i].points
            points_b = b.shapes[i].points

            if points_a[:, 0].size != points_b[:, 0].size:
                raise Exception("Number of shape segments do not match.")

            points = (1 - weight) * points_a + weight * points_b

            segments_a = a.shapes[i].segments
            segments_b = b.shapes[i].segments

            for j in range(len(segments_a)):
                if not isinstance(segments_a[j], geo.Shape2D.LineSegment):
                    raise Exception(
                        "Only line segments are currently supported")

                if not isinstance(segments_a[j], type(segments_b[j])):
                    raise Exception("Shape segment types do not match.")

                if j == 0:
                    shapes += [geo.Shape2D(points[j], points[j + 1])]
                else:
                    shapes[i].add_segment(points[j + 1])
        return Profile(shapes)


def to_list(var):
    if isinstance(var, list):
        return var
    if var is None:
        return []
    return [var]


class Section:
    """Defines a section"""

    def __init__(self, profiles, trace, profile_interpolations=None,
                 profile_positions=None):

        profiles = to_list(profiles)
        profile_positions = to_list(profile_positions)
        profile_interpolations = to_list(profile_interpolations)

        if not all(isinstance(profile, Profile) for profile in profiles):
            raise TypeError(
                "Only instances of lists or Shape2d objects are accepted.")

        if len(profiles) > 1 and len(profile_interpolations) != len(
                profiles) - 1:
            raise Exception(
                "Number of interpolations must be one less than number "
                "of profiles")

        if len(profiles) > 2 and len(profile_positions) != len(profiles) - 2:
            raise Exception(
                "If more than two profiles are used, the positions of the "
                "profiles between the first and last one need to be "
                "specified in a list of size: num_profiles -2.")

        self._profiles = profiles
        self._interpolations = profile_interpolations
        self._positions = [0] + profile_positions + [trace.length()]
        self.trace = trace

    def _profile_segment_index(self, position):
        position = np.clip(position, 0, self._positions[-1])
        idx = 0
        while position > self._positions[idx + 1]:
            idx += 1
        return idx

    def _interpolated_profile(self, position):
        if len(self._profiles) == 1:
            return self._profiles[0]
        else:
            idx = self._profile_segment_index(position)
            weight = (position - self._positions[idx]) / (
                    self._positions[idx + 1] - self._positions[idx])
            profile = self._interpolations[idx].interpolate(
                self._profiles[idx], self._profiles[idx + 1], weight)

            return profile

    def rasterize(self, raster_width):

        raster_data = np.empty([0, 3])
        trace = self.trace
        global_cs = CartesianCoordinateSystem3d()
        for position in np.arange(0, trace.length() + 0.01, 0.5):
            profile = self._interpolated_profile(position)
            profile_raster_data = profile.rasterize(raster_width)
            profile_raster_data = np.insert(profile_raster_data, 1, 0, axis=1)
            trace_cs = trace.local_coordinate_system(position)

            rotation = CartesianCoordinateSystem3d.change_of_base_rotation(
                trace_cs, global_cs)
            translation = \
                CartesianCoordinateSystem3d.change_of_base_translation(
                    trace_cs, global_cs)

            profile_raster_data = np.matmul(profile_raster_data,
                                            np.transpose(rotation))
            profile_raster_data += translation

            raster_data = np.vstack((raster_data, profile_raster_data))

        return raster_data
