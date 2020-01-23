"""Provides classes to define lines and surfaces."""

import numpy as np
import math
import mypackage.transformations as tf
from scipy.spatial.transform import Rotation as R


def check_point_data_valid(point):
    """
    Check if the data of a point is valid.

    :param point: Point that should be checked
    :return: ---
    """
    if not (np.ndim(point) == 1 and point.size == 2):
        raise Exception(
            "Point data is invalid. Must be an array with 2 values.")


# https://codereview.stackexchange.com/questions/193835
def is_row_in_array(row, array):
    """
    Check if a row (1d array) can be found inside of a 2d array.

    :param row: Row that should be checked
    :param array: 2d array
    :return: True or False
    """
    return (array == row).all(axis=1).any()


def vector_points_to_left_of_vector(vector, vector_reference):
    """
    Determine if a vector points to the left of another vector.

    Returns 1 if the vector points to the left of the reference vector and
    -1 if it points to the right. In case both vectors point into the same
    or the opposite directions, this function returns 0.

    :param vector: Vector
    :param vector_reference: Reference vector
    :return: 1,-1 or 0 (see description)
    """
    return int(np.sign(np.linalg.det([vector_reference, vector])))


def point_left_of_line(point, line_start, line_end):
    """
    Determine if a point lies left of a line.

    Returns 1 if the point is left of the line and -1 if it is to the right.
    If the point is located on the line, this function returns 0.

    :param point: Point
    :param line_start: Starting point of the line
    :param line_end: End point of the line
    :return: 1,-1 or 0 (see description)
    """
    vec_line_start_end = line_end - line_start
    vec_line_start_point = point - line_start
    return vector_points_to_left_of_vector(vec_line_start_point,
                                           vec_line_start_end)


def reflection_multiplier(transformation_matrix):
    """
    Get a multiplier indicating if the transformation is a reflection.

    Returns -1 if the transformation contains a reflection and 1 if not.

    :param transformation_matrix: Transformation matrix
    :return: 1 or -1 (see description)
    """
    points = np.identity(2)
    transformed_points = np.matmul(points,
                                   np.transpose(transformation_matrix))
    origin_left_of_line = point_left_of_line(np.array([0, 0]),
                                             transformed_points[0],
                                             transformed_points[1])

    if origin_left_of_line == 0:
        raise Exception("Invalid transformation")

    return np.sign(origin_left_of_line)


def reflection_sign(matrix):
    """
    Get a sign indicating if the transformation is a reflection.

    Returns -1 if the transformation contains a reflection and 1 if not.

    :param matrix: Transformation matrix
    :return: 1 or -1 (see description)
    """
    sign = int(np.sign(np.linalg.det(matrix)))

    if sign == 0:
        raise Exception("Invalid transformation")

    return sign


# LineSegment -----------------------------------------------------------------

class LineSegment:
    """Line segment."""

    def __init__(self, point_start, point_end):
        """
        Constructor.

        :param point_start: Starting point of the segment
        :param point_end: End point of the segment
        """
        self._points = np.transpose(
            np.array([point_start, point_end], dtype=float))
        self._calculate_length()

    def _calculate_length(self):
        """
        Calculate the segment length from its points.

        :return: ---
        """
        self._length = np.linalg.norm(self._points[:, 1] - self._points[:, 0])
        if math.isclose(self._length, 0):
            raise ValueError("Segment length is 0.")

    @property
    def length(self):
        """
        Get the segment length.

        :return: Segment length
        """
        return self._length

    @property
    def point_end(self):
        """
        Get the end point of the segment.

        :return: End point
        """
        return self._points[:, 1]

    @property
    def point_start(self):
        """
        Get the starting point of the segment.

        :return: Starting point
        """
        return self._points[:, 0]

    def apply_transformation(self, matrix):
        """
        Apply a transformation matrix to the segment.

        :param matrix: Transformation matrix
        :return: ---
        """
        self._points = np.matmul(matrix, self._points)
        self._calculate_length()

    def rasterize(self, raster_width, num_points_excluded_end=0):
        """
        Create an array of points that describe the segments contour.

        The effective raster width may vary from the specified one,
        since the algorithm enforces constant distances between two
        raster points.

        :param raster_width: The desired distance between two raster points
        :param num_points_excluded_end: Specifies how many points from the
        end should be excluded from the rasterization. The main purpose of
        this parameter is to avoid point duplication when rasterizing
        multiple segments.
        :return: Array of contour points
        """
        raster_width = np.clip(np.abs(raster_width), 0, self.length)
        if not raster_width > 0:
            raise ValueError("'raster_width' is zero")

        num_raster_segments = np.round(self.length / raster_width)

        # normalized effective raster width
        nerw = 1. / num_raster_segments

        range_modifier = (0.5 - np.floor(
            np.abs(num_points_excluded_end))) * nerw

        multiplier = np.arange(0, 1 + range_modifier, nerw)
        weight_matrix = np.array([1 - multiplier, multiplier])

        return np.matmul(self._points, weight_matrix)

    def translate(self, vector):
        """
        Apply a translation to the segment.

        :param vector: Translation vector
        :return: ---
        """
        self._points += np.ndarray((2, 1), float, np.array(vector, float))


# ArcSegment ------------------------------------------------------------------

class ArcSegment:
    """Arc segment."""

    def __init__(self, point_start, point_end, point_center,
                 arc_winding_ccw=True):
        """
        Constructor.

        :param point_start: Starting point of the segment
        :param point_end: End point of the segment
        :param point_center: Center point of the arc
        :param: arc_winding_ccw: Specifies if the arcs winding order is
        counter-clockwise
        """
        if arc_winding_ccw:
            self._sign_arc_winding = 1
        else:
            self._sign_arc_winding = -1

        self._points = np.transpose(
            np.array([point_start, point_end, point_center], dtype=float))
        self._calculate_arc_parameters()

    def _calculate_arc_angle(self):
        """
        Calculate the arc angle.

        :return: ---
        """
        point_start = self._points[:, 0]
        point_end = self._points[:, 1]
        point_center = self._points[:, 2]

        # Calculate angle between vectors (always the smaller one)
        unit_center_start = tf.normalize(point_start - point_center)
        unit_center_end = tf.normalize(point_end - point_center)

        dot_unit = np.dot(unit_center_start, unit_center_end)
        angle_vecs = np.arccos(np.clip(dot_unit, -1, 1))

        sign_winding_points = vector_points_to_left_of_vector(
            unit_center_end, unit_center_start)

        if np.abs(sign_winding_points + self._sign_arc_winding) > 0:
            self._arc_angle = angle_vecs
        else:
            self._arc_angle = 2 * np.pi - angle_vecs

    def _calculate_arc_parameters(self):
        """
        Calculate radius, arc length and arc angle from the segments points.

        :return: ---
        """
        self._radius = np.linalg.norm(self._points[:, 0] - self._points[:, 2])
        self._calculate_arc_angle()
        self._arc_length = self._arc_angle * self._radius

        self._check_valid()

    def _check_valid(self):
        """
        Check if the segments data is valid.
        :return: ---
        """
        point_start = self._points[:, 0]
        point_end = self._points[:, 1]
        point_center = self._points[:, 2]

        vec_start_center = np.linalg.norm(point_start - point_center)
        vec_end_center = np.linalg.norm(point_end - point_center)

        if not math.isclose(vec_end_center - vec_start_center, 0):
            raise ValueError("Radius is not constant.")
        if math.isclose(self._arc_length, 0):
            raise Exception("Arc length is 0.")

    @property
    def arc_angle(self):
        """
        Get the arc angle.

        :return: Arc angle
        """
        return self._arc_angle

    @property
    def arc_length(self):
        """
        Get the arc length.

        :return: Arc length
        """
        return self._arc_length

    @property
    def point_center(self):
        """
        Get the center point of the segment.

        :return: Center point
        """
        return self._points[:, 2]

    @property
    def point_end(self):
        """
        Get the end point of the segment.

        :return: End point
        """
        return self._points[:, 1]

    @property
    def point_start(self):
        """
        Get the starting point of the segment.

        :return: Starting point
        """
        return self._points[:, 0]

    @property
    def radius(self):
        """
        Get the radius.

        :return: Radius
        """
        return self._radius

    def apply_transformation(self, matrix):
        """
        Apply a transformation to the segment.

        :param matrix: Transformation matrix
        :return: ---
        """
        matrix = np.array(matrix, float)
        self._points = np.matmul(matrix, self._points)
        self._sign_arc_winding *= reflection_sign(matrix)
        self._calculate_arc_parameters()

    def is_arc_winding_ccw(self):
        """
        Get True if the winding order is counter-clockwise. False if clockwise.

        :return: True or False
        """
        return self._sign_arc_winding > 0

    def rasterize(self, raster_width, num_points_excluded_end=0):
        """
        Create an array of points that describe the segments contour.

        The effective raster width may vary from the specified one,
        since the algorithm enforces constant distances between two
        raster points.

        :param raster_width: The desired distance between two raster points
        :param num_points_excluded_end: Specifies how many points from the
        end should be excluded from the rasterization. The main purpose of
        this parameter is to avoid point duplication when rasterizing
        multiple segments.
        :return: Array of contour points
        """
        point_start = self._points[:, 0]
        point_center = self._points[:, 2]
        vec_center_start = (point_start - point_center)

        raster_width = np.clip(raster_width, 0, self.arc_length)
        if not raster_width > 0:
            raise ValueError("'raster_width' is 0")

        num_raster_segments = int(np.round(self._arc_length / raster_width))
        delta_angle = self._arc_angle / num_raster_segments

        range_modifier = (0.5 - np.floor(
            np.abs(num_points_excluded_end))) * delta_angle
        max_angle = self._sign_arc_winding * (self._arc_angle + range_modifier)

        angles = np.arange(0, max_angle, self._sign_arc_winding * delta_angle)

        rotation_matrices = R.from_euler('z', angles).as_dcm()[:, 0:2, 0:2]

        data = np.matmul(rotation_matrices, vec_center_start) + point_center

        return data.transpose()

    def translate(self, vector):
        """
        Apply a translation to the segment.

        :param vector: Translation vector
        :return: ---
        """
        self._points += np.ndarray((2, 1), float, np.array(vector, float))


class Shape2D:
    """Defines a shape in 2 dimensions."""

    # Member variables --------------------------------------------------------

    min_segment_length = 1E-6
    tolerance_comparison = 1E-6

    # Member classes ----------------------------------------------------------

    class Segment:
        """Base class for segments."""

        def check_valid(self, point_start, point_end):
            """
            Check if the segments data is valid.

            Checks if the segments data is compatible with the passed start
            and end points. Raises an Exception if not.

            :param point_start: Starting point of the segment
            :param point_end: End point of the segment
            :return: ---
            """

        def translate(self, _unused_vector):
            """
            Apply a translation to the segment.

            :param _unused_vector: Translation vector
            :return: ---
            """

        def apply_transformation(self, _unused_transformation_matrix):
            """
            Apply a transformation to the segment.

            :param _unused_transformation_matrix: Transformation matrix
            :return: ---
            """

    class LineSegment(Segment):
        """Line segment."""

        @staticmethod
        def rasterize(raster_width, point_start, point_end):
            """
            Create an array of points that describe the segments contour.

            The effective raster width may vary from the specified one,
            since the algorithm enforces constant distances between two
            raster points.

            :param raster_width: The desired distance between two raster points
            :param point_start: Starting point of the segment
            :param point_end: End point of the segment
            :return: Array of contour points (3d)
            """
            length = np.linalg.norm(point_end - point_start)

            num_raster_segments = np.round(length / raster_width)
            nrw = 1. / num_raster_segments

            multiplier = np.arange(0, 1 - nrw / 2, nrw)[np.newaxis].transpose()

            raster_data = np.matmul((1 - multiplier),
                                    point_start[np.newaxis]) + np.matmul(
                multiplier, point_end[np.newaxis])

            return raster_data

    class ArcSegment(Segment):
        """Segment of a circle."""

        def __init__(self, point_center, arc_winding_ccw=True):
            """
            Constructor.

            :param point_center: Center point of the arc
            :param: arc_winding_ccw: Specifies if the arcs winding order is
            counter-clockwise
            """
            point_center = np.array(point_center, dtype=float)
            check_point_data_valid(point_center)

            self._point_center = point_center
            if arc_winding_ccw:
                self._sign_arc_winding = 1
            else:
                self._sign_arc_winding = -1

        def _arc_angle_and_length(self, vec_center_start, vec_center_end):
            """
            Calculate the arcs angle and the arc length.

            :param vec_center_start: Vector from the arcs center to the
            starting point
            :param vec_center_end: Vector from the arcs center to the end point
            :return: Array containing the arcs angle and arc length
            """
            sign_arc_winding = self._sign_arc_winding
            radius = np.linalg.norm(vec_center_start)

            # Calculate angle between vectors (always the smaller one)
            unit_center_start = vec_center_start / radius
            unit_center_end = vec_center_end / np.linalg.norm(vec_center_end)

            dot_unit = np.dot(unit_center_start, unit_center_end)
            angle_vecs = np.arccos(np.clip(dot_unit, -1, 1))

            sign_winding_points = vector_points_to_left_of_vector(
                vec_center_end, vec_center_start)

            if np.abs(sign_winding_points + sign_arc_winding) > 0:
                arc_angle = angle_vecs
            else:
                arc_angle = 2 * np.pi - angle_vecs

            arc_length = arc_angle * radius

            return [arc_angle, arc_length]

        def _rotation_angles(self, vec_center_start, vec_center_end,
                             raster_width):
            """
            Calculate the rotation angle of each raster point.

            The angles are referring to the vector to the starting point.

            :param vec_center_start: Vector from the arcs center to the
            starting point
            :param vec_center_end: Vector from the arcs center to the end point
            :param raster_width: Desired raster width
            :return: Array containing the rotation angles
            """
            sign = self._sign_arc_winding
            [angle_arc, arc_length] = self._arc_angle_and_length(
                vec_center_start,
                vec_center_end)

            num_raster_segments = int(np.round(arc_length / raster_width))
            delta_angle = angle_arc / num_raster_segments

            rotation_angles = np.arange(0,
                                        sign * (angle_arc - 0.5 * delta_angle),
                                        sign * delta_angle)

            return rotation_angles

        def _rasterize(self, vec_center_start, rotation_angles):
            """
            Create an array of points that describe the segments contour.

            :param vec_center_start: Vector from the arcs center to the
            starting point
            :param rotation_angles: Array containing the rotation angles
            :return: Array of contour points (3d)
            """
            vec_center_start_3d = vec_center_start[np.newaxis, :, np.newaxis]
            point_center_3d = self._point_center[:, np.newaxis]

            rotation_matrices = R.from_euler('z', rotation_angles).as_dcm()
            rotation_matrices = rotation_matrices[:, 0:2, 0:2]

            raster_data = np.matmul(rotation_matrices,
                                    vec_center_start_3d) + point_center_3d

            return raster_data[:, :, 0]

        def check_valid(self, point_start, point_end):
            """
            Check if the segments data is valid.

            Checks if the segments data is compatible with the passed start
            and end points. Raises an Exception if not.

            :param point_start: Starting point of the segment
            :param point_end: End point of the segment
            :return: ---
            """
            tolerance = Shape2D.tolerance_comparison
            point_center = self._point_center

            dist_start_center = np.linalg.norm(point_start - point_center)
            dist_end_center = np.linalg.norm(point_end - point_center)

            if not np.abs(dist_end_center - dist_start_center) <= tolerance:
                raise ValueError(
                    "Segment start and end points are not compatible with "
                    "given center of the arc.")

        def apply_transformation(self, transformation_matrix):
            """
            Apply a transformation to the segment.

            :param transformation_matrix: Transformation matrix
            :return: ---
            """
            self._point_center = np.matmul(transformation_matrix,
                                           self._point_center)
            self._sign_arc_winding *= reflection_multiplier(
                transformation_matrix)

        def rasterize(self, raster_width, point_start, point_end):
            """
            Create an array of points that describe the segments contour.

            The effective raster width may vary from the specified one,
            since the algorithm enforces constant distances between two
            raster points.

            :param raster_width: The desired distance between two raster points
            :param point_start: Starting point of the segment
            :param point_end: End point of the segment
            :return: Array of contour points (3d)
            """
            point_center = self._point_center

            vec_center_start = point_start - point_center
            vec_center_end = point_end - point_center

            rotation_angles = self._rotation_angles(vec_center_start,
                                                    vec_center_end,
                                                    raster_width)

            return self._rasterize(vec_center_start, rotation_angles)

        def translate(self, vector):
            """
            Apply a translation to the segment.

            :param vector: Translation vector
            :return: ---
            """
            self._point_center += vector

    def __init__(self, point0, point1, segment=LineSegment()):
        """
        Construct the shape with an initial segment.

        :param point0: first point
        :param point1: second point
        :param segment: segment
        """
        point0 = np.array(point0)
        point1 = np.array(point1)

        Shape2D._check_segment(segment, point0, point1)

        self._points = np.array([point0, point1], dtype=float)
        self._segments = [segment]

    # Private methods ---------------------------------------------------------

    @staticmethod
    def _check_segment(segment, point_start, point_end):
        """
        Check if segment is valid.

        :param segment: segment
        :param point_start: Starting point of the segment
        :param point_end: End point of the segment
        :return: ---
        """
        check_point_data_valid(point_start)
        check_point_data_valid(point_end)
        Shape2D._check_segment_length_valid(point_start, point_end)
        Shape2D._check_segment_type_valid(segment)
        segment.check_valid(point_start, point_end)

    @staticmethod
    def _check_segment_length_valid(point_start, point_end):
        """
        Check if a segment length is valid.

        :param point_start: Starting point of the segment
        :param point_end: End point of the segment
        :return: ---
        """
        diff = point_start - point_end
        if not np.linalg.norm(diff) >= Shape2D.min_segment_length:
            raise Exception("Segment length is too small.")

    @staticmethod
    def _check_segment_type_valid(segment):
        """
        Check if the segment type is valid.

        :return: ---
        """
        if not isinstance(segment, Shape2D.Segment):
            raise TypeError("Invalid segment type")

    # Public methods ----------------------------------------------------------

    def add_segment(self, point, segment=LineSegment()):
        """
        Add a new segment which is connected to previous one.

        :param point: end point of the new segment
        :param segment: segment
        :return: ---
        """
        point = np.array(point)

        Shape2D._check_segment(segment, self._points[-1], point)

        if self.is_shape_closed():
            raise ValueError("Shape is already closed")

        self._points = np.vstack((self._points, point))
        self._segments.append(segment)

    def apply_transformation(self, transformation_matrix):
        """
        Apply a transformation to the shape.

        :param transformation_matrix: Transformation matrix
        :return: ---
        """
        self._points = np.matmul(self._points,
                                 np.transpose(transformation_matrix))

        for i in range(self.num_segments()):
            self._segments[i].apply_transformation(transformation_matrix)

    def reflect(self, reflection_normal, distance_to_origin=0):
        """
        Apply a reflection at the given axis to the shape.

        :param reflection_normal: Normal of the reflection axis
        :param distance_to_origin: Distance of the reflection axis to the
        origin
        :return: ---
        """
        dot_product = np.dot(reflection_normal, reflection_normal)
        outer_product = np.outer(reflection_normal, reflection_normal)
        householder_matrix = np.identity(2) - 2 * outer_product / dot_product

        offset = np.array(reflection_normal) / np.sqrt(
            dot_product) * distance_to_origin

        self.translate(-offset)
        self.apply_transformation(householder_matrix)
        self.translate(offset)

    def translate(self, vector):
        """
        Apply a translation to the shape.

        :param vector: Translation vector
        :return: ---
        """
        self._points += vector

        for i in range(self.num_segments()):
            self._segments[i].translate(vector)

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

    def num_segments(self):
        """
        Get the number of segments of the shape.

        :return: number of segments
        """
        return len(self._segments)

    def num_points(self):
        """
        Get the number of points of the shape.

        :return: number of points
        """
        return self._points[:, 0].size

    @property
    def segments(self):
        return self._segments

    @property
    def points(self):
        return self._points

    def rasterize(self, raster_width):
        """
        Create an array of points that describe the shapes contour.

        The effective raster width may vary from the specified one,
        since the algorithm enforces constant distances between two
        raster points inside of each segment.

        :param raster_width: The desired distance between two raster points
        :return: Array of contour points (3d)
        """
        points = self._points
        segments = self._segments

        raster_data = np.empty([0, 2])
        for i in range(self.num_segments()):
            raster_data = np.vstack((raster_data,
                                     segments[i].rasterize(raster_width,
                                                           points[i],
                                                           points[i + 1])))

        raster_data = np.vstack(
            (raster_data, self._points[-1]))
        return raster_data
