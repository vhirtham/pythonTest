"""Provides classes to define lines and surfaces."""

import numpy as np
import math
import mypackage._utility as utils
import mypackage.transformations as tf
from scipy.spatial.transform import Rotation as Rot


# Helper functions ------------------------------------------------------------


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

    def __init__(self, points):
        """
        Constructor.

        :param points: 2x2 matrix of points. The first column is the
        starting point and the second column the end point.
        """
        points = np.array(points, float)
        if not len(points.shape) == 2:
            raise ValueError("'points' must be a 2d array/matrix.")
        if not (points.shape[0] == 2 and points.shape[1] == 2):
            raise ValueError("'points' is not a 2x2 matrix.")

        self._points = points
        self._calculate_length()

    def _calculate_length(self):
        """
        Calculate the segment length from its points.

        :return: ---
        """
        self._length = np.linalg.norm(self._points[:, 1] - self._points[:, 0])
        if math.isclose(self._length, 0):
            raise ValueError("Segment length is 0.")

    @classmethod
    def construct_from_points(cls, point_start, point_end):
        """
        Construct a line segment from two points.

        :param point_start: Starting point of the segment
        :param point_end: End point of the segment
        :return: Line segment
        """
        points = np.transpose(np.array([point_start, point_end], dtype=float))
        return cls(points)

    @classmethod
    def linear_interpolation(cls, segment_a, segment_b, weight):
        """
        Interpolate two line segments linearly.

        :param segment_a: First segment
        :param segment_b: Second segment
        :param weight: Weighting factor in the range [0 .. 1] where 0 is
        segment a and 1 is segment b
        :return: Interpolated segment
        """
        if not isinstance(segment_a, cls) or not isinstance(segment_b, cls):
            raise TypeError("Parameters a and b must both be line segments.")

        weight = np.clip(weight, 0, 1)
        points = (1 - weight) * segment_a.points + weight * segment_b.points
        return cls(points)

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

    @property
    def points(self):
        """
        Get the segments points in form of a 2x2 matrix.

        The first column represents the starting point and the second one
        the end point.

        :return: 2x2 matrix containing the segments points
        """
        return self._points

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

    def __init__(self, points, arc_winding_ccw=True):
        """
        Constructor.

        :param points: 2x3 matrix of points. The first column is the
        starting point, the second column the end point and the last the
        center point.
        :param: arc_winding_ccw: Specifies if the arcs winding order is
        counter-clockwise
        """
        points = np.array(points, float)
        if not len(points.shape) == 2:
            raise ValueError("'points' must be a 2d array/matrix.")
        if not (points.shape[0] == 2 and points.shape[1] == 3):
            raise ValueError("'points' is not a 2x3 matrix.")

        if arc_winding_ccw:
            self._sign_arc_winding = 1
        else:
            self._sign_arc_winding = -1
        self._points = points

        self._arc_angle = None
        self._arc_length = None
        self._radius = None
        self._calculate_arc_parameters()

    def _calculate_arc_angle(self):
        """
        Calculate the arc angle.

        :return: ---
        """
        point_start = self.point_start
        point_end = self.point_end
        point_center = self.point_center

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
        point_start = self.point_start
        point_end = self.point_end
        point_center = self.point_center

        radius_start_center = np.linalg.norm(point_start - point_center)
        radius_end_center = np.linalg.norm(point_end - point_center)
        radius_diff = radius_end_center - radius_start_center

        if not math.isclose(radius_diff, 0, abs_tol=1E-9):
            raise ValueError("Radius is not constant.")
        if math.isclose(self._arc_length, 0):
            raise Exception("Arc length is 0.")

    @classmethod
    def construct_from_points(cls, point_start, point_end, point_center,
                              arc_winding_ccw=True):
        """
        Construct an arc segment from three points.

        :param point_start: Starting point of the segment
        :param point_end: End point of the segment
        :param point_center: Center point of the arc
        :param arc_winding_ccw: Specifies if the arcs winding order is
        counter-clockwise
        :return: Arc segment
        """
        points = np.transpose(
            np.array([point_start, point_end, point_center], dtype=float))
        return cls(points, arc_winding_ccw)

    @classmethod
    def linear_interpolation(cls, segment_a, segment_b, weight):
        """
        Interpolate two arc segments linearly.

        :param segment_a: First segment
        :param segment_b: Second segment
        :param weight: Weighting factor in the range [0 .. 1] where 0 is
        segment a and 1 is segment b
        :return: Interpolated segment
        """
        # implementation ->segment start and end have to be interpolated
        # linearly. Otherwise there might occur gaps in interpolated shapes
        # at the connecting segment points --- the center point has to be
        # determined automatically. 2 ways -> linear angle interpolation or
        # linear radius interpolation
        raise Exception("Not implemented.")

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
        point_start = self.point_start
        point_center = self.point_center
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

        rotation_matrices = Rot.from_euler('z', angles).as_dcm()[:, 0:2, 0:2]

        data = np.matmul(rotation_matrices, vec_center_start) + point_center

        return data.transpose()

    def translate(self, vector):
        """
        Apply a translation to the segment.

        :param vector: Translation vector
        :return: ---
        """
        self._points += np.ndarray((2, 1), float, np.array(vector, float))


# Shape class -----------------------------------------------------------------

class Shape:
    """Defines a shape in 2 dimensions."""

    def __init__(self, segments=None):
        """
        Constructor.

        :param segments: Single segment or list of segments
        """
        segments = utils.to_list(segments)
        self._check_segments_connected(segments)
        self._segments = segments

    @staticmethod
    def _check_segments_connected(segments):
        """
        Check if all segments are connected to each other.

        The start point of a segment must be identical to the end point of
        the previous segment.

        :param segments: List of segments
        :return: ---
        """
        for i in range(len(segments) - 1):
            if not utils.vector_is_close(segments[i].point_end,
                                         segments[i + 1].point_start):
                raise Exception("Segments are not connected.")

    @classmethod
    def interpolate(cls, shape_a, shape_b, weight, interpolation_schemes):
        """
        Interpolate 2 shapes.

        :param shape_a: First shape
        :param shape_b: Second shape
        :param weight: Weighting factor in the range [0 .. 1] where 0 is
        shape a and 1 is shape b
        :param interpolation_schemes: List of interpolation schemes for each
        segment of the shape.
        :return: Interpolated shape
        """
        if not shape_a.num_segments == shape_b.num_segments:
            raise Exception("Number of segments differ.")

        weight = np.clip(weight, 0, 1)

        segments_c = []
        for i in range(shape_a.num_segments):
            segments_c += [interpolation_schemes[i](shape_a.segments[i],
                                                    shape_b.segments[i],
                                                    weight)]
        return cls(segments_c)

    @classmethod
    def linear_interpolation(cls, shape_a, shape_b, weight):
        """
        Interpolate 2 shapes linearly.

        Each segment is interpolated individually, using the corresponding
        linear segment interpolation.

        :param shape_a: First shape
        :param shape_b: Second shape
        :param weight: Weighting factor in the range [0 .. 1] where 0 is
        shape a and 1 is shape b
        :return: Interpolated shape
        """
        interpolation_schemes = []
        for i in range(shape_a.num_segments):
            interpolation_schemes += [shape_a.segments[i].linear_interpolation]

        return cls.interpolate(shape_a, shape_b, weight, interpolation_schemes)

    @property
    def num_segments(self):
        """
        Get the number of segments of the shape.

        :return: number of segments
        """
        return len(self._segments)

    @property
    def segments(self):
        """
        Get the shape's segments.

        :return: List of segments
        """
        return self._segments

    def add_segments(self, segments):
        """
        Add segments to the shape.

        :param segments: Single segment or list of segments
        :return: ---
        """
        segments = utils.to_list(segments)
        if self.num_segments > 0:
            self._check_segments_connected([self.segments[-1], segments[0]])
        self._check_segments_connected(segments)
        self._segments += segments

    def apply_transformation(self, transformation_matrix):
        """
        Apply a transformation to the shape.

        :param transformation_matrix: Transformation matrix
        :return: ---
        """
        for i in range(self.num_segments):
            self._segments[i].apply_transformation(transformation_matrix)

    def reflect(self, reflection_normal, distance_to_origin=0):
        """
        Apply a reflection at the given axis to the shape.

        :param reflection_normal: Normal of the reflection axis
        :param distance_to_origin: Distance of the reflection axis to the
        origin
        :return: ---
        """
        normal = np.array(reflection_normal, float)
        dot_product = np.dot(normal, normal)
        outer_product = np.outer(normal, normal)
        householder_matrix = np.identity(2) - 2 / dot_product * outer_product

        offset = normal / np.sqrt(dot_product) * distance_to_origin

        self.translate(-offset)
        self.apply_transformation(householder_matrix)
        self.translate(offset)

    def translate(self, vector):
        """
        Apply a translation to the shape.

        :param vector: Translation vector
        :return: ---
        """
        for i in range(self.num_segments):
            self._segments[i].translate(vector)

    def rasterize(self, raster_width):
        """
        Create an array of points that describe the shapes contour.

        The effective raster width may vary from the specified one,
        since the algorithm enforces constant distances between two
        raster points inside of each segment.

        :param raster_width: The desired distance between two raster points
        :return: Array of contour points (3d)
        """
        segments = self._segments

        raster_data = np.empty([2, 0])
        for i in range(self.num_segments):
            segment_data = segments[i].rasterize(raster_width, 1)
            raster_data = np.hstack((raster_data, segment_data))

        last_point = segments[-1].point_end[:, np.newaxis]
        raster_data = np.hstack((raster_data, last_point))
        return raster_data
