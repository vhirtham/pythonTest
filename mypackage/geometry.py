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
            pass

    class LineSegment(Segment):
        """Line segment."""

    class ArcSegment(Segment):
        """Segment of a circle."""

        def __init__(self, point_center):
            """
            Constructor.

            :param point_center: Center point of the arc
            """
            point_center = np.array(point_center)
            Shape2D._check_point_data_valid(point_center)

            self._point_center = point_center

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

    # Private methods ---------------------------------------------------------

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

        self._points = np.array([point0, point1])
        self._segments = []
        self._append_segments(segment)

    def _append_segments(self, segment):
        """
        Append the internal segment array.

        :param segment: segment that should be added
        :return: ---
        """
        if Shape2D.is_segment_type_valid(segment):
            self._segments.append(segment)
        else:
            raise ValueError("Unknown segment type")

    def _check_segment(segment, point_start, point_end):
        """
        Check if segment is valid.

        :param segment: segment
        :param point_start: Starting point of the segment
        :param point_end: End point of the segment
        :return: ---
        """
        Shape2D._check_point_data_valid(point_start)
        Shape2D._check_point_data_valid(point_end)
        Shape2D._check_segment_length_valid(point_start, point_end)
        segment.check_valid(point_start, point_end)

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

    def _check_point_data_valid(point):
        """
        Check if the data of a point is valid.

        :param point: Point that should be checked
        :return: ---
        """
        if not (np.ndim(point) == 1 and point.size == 2):
            raise Exception(
                "Point data is invalid. Must be an array with 2 values.")

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

    def is_point_included(self, point):
        """
        Check if a point is already part of the shape.

        :param point: Point which should be checked
        :return: True or False
        """
        return is_row_in_array(point, self._points)

    def is_segment_type_valid(segment):
        """
        Check if the passed segment type is valid.

        :param segment: segment
        :return: True or False
        """
        if isinstance(segment, Shape2D.Segment):
            return True
        else:
            return False

    def is_shape_closed(self):
        """
        Check if the shape is already closed.

        :return: True or False
        """
        return is_row_in_array(self._points[-1, :], self._points[:-1])
