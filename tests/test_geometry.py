import pytest
import mypackage.geometry as geo
import numpy as np


def test_shape2d_construction():
    # Test Exception: Segment length too small
    with pytest.raises(Exception):
        geo.Shape2D([0, 0], [0, 0])

    # Test Exception: Invalid point format
    with pytest.raises(Exception):
        geo.Shape2D([0, 0], [1])
    with pytest.raises(Exception):
        geo.Shape2D([0, 0, 4], [1, 1])

    # Test Exception: Invalid shape type
    with pytest.raises(Exception):
        geo.Shape2D([0, 0], [0, 1], "wrong type")

    # Create shape
    geo.Shape2D([0, 0], [0, 1])


def test_shape2d_boolean_functions():
    shape = geo.Shape2D([0, 0], [0, 1])

    # Point included or not
    assert (shape.is_point_included([0, 1]))
    assert (not shape.is_point_included([5, 1]))


def test_shape2d_segment_addition():
    # Create shape and add segments
    shape = geo.Shape2D([0, 0], [0, 1])
    shape.add_segment([2, 2])
    shape.add_segment([1, 0])

    # Test Exception: Invalid point format
    with pytest.raises(Exception):
        shape.add_segment(0)
    # Test Exception: Invalid shape type
    with pytest.raises(Exception):
        shape.add_segment([0, 0], "wrong type")

    # Close segment
    shape.add_segment([0, 0])

    # Test Exception: Shape is already closed
    with pytest.raises(ValueError):
        shape.add_segment([1, 0])

    # Number of segments has to be one less than the number of points
    assert shape.num_segments() == shape.num_points() - 1


def test_shape2d_with_arc_segment():
    # Invalid center point
    with pytest.raises(ValueError):
        geo.Shape2D([0, 0], [1, 1], geo.Shape2D.ArcSegment([0, 1.1]))

    shape = geo.Shape2D([0, 0], [1, 1], segment=geo.Shape2D.ArcSegment([0, 1]))
    shape.add_segment([2, 2], segment=geo.Shape2D.ArcSegment([2, 1]))
    # Invalid center point
    with pytest.raises(ValueError):
        shape.add_segment([3, 1], segment=geo.Shape2D.ArcSegment([2.1, 1]))


def default_rasterization_tests(data, raster_width, point_start, point_end):
    # Check if first point of the data are identical to the segment start
    assert np.linalg.norm(data[0, 0:2] - point_start) < 1E-9

    num_data_points = data[:, 0].size
    for i in range(num_data_points):
        point = data[i]

        # Check that z-component is 0
        assert point[2] == 0.

        # Check if the raster width is close to the specified value
        if i < num_data_points - 1:
            next_point = data[i + 1, 0:2]
        else:
            next_point = point_end

        raster_width_eff = np.linalg.norm(next_point - point[0:2])
        assert np.abs(raster_width_eff - raster_width) < 0.1 * raster_width


def test_line_segment_rasterizaion():
    point_start = np.array([3, -5])
    point_end = np.array([-4, 1])
    raster_width = 0.2
    vec_start_end = point_end - point_start

    line_segment = geo.Shape2D.LineSegment()
    data = line_segment.rasterize(raster_width, point_start, point_end)

    # Perform standard segment rasterization tests
    default_rasterization_tests(data, raster_width, point_start, point_end)

    num_data_points = data[:, 0].size
    for i in range(num_data_points):
        point = data[i]

        # Check if point is on line
        vec_start_point = point[0:2] - point_start
        assert np.abs(np.linalg.det([vec_start_end, vec_start_point])) < 1E-6

        # Check if point lies between start and end
        dot_product = np.dot(vec_start_point, vec_start_end)
        assert dot_product >= 0
        assert dot_product < np.dot(vec_start_end, vec_start_end)


def arc_segment_test(point_center, point_start, point_end, raster_width,
                     winding_ccw, check_winding):
    point_center = np.array(point_center)
    point_start = np.array(point_start)
    point_end = np.array(point_end)

    radius_arc = np.linalg.norm(point_start - point_center)

    arc_segment = geo.Shape2D.ArcSegment(point_center, winding_ccw=winding_ccw)
    arc_segment.check_valid(point_start, point_end)

    data = arc_segment.rasterize(raster_width, point_start, point_end)

    # Perform standard segment rasterization tests
    default_rasterization_tests(data, raster_width, point_start, point_end)

    num_data_points = data[:, 0].size
    for i in range(num_data_points):
        point = data[i]

        # Check if points are not rasterized clockwise
        assert (check_winding(point_start, point_center))

        # Check that points have the correct distance to the arcs center
        distance_center_point = np.linalg.norm(point[0:2] - point_center)
        assert np.abs(distance_center_point - radius_arc) < 1E-6


def test_arc_segment_rasterizaion():
    # center right of segment line
    # ----------------------------

    point_center = [3, 2]
    point_start = [1, 2]
    point_end = [3, 4]
    raster_width = 0.2

    def in_second_quadrant(p, c):
        return p[0] <= c[0] and p[1] >= c[1]

    def not_in_second_quadrant(p, c):
        return not (p[0] < c[0] and p[1] > c[1])

    arc_segment_test(point_center, point_start, point_end, raster_width, False,
                     in_second_quadrant)
    arc_segment_test(point_center, point_start, point_end, raster_width, True,
                     not_in_second_quadrant)

    # center left of segment line
    # ----------------------------

    point_center = [-4, -7]
    point_start = [-4, -2]
    point_end = [-9, -7]
    raster_width = 0.1

    arc_segment_test(point_center, point_start, point_end, raster_width, False,
                     not_in_second_quadrant)
    arc_segment_test(point_center, point_start, point_end, raster_width, True,
                     in_second_quadrant)


def test_shape2d_rasterization():
    points = np.array([[0, 0],
                       [0, 1],
                       [1, 1],
                       [1, 0]])
    raster_width = 0.2

    shape = geo.Shape2D(points[0], points[1])
    shape.add_segment(points[2])
    shape.add_segment(points[3])

    data = shape.rasterize(raster_width)

    # Segment points must be included
    for point in points:
        assert geo.is_row_in_array(point, data[:, 0:2])

    # check effective raster width
    for i in range(1, data[:, 0].size):
        raster_width_eff = np.linalg.norm(data[i] - data[i - 1])
        assert np.abs(raster_width_eff - raster_width) < 0.1 * raster_width


def test_line_segment_copy_and_reflect():
    reflection_matrix = [[0, 1], [1, 0]]

    segment = geo.Shape2D.LineSegment()
    segment_copy = segment.copy_and_transform(reflection_matrix)

    # ensure that segment_copy is a deepcopy of segment
    assert segment_copy is not segment


def test_arc_segment_copy_and_reflect():
    reflection_matrix = np.array([[0, 1], [1, 0]])
    offset = np.array([2, -1])
    point_center = [2, 3]

    segment = geo.Shape2D.ArcSegment(point_center)
    segment_copy = segment.copy_and_transform(reflection_matrix, -offset,
                                              offset)
    segment_copy2 = segment_copy.copy_and_transform(reflection_matrix, -offset,
                                                    offset)

    # ensure that segment_copy is a deepcopy of segment
    assert segment_copy is not segment
    assert segment_copy2 is not segment
    assert segment_copy is not segment_copy2

    # Check if new center point is correct
    assert segment_copy._point_center[0] == 6
    assert segment_copy._point_center[1] == -1
    assert segment_copy2._point_center[0] == point_center[0]
    assert segment_copy2._point_center[1] == point_center[1]

    # Check that winding order is changed
    assert segment_copy._sign_winding == segment._sign_winding * -1
    assert segment_copy2._sign_winding == segment._sign_winding
