import pytest
import mypackage.geometry as geo
import numpy as np
import math
import copy
import tests.helpers as helper


def test_vector_points_to_left_of_vector():
    assert geo.vector_points_to_left_of_vector([-0.1, 1], [0, 1]) > 0
    assert geo.vector_points_to_left_of_vector([-0.1, -1], [0, 1]) > 0
    assert geo.vector_points_to_left_of_vector([3, 5], [1, 0]) > 0
    assert geo.vector_points_to_left_of_vector([-3, 5], [1, 0]) > 0
    assert geo.vector_points_to_left_of_vector([0, -0.1], [-4, 2]) > 0
    assert geo.vector_points_to_left_of_vector([-1, -0.1], [-4, 2]) > 0

    assert geo.vector_points_to_left_of_vector([0.1, 1], [0, 1]) < 0
    assert geo.vector_points_to_left_of_vector([0.1, -1], [0, 1]) < 0
    assert geo.vector_points_to_left_of_vector([3, -5], [1, 0]) < 0
    assert geo.vector_points_to_left_of_vector([-3, -5], [1, 0]) < 0
    assert geo.vector_points_to_left_of_vector([0, 0.1], [-4, 2]) < 0
    assert geo.vector_points_to_left_of_vector([1, -0.1], [-4, 2]) < 0

    assert geo.vector_points_to_left_of_vector([4, 4], [2, 2]) == 0
    assert geo.vector_points_to_left_of_vector([-4, -4], [2, 2]) == 0


def test_point_left_of_line():
    line_start = np.array([2, 3])
    line_end = np.array([5, 6])
    assert geo.point_left_of_line([-8, 10], line_start, line_end) > 0
    assert geo.point_left_of_line([3, 0], line_start, line_end) < 0
    assert geo.point_left_of_line(line_start, line_start, line_end) == 0

    line_start = np.array([2, 3])
    line_end = np.array([1, -4])
    assert geo.point_left_of_line([3, 0], line_start, line_end) > 0
    assert geo.point_left_of_line([-8, 10], line_start, line_end) < 0
    assert geo.point_left_of_line(line_start, line_start, line_end) == 0


def test_reflection_multiplier():
    assert geo.reflection_multiplier([[-1, 0], [0, 1]]) == -1
    assert geo.reflection_multiplier([[1, 0], [0, -1]]) == -1
    assert geo.reflection_multiplier([[0, 1], [1, 0]]) == -1
    assert geo.reflection_multiplier([[0, -1], [-1, 0]]) == -1
    assert geo.reflection_multiplier([[-4, 0], [0, 2]]) == -1
    assert geo.reflection_multiplier([[6, 0], [0, -4]]) == -1
    assert geo.reflection_multiplier([[0, 3], [8, 0]]) == -1
    assert geo.reflection_multiplier([[0, -3], [-2, 0]]) == -1

    assert geo.reflection_multiplier([[1, 0], [0, 1]]) == 1
    assert geo.reflection_multiplier([[-1, 0], [0, -1]]) == 1
    assert geo.reflection_multiplier([[0, -1], [1, 0]]) == 1
    assert geo.reflection_multiplier([[0, 1], [-1, 0]]) == 1
    assert geo.reflection_multiplier([[5, 0], [0, 6]]) == 1
    assert geo.reflection_multiplier([[-3, 0], [0, -7]]) == 1
    assert geo.reflection_multiplier([[0, -8], [9, 0]]) == 1
    assert geo.reflection_multiplier([[0, 3], [-2, 0]]) == 1

    with pytest.raises(Exception):
        geo.reflection_multiplier([[0, 0], [0, 0]])
    with pytest.raises(Exception):
        geo.reflection_multiplier([[1, 0], [0, 0]])
    with pytest.raises(Exception):
        geo.reflection_multiplier([[2, 2], [1, 1]])


# helper for segment tests ----------------------------------------------------

def default_segment_rasterization_tests(segment, raster_width, point_start,
                                        point_end):
    data = segment.rasterize(raster_width)

    # check dimensions are correct
    assert len(data.shape) == 2

    point_dimension = data.shape[0]
    num_points = data.shape[1]
    assert point_dimension == 2

    # Check if first and last point of the data are identical to the segment
    # start and end
    helper.check_vectors_identical(data[:, 0], point_start)
    helper.check_vectors_identical(data[:, -1], point_end)

    for i in range(num_points - 1):
        point = data[:, i]
        next_point = data[:, i + 1]

        raster_width_eff = np.linalg.norm(next_point - point)
        assert np.abs(raster_width_eff - raster_width) < 0.1 * raster_width

    # check rasterization with excluded points
    data_m2 = segment.rasterize(raster_width, 2)

    num_points_m2 = data_m2.shape[1]

    assert num_points - 2 == num_points_m2

    for i in range(num_points_m2):
        helper.check_vectors_identical(data[:, i], data_m2[:, i])

    # check that rasterization with to large raster width still works
    data_200 = segment.rasterize(200)

    num_points_200 = data_200.shape[1]
    assert num_points_200 == 2
    helper.check_vectors_identical(point_start, data_200[:, 0])
    helper.check_vectors_identical(point_end, data_200[:, 1])

    # check exceptions
    with pytest.raises(ValueError):
        segment.rasterize(0)


# test LineSegment ------------------------------------------------------------

def test_line_segment():
    segment = geo.LineSegment([3, 3], [4, 5])
    assert math.isclose(segment.length, np.sqrt(5))

    with pytest.raises(ValueError):
        geo.LineSegment([0, 1], [0, 1])


def test_line_segment_rasterization():
    raster_width = 0.1

    point_start = np.array([3, 3])
    point_end = np.array([4, 5])
    vec_start_end = point_end - point_start
    unit_vec_start_end = vec_start_end / np.linalg.norm(vec_start_end)

    segment = geo.LineSegment(point_start, point_end)

    # perform default tests
    default_segment_rasterization_tests(segment, raster_width, point_start,
                                        point_end)

    # check that points lie between start and end
    raster_data = segment.rasterize(raster_width)
    num_points = raster_data.shape[1]
    for i in np.arange(1, num_points - 1, 1):
        point = raster_data[:, i]

        vec_start_point = point - point_start
        unit_vec_start_point = vec_start_point / np.linalg.norm(
            vec_start_point)

        assert math.isclose(np.dot(unit_vec_start_point, unit_vec_start_end),
                            1)


# test ArcSegment ------------------------------------------------------------


def arc_segment_test(point_center, point_start, point_end, raster_width,
                     arc_winding_ccw, check_winding):
    point_center = np.array(point_center)
    point_start = np.array(point_start)
    point_end = np.array(point_end)

    radius_arc = np.linalg.norm(point_start - point_center)

    arc_segment = geo.ArcSegment(point_start, point_end, point_center,
                                 arc_winding_ccw=arc_winding_ccw)

    # Perform standard segment rasterization tests
    default_segment_rasterization_tests(arc_segment, raster_width, point_start,
                                        point_end)

    data = arc_segment.rasterize(raster_width)

    num_points = data.shape[1]
    for i in range(num_points):
        point = data[:, i]

        # Check that winding is correct
        assert (check_winding(point, point_center))

        # Check that points have the correct distance to the arcs center
        distance_center_point = np.linalg.norm(point - point_center)
        assert math.isclose(distance_center_point, radius_arc, abs_tol=1E-6)


def test_arc_segment_construction():
    segment_cw = geo.ArcSegment([3, 3], [6, 6], [6, 3], False)
    segment_ccw = geo.ArcSegment([3, 3], [6, 6], [6, 3], True)

    assert not segment_cw.is_arc_winding_ccw()
    assert segment_ccw.is_arc_winding_ccw()

    assert math.isclose(segment_cw.radius, 3)
    assert math.isclose(segment_ccw.radius, 3)

    assert math.isclose(segment_cw.arc_angle, 1 / 2 * np.pi)
    assert math.isclose(segment_ccw.arc_angle, 3 / 2 * np.pi)

    assert math.isclose(segment_cw.arc_length, 3 / 2 * np.pi)
    assert math.isclose(segment_ccw.arc_length, 9 / 2 * np.pi)

    # check exceptions ------------------------------------
    # radius differs
    with pytest.raises(Exception):
        geo.ArcSegment([3, 3], [6, 10], [6, 3], False)
    # radius is zero
    with pytest.raises(Exception):
        geo.ArcSegment([3, 3], [3, 3], [3, 3], False)
    # arc length zero
    with pytest.raises(Exception):
        geo.ArcSegment([3, 3], [3, 3], [6, 3], False)
    with pytest.raises(Exception):
        geo.ArcSegment([3, 3], [3, 3], [6, 3], True)


def test_arc_segment_rasterization():
    # center right of segment line
    # ----------------------------

    point_center = [3, 2]
    point_start = [1, 2]
    point_end = [3, 4]
    raster_width = 0.2

    def in_second_quadrant(p, c):
        return p[0] - 1E-9 <= c[0] and p[1] >= c[1] - 1E-9

    def not_in_second_quadrant(p, c):
        return not (p[0] + 1E-9 < c[0] and p[1] > c[1] + 1E-9)

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

    # center on segment line
    # ----------------------

    point_center = [3, 2]
    point_start = [2, 2]
    point_end = [4, 2]
    raster_width = 0.1

    def not_below_center(p, c):
        return p[1] >= c[1] - 1E-9

    def not_above_center(p, c):
        return p[1] - 1E-9 <= c[1]

    arc_segment_test(point_center, point_start, point_end, raster_width, False,
                     not_below_center)
    arc_segment_test(point_center, point_start, point_end, raster_width, True,
                     not_above_center)

    # special testcase
    # ----------------
    # In a previous version the unit vectors to the start and end point were
    # calculated using the norm of the vector to the start, since both
    # vector length should be identical (radius). However, floating point
    # errors caused the dot product to get greater than 1. In result,
    # the angle between both vectors could not be calculated using the arccos.
    # This test case will fail in this case.
    point_center = [0, 0]
    point_start = [-6.6 - 2.8]
    point_end = [-6.4 - 4.2]
    raster_width = 0.1

    arc_segment = geo.Shape2D.ArcSegment(point_center)
    arc_segment.rasterize(raster_width, point_start, point_end)


# test Shape2d ----------------------------------------------------------------

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


def default_rasterization_tests_old(data, raster_width, point_start,
                                    point_end):
    # Check if first point of the data are identical to the segment start
    assert np.linalg.norm(data[0, 0:2] - point_start) < 1E-9

    point_dimension = data[0, :].size
    num_data_points = data[:, 0].size

    assert point_dimension == 2

    for i in range(num_data_points):
        point = data[i]

        # Check if the raster width is close to the specified value
        if i < num_data_points - 1:
            next_point = data[i + 1, 0:2]
        else:
            next_point = point_end

        raster_width_eff = np.linalg.norm(next_point - point[0:2])
        assert np.abs(raster_width_eff - raster_width) < 0.1 * raster_width


def test_line_segment_rasterizaion_old():
    point_start = np.array([3, -5])
    point_end = np.array([-4, 1])
    raster_width = 0.2
    vec_start_end = point_end - point_start

    line_segment = geo.Shape2D.LineSegment()
    data = line_segment.rasterize(raster_width, point_start, point_end)

    # Perform standard segment rasterization tests
    default_rasterization_tests_old(data, raster_width, point_start, point_end)

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


def arc_segment_test_old(point_center, point_start, point_end, raster_width,
                         arc_winding_ccw, check_winding):
    point_center = np.array(point_center)
    point_start = np.array(point_start)
    point_end = np.array(point_end)

    radius_arc = np.linalg.norm(point_start - point_center)

    arc_segment = geo.Shape2D.ArcSegment(point_center,
                                         arc_winding_ccw=arc_winding_ccw)
    arc_segment.check_valid(point_start, point_end)

    data = arc_segment.rasterize(raster_width, point_start, point_end)

    # Perform standard segment rasterization tests
    default_rasterization_tests_old(data, raster_width, point_start, point_end)

    num_data_points = data[:, 0].size
    for i in range(num_data_points):
        point = data[i]

        # Check if points are not rasterized clockwise
        assert (check_winding(point[0:2], point_center))

        # Check that points have the correct distance to the arcs center
        distance_center_point = np.linalg.norm(point[0:2] - point_center)
        assert np.abs(distance_center_point - radius_arc) < 1E-6


def test_arc_segment_rasterizaion_old():
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

    arc_segment_test_old(point_center, point_start, point_end, raster_width,
                         False,
                         in_second_quadrant)
    arc_segment_test_old(point_center, point_start, point_end, raster_width,
                         True,
                         not_in_second_quadrant)

    # center left of segment line
    # ----------------------------

    point_center = [-4, -7]
    point_start = [-4, -2]
    point_end = [-9, -7]
    raster_width = 0.1

    arc_segment_test_old(point_center, point_start, point_end, raster_width,
                         False,
                         not_in_second_quadrant)
    arc_segment_test_old(point_center, point_start, point_end, raster_width,
                         True,
                         in_second_quadrant)

    # center on segment line
    # ----------------------

    point_center = [3, 2]
    point_start = [2, 2]
    point_end = [4, 2]
    raster_width = 0.1

    def not_below_center(p, c):
        return p[1] >= c[1]

    def not_above_center(p, c):
        return p[1] <= c[1]

    arc_segment_test_old(point_center, point_start, point_end, raster_width,
                         False,
                         not_below_center)
    arc_segment_test_old(point_center, point_start, point_end, raster_width,
                         True,
                         not_above_center)

    # special testcase
    # ----------------
    # In a previous version the unit vectors to the start and end point were
    # calculated using the norm of the vector to the start, since both
    # vector length should be identical (radius). However, floating point
    # errors caused the dot product to get greater than 1. In result,
    # the angle between both vectors could not be calculated using the arccos.
    # This test case will fail in this case.
    point_center = [0, 0]
    point_start = [-6.6 - 2.8]
    point_end = [-6.4 - 4.2]
    raster_width = 0.1

    arc_segment = geo.Shape2D.ArcSegment(point_center)
    arc_segment.rasterize(raster_width, point_start, point_end)


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


def test_arc_segment_transformations():
    # create arc segment
    point_center = [2, 3]
    segment = geo.Shape2D.ArcSegment(point_center)

    # check transformation with reflection
    reflection_matrix = np.array([[0, 1], [1, 0]])
    translation_pre = np.array([-1, 1])
    translation_post = np.array([2, 1])
    segment_copy = copy.deepcopy(segment)

    segment_copy.translate(translation_pre)
    assert segment_copy._point_center[0] == 1
    assert segment_copy._point_center[1] == 4

    segment_copy.apply_transformation(reflection_matrix)
    assert segment_copy._point_center[0] == 4
    assert segment_copy._point_center[1] == 1
    # Check that winding order is NOT changed
    assert segment_copy._sign_arc_winding == segment._sign_arc_winding * -1

    segment_copy.translate(translation_post)
    # Check if new center point is correct
    assert segment_copy._point_center[0] == 6
    assert segment_copy._point_center[1] == 2

    # check transformation without reflection
    rotation_matrix = np.array([[0, 1], [-1, 0]])
    translation_pre = np.array([3, -2])
    translation_post = np.array([-3, -3])

    segment_copy = copy.deepcopy(segment)
    segment_copy.translate(translation_pre)
    assert segment_copy._point_center[0] == 5
    assert segment_copy._point_center[1] == 1

    segment_copy.apply_transformation(rotation_matrix)
    assert segment_copy._point_center[0] == 1
    assert segment_copy._point_center[1] == -5
    # Check that winding order is NOT changed
    assert segment_copy._sign_arc_winding == segment._sign_arc_winding

    segment_copy.translate(translation_post)
    assert segment_copy._point_center[0] == -2
    assert segment_copy._point_center[1] == -8


def default_test_shape():
    points = np.array([[3, 4],
                       [5, 0],
                       [11, 3]])
    point_center = np.array([6, 3])

    # create shape
    arc_segment = geo.Shape2D.ArcSegment(point_center)
    shape = geo.Shape2D(points[0], points[1], arc_segment)
    shape.add_segment(points[2])

    return shape


def test_shape2d_translation():
    def check_point(point, point_ref, translation):
        assert point[0] - point_ref[0] - translation[0] < 1E-9
        assert point[1] - point_ref[1] - translation[1] < 1E-9

    translation = [3, 4]

    shape_ref = default_test_shape()
    shape = copy.deepcopy(shape_ref)

    # apply translation
    shape.translate(translation)

    for i in range(shape.num_points()):
        check_point(shape._points[i], shape_ref._points[i], translation)

    arc_segment = shape._segments[0]
    arc_segment_ref = shape_ref._segments[0]

    check_point(arc_segment._point_center, arc_segment_ref._point_center,
                translation)
    assert arc_segment._sign_arc_winding == arc_segment_ref._sign_arc_winding


def test_shape2d_transformation():
    # without reflection
    def check_point_rotation(point, point_ref):
        assert point[0] == point_ref[1]
        assert point[1] == -point_ref[0]

    rotation_matrix = np.array([[0, 1], [-1, 0]])

    shape_ref = default_test_shape()
    shape = copy.deepcopy(shape_ref)

    # apply transformation
    shape.apply_transformation(rotation_matrix)

    for i in range(shape.num_points()):
        check_point_rotation(shape._points[i], shape_ref._points[i])

    arc_segment = shape._segments[0]
    arc_segment_ref = shape_ref._segments[0]
    check_point_rotation(arc_segment._point_center,
                         arc_segment_ref._point_center)
    assert arc_segment._sign_arc_winding == arc_segment_ref._sign_arc_winding

    # with reflection
    def check_point_reflection(point, point_ref):
        assert point[0] == point_ref[1]
        assert point[1] == point_ref[0]

    reflection_matrix = np.array([[0, 1], [1, 0]])

    shape = copy.deepcopy(shape_ref)

    # apply transformation
    shape.apply_transformation(reflection_matrix)

    for i in range(shape.num_points()):
        check_point_reflection(shape._points[i], shape_ref._points[i])

    arc_segment = shape._segments[0]
    arc_segment_ref = shape_ref._segments[0]
    check_point_reflection(arc_segment._point_center,
                           arc_segment_ref._point_center)
    assert arc_segment._sign_arc_winding == -arc_segment_ref._sign_arc_winding


def check_reflected_point(point, reflected_point, axis_offset,
                          direction_reflection_axis):
    """Check if the midpoint lies on the reflection axis."""
    vec_original_reflected = reflected_point - point
    mid_point = point + 0.5 * vec_original_reflected
    shifted_mid_point = mid_point - axis_offset
    determinant = np.linalg.det([shifted_mid_point, direction_reflection_axis])
    assert np.abs(determinant) < 1E-8


def shape2d_reflect_testcase(normal, distance_to_origin):
    direction_reflection_axis = np.array([normal[1], -normal[0]])
    normal_length = np.linalg.norm(normal)
    unit_normal = np.array(normal) / normal_length
    offset = distance_to_origin * unit_normal

    shape = default_test_shape()

    # create reflected shape
    shape_reflected = copy.deepcopy(shape)
    shape_reflected.reflect(normal, distance_to_origin)

    # check reflected points
    check_reflected_point(shape._segments[0]._point_center,
                          shape_reflected._segments[0]._point_center, offset,
                          direction_reflection_axis)

    for i in range(shape.num_points()):
        check_reflected_point(shape._points[i], shape_reflected._points[i],
                              offset, direction_reflection_axis)


def test_shape2d_reflect():
    shape2d_reflect_testcase([2, 1], np.linalg.norm([2, 1]))
    shape2d_reflect_testcase([0, 1], 5)
    shape2d_reflect_testcase([1, 0], 3)
    shape2d_reflect_testcase([1, 0], -3)
    shape2d_reflect_testcase([-7, 2], 4.12)
    shape2d_reflect_testcase([-7, -2], 4.12)
    shape2d_reflect_testcase([7, -2], 4.12)
