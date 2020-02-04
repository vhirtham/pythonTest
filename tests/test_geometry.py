import mypackage._utility as utils
import mypackage.geometry as geo
import tests._helpers as helpers

import pytest
import numpy as np
import math
import copy


# helpers ---------------------------------------------------------------------

def check_segments_identical(a, b):
    assert isinstance(a, type(b))
    helpers.check_vectors_identical(a.point_start, b.point_start)
    helpers.check_vectors_identical(a.point_end, b.point_end)
    if isinstance(a, geo.ArcSegment):
        assert a.arc_winding_ccw == b.arc_winding_ccw
        helpers.check_vectors_identical(a.point_center, b.point_center)


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
    helpers.check_vectors_identical(data[:, 0], point_start)
    helpers.check_vectors_identical(data[:, -1], point_end)

    for i in range(num_points - 1):
        point = data[:, i]
        next_point = data[:, i + 1]

        raster_width_eff = np.linalg.norm(next_point - point)
        assert np.abs(raster_width_eff - raster_width) < 0.1 * raster_width

    # check that there are no duplicate points
    assert helpers.are_all_points_unique(data)

    # check rasterization with excluded points
    data_m2 = segment.rasterize(raster_width, 2)

    num_points_m2 = data_m2.shape[1]

    assert num_points - 2 == num_points_m2

    for i in range(num_points_m2):
        helpers.check_vectors_identical(data[:, i], data_m2[:, i])

    # check that rasterization with to large raster width still works
    data_200 = segment.rasterize(200)

    num_points_200 = data_200.shape[1]
    assert num_points_200 == 2
    helpers.check_vectors_identical(point_start, data_200[:, 0])
    helpers.check_vectors_identical(point_end, data_200[:, 1])

    # check exceptions when raster width <= 0
    with pytest.raises(ValueError):
        segment.rasterize(0)
    with pytest.raises(ValueError):
        segment.rasterize(-3)


# test LineSegment ------------------------------------------------------------

def test_line_segment_construction():
    # class constructor -----------------------------------
    segment = geo.LineSegment([[3, 5], [3, 4]])
    assert math.isclose(segment.length, np.sqrt(5))

    # exceptions ------------------------------------------
    # length = 0
    with pytest.raises(ValueError):
        geo.LineSegment([[0, 0], [1, 1]])
    # not 2x2
    with pytest.raises(ValueError):
        geo.LineSegment([[3, 5], [3, 4], [3, 2]])
    # not a 2d array
    with pytest.raises(ValueError):
        geo.LineSegment([[[3, 5], [3, 4]]])

    # factories -------------------------------------------
    segment = geo.LineSegment.construct_with_points([3, 3], [4, 5])
    assert math.isclose(segment.length, np.sqrt(5))


def test_line_segment_rasterization():
    raster_width = 0.1

    point_start = np.array([3, 3])
    point_end = np.array([4, 5])
    points = np.array([point_start, point_end]).transpose()
    vec_start_end = point_end - point_start
    unit_vec_start_end = vec_start_end / np.linalg.norm(vec_start_end)

    segment = geo.LineSegment(points)

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


def test_line_segment_transformations():
    # translation -----------------------------------------
    segment = geo.LineSegment.construct_with_points([3, 3], [4, 5])
    segment.translate([-1, 4])

    helpers.check_vectors_identical(segment.point_start, np.array([2, 7]))
    helpers.check_vectors_identical(segment.point_end, np.array([3, 9]))
    assert math.isclose(segment.length, np.sqrt(5))

    # 45 degree rotation ----------------------------------
    s = np.sin(np.pi / 4.)
    c = np.cos(np.pi / 4.)
    rotation_matrix = [[c, -s], [s, c]]

    segment = geo.LineSegment.construct_with_points([2, 2], [3, 6])
    segment.apply_transformation(rotation_matrix)

    exp_start = [0, np.sqrt(8)]
    exp_end = np.matmul(rotation_matrix, [3, 6])

    helpers.check_vectors_identical(segment.point_start, exp_start)
    helpers.check_vectors_identical(segment.point_end, exp_end)
    assert math.isclose(segment.length, np.sqrt(17))

    # reflection at 45 degree line ------------------------
    v = np.array([-1, 1], dtype=float)
    reflection_matrix = np.identity(2) - 2 / np.dot(v, v) * np.outer(v, v)

    segment = geo.LineSegment.construct_with_points([-1, 3], [6, 1])
    segment.apply_transformation(reflection_matrix)

    helpers.check_vectors_identical(segment.point_start, [3, -1])
    helpers.check_vectors_identical(segment.point_end, [1, 6])
    assert math.isclose(segment.length, np.sqrt(53))

    # scaling ---------------------------------------------
    scale_matrix = [[4, 0], [0, 0.5]]

    segment = geo.LineSegment.construct_with_points([-2, 2], [1, 4])
    segment.apply_transformation(scale_matrix)

    helpers.check_vectors_identical(segment.point_start, [-8, 1])
    helpers.check_vectors_identical(segment.point_end, [4, 2])
    # length changes due to scaling!
    assert math.isclose(segment.length, np.sqrt(145))

    # exceptions ------------------------------------------

    # transformation results in length = 0
    zero_matrix = np.zeros((2, 2))
    with pytest.raises(Exception):
        segment.apply_transformation(zero_matrix)


def test_line_segment_interpolation():
    segment_a = geo.LineSegment.construct_with_points([1, 3], [7, -3])
    segment_b = geo.LineSegment.construct_with_points([5, -5], [-1, 13])

    for i in range(5):
        weight = i / 4
        segment_c = geo.LineSegment.linear_interpolation(segment_a,
                                                         segment_b,
                                                         weight)
        assert math.isclose(segment_c.points[0, 0], 1 + i)
        assert math.isclose(segment_c.points[1, 0], 3 - 2 * i)
        assert math.isclose(segment_c.points[0, 1], 7 - 2 * i)
        assert math.isclose(segment_c.points[1, 1], -3 + 4 * i)

    # check weight clipped to valid range -----------------

    segment_c = geo.LineSegment.linear_interpolation(segment_a, segment_b, -3)
    helpers.check_vectors_identical(segment_c.point_start,
                                    segment_a.point_start)
    helpers.check_vectors_identical(segment_c.point_end, segment_a.point_end)

    segment_c = geo.LineSegment.linear_interpolation(segment_a, segment_b, 6)
    helpers.check_vectors_identical(segment_c.point_start,
                                    segment_b.point_start)
    helpers.check_vectors_identical(segment_c.point_end, segment_b.point_end)

    # exceptions ------------------------------------------

    # wrong types
    arc_segment = geo.ArcSegment.construct_with_points([0, 0], [1, 1], [1, 0])
    with pytest.raises(TypeError):
        geo.LineSegment.linear_interpolation(segment_a, arc_segment, weight)
    with pytest.raises(TypeError):
        geo.LineSegment.linear_interpolation(arc_segment, segment_a, weight)
    with pytest.raises(TypeError):
        geo.LineSegment.linear_interpolation(arc_segment, arc_segment, weight)


# test ArcSegment ------------------------------------------------------------

def check_arc_segment_values(segment, point_start, point_end, point_center,
                             winding_ccw, radius, arc_angle, arc_length):
    helpers.check_vectors_identical(segment.point_start, point_start)
    helpers.check_vectors_identical(segment.point_end, point_end)
    helpers.check_vectors_identical(segment.point_center, point_center)

    assert segment.arc_winding_ccw is winding_ccw
    assert math.isclose(segment.radius, radius)
    assert math.isclose(segment.arc_angle, arc_angle)
    assert math.isclose(segment.arc_length, arc_length)


def arc_segment_test(point_center, point_start, point_end, raster_width,
                     arc_winding_ccw, check_winding):
    point_center = np.array(point_center)
    point_start = np.array(point_start)
    point_end = np.array(point_end)

    radius_arc = np.linalg.norm(point_start - point_center)

    arc_segment = geo.ArcSegment.construct_with_points(point_start,
                                                       point_end,
                                                       point_center,
                                                       arc_winding_ccw)

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
    points = [[3, 6, 6], [3, 6, 3]]
    segment_cw = geo.ArcSegment(points, False)
    segment_ccw = geo.ArcSegment(points, True)

    assert not segment_cw.arc_winding_ccw
    assert segment_ccw.arc_winding_ccw

    assert math.isclose(segment_cw.radius, 3)
    assert math.isclose(segment_ccw.radius, 3)

    assert math.isclose(segment_cw.arc_angle, 1 / 2 * np.pi)
    assert math.isclose(segment_ccw.arc_angle, 3 / 2 * np.pi)

    assert math.isclose(segment_cw.arc_length, 3 / 2 * np.pi)
    assert math.isclose(segment_ccw.arc_length, 9 / 2 * np.pi)

    # check exceptions ------------------------------------

    # radius differs
    points = [[3, 6, 6], [3, 10, 3]]
    with pytest.raises(Exception):
        geo.ArcSegment(points, False)

    # radius is zero
    points = [[3, 3, 3], [3, 3, 3]]
    with pytest.raises(Exception):
        geo.ArcSegment(points, False)

    # arc length zero
    points = [[3, 3, 6], [3, 3, 3]]
    with pytest.raises(Exception):
        geo.ArcSegment(points, False)
    with pytest.raises(Exception):
        geo.ArcSegment(points, True)

    # not 2x3
    points = [[3, 3], [3, 3]]
    with pytest.raises(ValueError):
        geo.ArcSegment(points)

    # not a 2d array
    points = [[[3, 3, 6], [3, 3, 3]]]
    with pytest.raises(ValueError):
        geo.ArcSegment([[[3, 5], [3, 4]]])


def test_arc_segment_factories():
    # construction with center point ----------------------
    point_start = [3, 3]
    point_end = [6, 6]
    point_center_left = [3, 6]
    point_center_right = [6, 3]

    # expected results
    radius = 3
    angle_small = np.pi * 0.5
    angle_large = np.pi * 1.5
    arc_length_small = np.pi * 1.5
    arc_length_large = np.pi * 4.5

    segment_cw = geo.ArcSegment.construct_with_points(point_start, point_end,
                                                      point_center_right,
                                                      False)
    segment_ccw = geo.ArcSegment.construct_with_points(point_start, point_end,
                                                       point_center_right,
                                                       True)

    check_arc_segment_values(segment_cw, point_start, point_end,
                             point_center_right, False, radius, angle_small,
                             arc_length_small)
    check_arc_segment_values(segment_ccw, point_start, point_end,
                             point_center_right, True, radius, angle_large,
                             arc_length_large)

    # construction with radius ----------------------

    # center left of line
    segment_cw = geo.ArcSegment.construct_with_radius(point_start, point_end,
                                                      radius, True, False)
    segment_ccw = geo.ArcSegment.construct_with_radius(point_start, point_end,
                                                       radius, True, True)

    check_arc_segment_values(segment_cw, point_start, point_end,
                             point_center_left, False, radius, angle_large,
                             arc_length_large)
    check_arc_segment_values(segment_ccw, point_start, point_end,
                             point_center_left, True, radius, angle_small,
                             arc_length_small)

    # center right of line
    segment_cw = geo.ArcSegment.construct_with_radius(point_start, point_end,
                                                      radius, False, False)
    segment_ccw = geo.ArcSegment.construct_with_radius(point_start, point_end,
                                                       radius, False, True)

    check_arc_segment_values(segment_cw, point_start, point_end,
                             point_center_right, False, radius, angle_small,
                             arc_length_small)
    check_arc_segment_values(segment_ccw, point_start, point_end,
                             point_center_right, True, radius, angle_large,
                             arc_length_large)

    # check that too small radii will be clipped to minimal radius
    segment_cw = geo.ArcSegment.construct_with_radius(point_start, point_end,
                                                      0.1, False, False)
    segment_ccw = geo.ArcSegment.construct_with_radius(point_start, point_end,
                                                       0.1, False, True)

    check_arc_segment_values(segment_cw, point_start, point_end, [4.5, 4.5],
                             False, np.sqrt(18) / 2, np.pi,
                             np.pi * np.sqrt(18) / 2)
    check_arc_segment_values(segment_ccw, point_start, point_end, [4.5, 4.5],
                             True, np.sqrt(18) / 2, np.pi,
                             np.pi * np.sqrt(18) / 2)


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


def test_arc_segment_transformations():
    # translation -----------------------------------------
    segment_cw = geo.ArcSegment.construct_with_points([3, 3], [5, 5], [5, 3],
                                                      False)
    segment_ccw = geo.ArcSegment.construct_with_points([3, 3], [5, 5], [5, 3],
                                                       True)
    segment_cw.translate([-1, 4])
    segment_ccw.translate([-1, 4])

    exp_start = [2, 7]
    exp_end = [4, 9]
    exp_center = [4, 7]
    exp_radius = 2
    exp_angle_cw = 0.5 * np.pi
    exp_angle_ccw = 1.5 * np.pi
    exp_arc_length_cw = np.pi
    exp_arc_length_ccw = 3 * np.pi

    check_arc_segment_values(segment_cw, exp_start, exp_end, exp_center, False,
                             exp_radius, exp_angle_cw, exp_arc_length_cw)
    check_arc_segment_values(segment_ccw, exp_start, exp_end, exp_center, True,
                             exp_radius, exp_angle_ccw, exp_arc_length_ccw)

    # 45 degree rotation ----------------------------------
    s = np.sin(np.pi / 4.)
    c = np.cos(np.pi / 4.)
    rotation_matrix = [[c, -s], [s, c]]

    segment_cw = geo.ArcSegment.construct_with_points([3, 3], [5, 5], [5, 3],
                                                      False)
    segment_ccw = geo.ArcSegment.construct_with_points([3, 3], [5, 5], [5, 3],
                                                       True)
    segment_cw.apply_transformation(rotation_matrix)
    segment_ccw.apply_transformation(rotation_matrix)

    exp_start = [0, np.sqrt(18)]
    exp_end = [0, np.sqrt(50)]
    exp_center = np.matmul(rotation_matrix, [5, 3])

    check_arc_segment_values(segment_cw, exp_start, exp_end, exp_center, False,
                             exp_radius, exp_angle_cw, exp_arc_length_cw)
    check_arc_segment_values(segment_ccw, exp_start, exp_end, exp_center, True,
                             exp_radius, exp_angle_ccw, exp_arc_length_ccw)

    # reflection at 45 degree line ------------------------
    v = np.array([-1, 1], dtype=float)
    reflection_matrix = np.identity(2) - 2 / np.dot(v, v) * np.outer(v, v)

    segment_cw = geo.ArcSegment.construct_with_points([3, 2], [5, 4], [5, 2],
                                                      False)
    segment_ccw = geo.ArcSegment.construct_with_points([3, 2], [5, 4], [5, 2],
                                                       True)
    segment_cw.apply_transformation(reflection_matrix)
    segment_ccw.apply_transformation(reflection_matrix)

    exp_start = [2, 3]
    exp_end = [4, 5]
    exp_center = [2, 5]

    # Reflection must change winding!
    check_arc_segment_values(segment_cw, exp_start, exp_end, exp_center, True,
                             exp_radius, exp_angle_cw, exp_arc_length_cw)
    check_arc_segment_values(segment_ccw, exp_start, exp_end, exp_center,
                             False, exp_radius, exp_angle_ccw,
                             exp_arc_length_ccw)

    # scaling both coordinates equally --------------------
    scaling_matrix = [[4, 0], [0, 4]]

    segment_cw = geo.ArcSegment.construct_with_points([3, 2], [5, 4], [5, 2],
                                                      False)
    segment_ccw = geo.ArcSegment.construct_with_points([3, 2], [5, 4], [5, 2],
                                                       True)
    segment_cw.apply_transformation(scaling_matrix)
    segment_ccw.apply_transformation(scaling_matrix)

    exp_start = [12, 8]
    exp_end = [20, 16]
    exp_center = [20, 8]

    # arc_length and radius changed due to scaling!
    exp_radius = 8
    exp_arc_length_cw = np.pi * 4
    exp_arc_length_ccw = np.pi * 12

    check_arc_segment_values(segment_cw, exp_start, exp_end, exp_center, False,
                             exp_radius, exp_angle_cw, exp_arc_length_cw)
    check_arc_segment_values(segment_ccw, exp_start, exp_end, exp_center, True,
                             exp_radius, exp_angle_ccw, exp_arc_length_ccw)

    # non-uniform scaling which results in a valid arc ----
    scaling_matrix = [[0.25, 0], [0, 2]]

    segment_cw = geo.ArcSegment.construct_with_points([8, 4], [32, 4], [20, 2],
                                                      False)
    segment_ccw = geo.ArcSegment.construct_with_points([8, 4], [32, 4],
                                                       [20, 2], True)
    segment_cw.apply_transformation(scaling_matrix)
    segment_ccw.apply_transformation(scaling_matrix)

    exp_start = [2, 8]
    exp_end = [8, 8]
    exp_center = [5, 4]

    # angle, arc length and radius changed due to scaling!
    exp_radius = 5
    exp_angle_cw = 2 * np.arcsin(3 / 5)
    exp_angle_ccw = 2 * np.pi - 2 * np.arcsin(3 / 5)
    exp_arc_length_cw = exp_angle_cw * exp_radius
    exp_arc_length_ccw = exp_angle_ccw * exp_radius

    check_arc_segment_values(segment_cw, exp_start, exp_end, exp_center, False,
                             exp_radius, exp_angle_cw, exp_arc_length_cw)
    check_arc_segment_values(segment_ccw, exp_start, exp_end, exp_center, True,
                             exp_radius, exp_angle_ccw, exp_arc_length_ccw)

    # exceptions ------------------------------------------

    # transformation distorts arc
    segment = geo.ArcSegment.construct_with_points([3, 2], [5, 4], [5, 2],
                                                   False)
    with pytest.raises(Exception):
        segment.apply_transformation(scaling_matrix)

    # transformation results in length = 0
    segment = geo.ArcSegment.construct_with_points([3, 2], [5, 4], [5, 2],
                                                   False)
    zero_matrix = np.zeros((2, 2))
    with pytest.raises(Exception):
        segment.apply_transformation(zero_matrix)


def test_arc_segment_interpolation():
    segment_a = geo.ArcSegment.construct_with_points([0, 0], [1, 1], [1, 0])
    segment_b = geo.ArcSegment.construct_with_points([0, 0], [2, 2], [0, 2])

    # not implemented yet
    with pytest.raises(Exception):
        geo.ArcSegment.linear_interpolation(segment_a, segment_b, 1)


# test Shape ------------------------------------------------------------------

def test_shape_construction():
    line_segment = geo.LineSegment.construct_with_points([1, 1], [1, 2])
    arc_segment = geo.ArcSegment.construct_with_points([0, 0], [1, 1], [0, 1])

    # Empty construction
    shape = geo.Shape()
    assert shape.num_segments == 0

    # Single element construction shape
    shape = geo.Shape(line_segment)
    assert shape.num_segments == 1

    # Multi segment construction
    shape = geo.Shape([arc_segment, line_segment])
    assert shape.num_segments == 2
    assert isinstance(shape.segments[0], geo.ArcSegment)
    assert isinstance(shape.segments[1], geo.LineSegment)

    # exceptions ------------------------------------------

    # segments not connected
    with pytest.raises(Exception):
        shape = geo.Shape([line_segment, arc_segment])


def test_shape_segment_addition():
    # Create shape and add segments
    line_segment = geo.LineSegment.construct_with_points([1, 1], [0, 0])
    arc_segment = geo.ArcSegment.construct_with_points([0, 0], [1, 1], [0, 1])
    arc_segment2 = geo.ArcSegment.construct_with_points([1, 1], [0, 0], [0, 1])

    shape = geo.Shape()
    shape.add_segments(line_segment)
    assert shape.num_segments == 1

    shape.add_segments([arc_segment, arc_segment2])
    assert shape.num_segments == 3
    assert isinstance(shape.segments[0], geo.LineSegment)
    assert isinstance(shape.segments[1], geo.ArcSegment)
    assert isinstance(shape.segments[2], geo.ArcSegment)

    # exceptions ------------------------------------------

    # new segment are not connected to already included segments
    with pytest.raises(Exception):
        shape.add_segments(arc_segment2)
    assert shape.num_segments == 3  # ensure shape is unmodified

    with pytest.raises(Exception):
        shape.add_segments([arc_segment2, arc_segment])
    assert shape.num_segments == 3  # ensure shape is unmodified

    with pytest.raises(Exception):
        shape.add_segments([arc_segment, arc_segment])
    assert shape.num_segments == 3  # ensure shape is unmodified


def test_shape_rasterization():
    points = np.array([[0, 0],
                       [0, 1],
                       [1, 1],
                       [1, 0]])

    raster_width = 0.2

    shape = geo.Shape(
        geo.LineSegment.construct_with_points(points[0], points[1]))
    shape.add_segments(
        geo.LineSegment.construct_with_points(points[1], points[2]))
    shape.add_segments(
        geo.LineSegment.construct_with_points(points[2], points[3]))

    data = shape.rasterize(raster_width)

    # no duplications
    assert helpers.are_all_points_unique(data)

    # check each data point
    num_data_points = data.shape[1]
    for i in range(num_data_points):
        if i < 6:
            helpers.check_vectors_identical([0, i * 0.2], data[:, i])
        elif i < 11:
            helpers.check_vectors_identical([(i - 5) * 0.2, 1], data[:, i])
        else:
            helpers.check_vectors_identical([1, 1 - (i - 10) * 0.2],
                                            data[:, i])

    # Test with too large raster width --------------------
    # The shape does not clip large values to the valid range itself. The
    # added segments do the clipping. If a custom segment does not do that,
    # there is currently no mechanism to correct it.
    # However, this test somewhat ensures, that each segment is rasterized
    # individually.

    data = shape.rasterize(10)

    for point in points:
        assert utils.is_column_in_matrix(point, data)

    # exceptions ------------------------------------------
    with pytest.raises(Exception):
        shape.rasterize(0)
    with pytest.raises(Exception):
        shape.rasterize(-3)


def default_test_shape():
    # create shape
    arc_segment = geo.ArcSegment.construct_with_points([3, 4], [5, 0], [6, 3])
    line_segment = geo.LineSegment.construct_with_points([5, 0], [11, 3])
    return geo.Shape([arc_segment, line_segment])


def test_shape_translation():
    def check_point(point, point_ref, translation):
        helpers.check_vectors_identical(point - translation, point_ref)

    translation = [3, 4]

    shape_ref = default_test_shape()
    shape = copy.deepcopy(shape_ref)

    # apply translation
    shape.translate(translation)

    arc_segment = shape.segments[0]
    arc_segment_ref = shape_ref.segments[0]

    assert (arc_segment.arc_winding_ccw == arc_segment_ref.arc_winding_ccw)

    check_point(arc_segment.point_start, arc_segment_ref.point_start,
                translation)
    check_point(arc_segment.point_end, arc_segment_ref.point_end,
                translation)
    check_point(arc_segment.point_center, arc_segment_ref.point_center,
                translation)

    line_segment = shape.segments[1]
    line_segment_ref = shape_ref.segments[1]

    check_point(line_segment.point_start, line_segment_ref.point_start,
                translation)
    check_point(line_segment.point_end, line_segment_ref.point_end,
                translation)


def test_shape_transformation():
    # without reflection
    def check_point_rotation(point, point_ref):
        assert point[0] == point_ref[1]
        assert point[1] == -point_ref[0]

    rotation_matrix = np.array([[0, 1], [-1, 0]])

    shape_ref = default_test_shape()
    shape = copy.deepcopy(shape_ref)

    # apply transformation
    shape.apply_transformation(rotation_matrix)

    arc_segment = shape.segments[0]
    arc_segment_ref = shape_ref.segments[0]

    assert (arc_segment.arc_winding_ccw == arc_segment_ref.arc_winding_ccw)

    check_point_rotation(arc_segment.point_start, arc_segment_ref.point_start)
    check_point_rotation(arc_segment.point_end, arc_segment_ref.point_end)
    check_point_rotation(arc_segment.point_center,
                         arc_segment_ref.point_center)

    line_segment = shape.segments[1]
    line_segment_ref = shape_ref.segments[1]

    check_point_rotation(line_segment.point_start,
                         line_segment_ref.point_start)
    check_point_rotation(line_segment.point_end, line_segment_ref.point_end)

    # with reflection
    def check_point_reflection(point, point_ref):
        assert point[0] == point_ref[1]
        assert point[1] == point_ref[0]

    reflection_matrix = np.array([[0, 1], [1, 0]])

    shape = copy.deepcopy(shape_ref)

    # apply transformation
    shape.apply_transformation(reflection_matrix)

    arc_segment = shape.segments[0]
    arc_segment_ref = shape_ref.segments[0]

    assert (arc_segment.arc_winding_ccw != arc_segment_ref.arc_winding_ccw)

    check_point_reflection(arc_segment.point_start,
                           arc_segment_ref.point_start)
    check_point_reflection(arc_segment.point_end, arc_segment_ref.point_end)
    check_point_reflection(arc_segment.point_center,
                           arc_segment_ref.point_center)

    line_segment = shape.segments[1]
    line_segment_ref = shape_ref.segments[1]

    check_point_reflection(line_segment.point_start,
                           line_segment_ref.point_start)
    check_point_reflection(line_segment.point_end, line_segment_ref.point_end)


def check_reflected_point(point, reflected_point, axis_offset,
                          direction_reflection_axis):
    """Check if the midpoint lies on the reflection axis."""
    vec_original_reflected = reflected_point - point
    mid_point = point + 0.5 * vec_original_reflected
    shifted_mid_point = mid_point - axis_offset
    determinant = np.linalg.det(
        [shifted_mid_point, direction_reflection_axis])
    assert np.abs(determinant) < 1E-8


def shape_reflect_testcase(normal, distance_to_origin):
    direction_reflection_axis = np.array([normal[1], -normal[0]])
    normal_length = np.linalg.norm(normal)
    unit_normal = np.array(normal) / normal_length
    offset = distance_to_origin * unit_normal

    shape = default_test_shape()

    # create reflected shape
    shape_reflected = copy.deepcopy(shape)
    shape_reflected.reflect(normal, distance_to_origin)

    arc_segment = shape.segments[0]
    arc_segment_ref = shape_reflected.segments[0]
    line_segment = shape.segments[1]
    line_segment_ref = shape_reflected.segments[1]

    # check reflected points
    check_reflected_point(arc_segment.point_start,
                          arc_segment_ref.point_start,
                          offset,
                          direction_reflection_axis)
    check_reflected_point(arc_segment.point_end,
                          arc_segment_ref.point_end,
                          offset,
                          direction_reflection_axis)
    check_reflected_point(arc_segment.point_center,
                          arc_segment_ref.point_center,
                          offset,
                          direction_reflection_axis)

    check_reflected_point(line_segment.point_start,
                          line_segment_ref.point_start,
                          offset,
                          direction_reflection_axis)
    check_reflected_point(line_segment.point_end,
                          line_segment_ref.point_end,
                          offset,
                          direction_reflection_axis)


def test_shape_reflect():
    shape_reflect_testcase([2, 1], np.linalg.norm([2, 1]))
    shape_reflect_testcase([0, 1], 5)
    shape_reflect_testcase([1, 0], 3)
    shape_reflect_testcase([1, 0], -3)
    shape_reflect_testcase([-7, 2], 4.12)
    shape_reflect_testcase([-7, -2], 4.12)
    shape_reflect_testcase([7, -2], 4.12)


def interpolation_nearest(segment_a, segment_b, weight):
    if weight > 0.5:
        return segment_b
    return segment_a


def test_shape_interpolation_general():
    segment_a0 = geo.LineSegment.construct_with_points([-1, -1], [1, 1])
    segment_a1 = geo.LineSegment.construct_with_points([1, 1], [3, -1])
    shape_a = geo.Shape([segment_a0, segment_a1])

    segment_b0 = geo.LineSegment.construct_with_points([-1, 4], [1, 1])
    segment_b1 = geo.LineSegment.construct_with_points([1, 1], [3, 4])
    shape_b = geo.Shape([segment_b0, segment_b1])

    interpolations = [geo.LineSegment.linear_interpolation,
                      interpolation_nearest]
    for i in range(6):
        weight = i / 5.
        shape_c = geo.Shape.interpolate(shape_a, shape_b, weight,
                                        interpolations)
        assert shape_c.num_segments == 2

        exp_segment_c0 = geo.LineSegment.construct_with_points(
            [-1, -1 + 5 * weight], [1, 1])
        check_segments_identical(shape_c.segments[0], exp_segment_c0)

        if weight > 0.5:
            check_segments_identical(shape_c.segments[1], segment_b1)
        else:
            check_segments_identical(shape_c.segments[1], segment_a1)


def test_shape_linear_interpolation():
    segment_a0 = geo.LineSegment.construct_with_points([0, 0], [1, 1])
    segment_a1 = geo.LineSegment.construct_with_points([1, 1], [2, 0])
    shape_a = geo.Shape([segment_a0, segment_a1])

    segment_b0 = geo.LineSegment.construct_with_points([1, 1], [2, -1])
    segment_b1 = geo.LineSegment.construct_with_points([2, -1], [3, 5])
    shape_b = geo.Shape([segment_b0, segment_b1])

    for i in range(5):
        weight = i / 4.
        shape_c = geo.Shape.linear_interpolation(shape_a, shape_b, weight)

        helpers.check_vectors_identical(shape_c.segments[0].point_start,
                                        [weight, weight])
        helpers.check_vectors_identical(shape_c.segments[0].point_end,
                                        [1 + weight, 1 - 2 * weight])

        helpers.check_vectors_identical(shape_c.segments[1].point_start,
                                        [1 + weight, 1 - 2 * weight])
        helpers.check_vectors_identical(shape_c.segments[1].point_end,
                                        [2 + weight, 5 * weight])

    # check weight clipped to valid range -----------------

    shape_c = geo.Shape.linear_interpolation(shape_a, shape_b, -3)

    helpers.check_vectors_identical(shape_c.segments[0].point_start,
                                    shape_a.segments[0].point_start)
    helpers.check_vectors_identical(shape_c.segments[0].point_end,
                                    shape_a.segments[0].point_end)
    helpers.check_vectors_identical(shape_c.segments[1].point_start,
                                    shape_a.segments[1].point_start)
    helpers.check_vectors_identical(shape_c.segments[1].point_end,
                                    shape_a.segments[1].point_end)

    shape_c = geo.Shape.linear_interpolation(shape_a, shape_b, 100)

    helpers.check_vectors_identical(shape_c.segments[0].point_start,
                                    shape_b.segments[0].point_start)
    helpers.check_vectors_identical(shape_c.segments[0].point_end,
                                    shape_b.segments[0].point_end)
    helpers.check_vectors_identical(shape_c.segments[1].point_start,
                                    shape_b.segments[1].point_start)
    helpers.check_vectors_identical(shape_c.segments[1].point_end,
                                    shape_b.segments[1].point_end)

    # exceptions ------------------------------------------

    shape_a.add_segments(geo.LineSegment.construct_with_points([2, 0], [2, 2]))
    with pytest.raises(Exception):
        geo.Shape.linear_interpolation(shape_a, shape_b, 0.25)
