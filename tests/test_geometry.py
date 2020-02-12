import mypackage.geometry as geo
import mypackage.transformations as tf
import mypackage._utility as utils

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


def check_profiles_identical(a, b):
    assert a.num_shapes == b.num_shapes
    for i in range(a.num_shapes):
        check_shapes_identical(a.shapes[i], b.shapes[i])


def check_shapes_identical(a, b):
    assert a.num_segments == b.num_segments
    for i in range(a.num_segments):
        check_segments_identical(a.segments[i], b.segments[i])


def check_trace_segments_identical(a, b):
    assert isinstance(a, type(b))
    if isinstance(a, geo.LinearHorizontalTraceSegment):
        assert a.length == b.length
    else:
        assert a.is_clockwise == b.is_clockwise
        assert math.isclose(a.angle, b.angle)
        assert math.isclose(a.length, b.length)
        assert math.isclose(a.radius, b.radius)


def check_traces_identical(a, b):
    assert a.num_segments == b.num_segments
    for i in range(a.num_segments):
        check_trace_segments_identical(a.segments[i], b.segments[i])


def get_default_profiles():
    a_0 = [0, 0]
    a_1 = [8, 16]
    a_2 = [16, 0]
    shape_a01 = geo.Shape(geo.LineSegment.construct_with_points(a_0, a_1))
    shape_a12 = geo.Shape(geo.LineSegment.construct_with_points(a_1, a_2))
    profile_a = geo.Profile([shape_a01, shape_a12])

    b_0 = [-4, 8]
    b_1 = [0, 8]
    b_2 = [16, -16]
    shape_b01 = geo.Shape(geo.LineSegment.construct_with_points(b_0, b_1))
    shape_b12 = geo.Shape(geo.LineSegment.construct_with_points(b_1, b_2))
    profile_b = geo.Profile([shape_b01, shape_b12])
    return [profile_a, profile_b]


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
    segment_2 = segment.translate([-1, 4])

    # original segment not modified
    helpers.check_vectors_identical(segment.point_start, np.array([3, 3]))
    helpers.check_vectors_identical(segment.point_end, np.array([4, 5]))

    # check new segment
    helpers.check_vectors_identical(segment_2.point_start, np.array([2, 7]))
    helpers.check_vectors_identical(segment_2.point_end, np.array([3, 9]))
    assert math.isclose(segment_2.length, np.sqrt(5))

    # apply same transformation in place
    segment.apply_translation([-1, 4])
    check_segments_identical(segment, segment_2)

    # 45 degree rotation ----------------------------------
    s = np.sin(np.pi / 4.)
    c = np.cos(np.pi / 4.)
    rotation_matrix = [[c, -s], [s, c]]

    segment = geo.LineSegment.construct_with_points([2, 2], [3, 6])
    segment_2 = segment.transform(rotation_matrix)

    # original segment not modified
    helpers.check_vectors_identical(segment.point_start, np.array([2, 2]))
    helpers.check_vectors_identical(segment.point_end, np.array([3, 6]))

    # check new segment
    exp_start = [0, np.sqrt(8)]
    exp_end = np.matmul(rotation_matrix, [3, 6])

    helpers.check_vectors_identical(segment_2.point_start, exp_start)
    helpers.check_vectors_identical(segment_2.point_end, exp_end)
    assert math.isclose(segment_2.length, np.sqrt(17))

    # apply same transformation in place
    segment.apply_transformation(rotation_matrix)
    check_segments_identical(segment, segment_2)

    # reflection at 45 degree line ------------------------
    v = np.array([-1, 1], dtype=float)
    reflection_matrix = np.identity(2) - 2 / np.dot(v, v) * np.outer(v, v)

    segment = geo.LineSegment.construct_with_points([-1, 3], [6, 1])
    segment_2 = segment.transform(reflection_matrix)

    # original segment not modified
    helpers.check_vectors_identical(segment.point_start, np.array([-1, 3]))
    helpers.check_vectors_identical(segment.point_end, np.array([6, 1]))

    # check new segment
    helpers.check_vectors_identical(segment_2.point_start, [3, -1])
    helpers.check_vectors_identical(segment_2.point_end, [1, 6])
    assert math.isclose(segment_2.length, np.sqrt(53))

    # apply same transformation in place
    segment.apply_transformation(reflection_matrix)
    check_segments_identical(segment, segment_2)

    # scaling ---------------------------------------------
    scale_matrix = [[4, 0], [0, 0.5]]

    segment = geo.LineSegment.construct_with_points([-2, 2], [1, 4])
    segment_2 = segment.transform(scale_matrix)

    # original segment not modified
    helpers.check_vectors_identical(segment.point_start, np.array([-2, 2]))
    helpers.check_vectors_identical(segment.point_end, np.array([1, 4]))

    # check new segment
    helpers.check_vectors_identical(segment_2.point_start, [-8, 1])
    helpers.check_vectors_identical(segment_2.point_end, [4, 2])
    # length changes due to scaling!
    assert math.isclose(segment_2.length, np.sqrt(145))

    # apply same transformation in place
    segment.apply_transformation(scale_matrix)
    check_segments_identical(segment, segment_2)

    # exceptions ------------------------------------------

    # transformation results in length = 0
    zero_matrix = np.zeros((2, 2))
    with pytest.raises(Exception):
        segment.apply_transformation(zero_matrix)
    with pytest.raises(Exception):
        segment.transform(zero_matrix)


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

    helpers.check_vectors_identical([3, 3], segment_cw.points[:, 0])
    helpers.check_vectors_identical([3, 3], segment_ccw.points[:, 0])
    helpers.check_vectors_identical([6, 6], segment_cw.points[:, 1])
    helpers.check_vectors_identical([6, 6], segment_ccw.points[:, 1])
    helpers.check_vectors_identical([6, 3], segment_cw.points[:, 2])
    helpers.check_vectors_identical([6, 3], segment_ccw.points[:, 2])

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

    segment_cw_2 = segment_cw.translate([-1, 4])
    segment_ccw_2 = segment_ccw.translate([-1, 4])

    # original segment not modified
    check_arc_segment_values(segment_cw, [3, 3], [5, 5], [5, 3],
                             False, 2, 0.5 * np.pi, np.pi)
    check_arc_segment_values(segment_ccw, [3, 3], [5, 5], [5, 3],
                             True, 2, 1.5 * np.pi, 3 * np.pi)

    # check new segment
    exp_start = [2, 7]
    exp_end = [4, 9]
    exp_center = [4, 7]
    exp_radius = 2
    exp_angle_cw = 0.5 * np.pi
    exp_angle_ccw = 1.5 * np.pi
    exp_arc_length_cw = np.pi
    exp_arc_length_ccw = 3 * np.pi

    check_arc_segment_values(segment_cw_2, exp_start, exp_end, exp_center,
                             False, exp_radius, exp_angle_cw,
                             exp_arc_length_cw)
    check_arc_segment_values(segment_ccw_2, exp_start, exp_end, exp_center,
                             True, exp_radius, exp_angle_ccw,
                             exp_arc_length_ccw)

    # apply same transformation in place
    segment_cw.apply_translation([-1, 4])
    segment_ccw.apply_translation([-1, 4])
    check_segments_identical(segment_cw_2, segment_cw)
    check_segments_identical(segment_ccw_2, segment_ccw)

    # 45 degree rotation ----------------------------------
    s = np.sin(np.pi / 4.)
    c = np.cos(np.pi / 4.)
    rotation_matrix = [[c, -s], [s, c]]

    segment_cw = geo.ArcSegment.construct_with_points([3, 3], [5, 5], [5, 3],
                                                      False)
    segment_ccw = geo.ArcSegment.construct_with_points([3, 3], [5, 5], [5, 3],
                                                       True)

    segment_cw_2 = segment_cw.transform(rotation_matrix)
    segment_ccw_2 = segment_ccw.transform(rotation_matrix)

    # original segment not modified
    check_arc_segment_values(segment_cw, [3, 3], [5, 5], [5, 3],
                             False, 2, 0.5 * np.pi, np.pi)
    check_arc_segment_values(segment_ccw, [3, 3], [5, 5], [5, 3],
                             True, 2, 1.5 * np.pi, 3 * np.pi)

    # check new segment
    exp_start = [0, np.sqrt(18)]
    exp_end = [0, np.sqrt(50)]
    exp_center = np.matmul(rotation_matrix, [5, 3])

    check_arc_segment_values(segment_cw_2, exp_start, exp_end, exp_center,
                             False, exp_radius, exp_angle_cw,
                             exp_arc_length_cw)
    check_arc_segment_values(segment_ccw_2, exp_start, exp_end, exp_center,
                             True, exp_radius, exp_angle_ccw,
                             exp_arc_length_ccw)

    # apply same transformation in place
    segment_cw.apply_transformation(rotation_matrix)
    segment_ccw.apply_transformation(rotation_matrix)
    check_segments_identical(segment_cw_2, segment_cw)
    check_segments_identical(segment_ccw_2, segment_ccw)

    # reflection at 45 degree line ------------------------
    v = np.array([-1, 1], dtype=float)
    reflection_matrix = np.identity(2) - 2 / np.dot(v, v) * np.outer(v, v)

    segment_cw = geo.ArcSegment.construct_with_points([3, 2], [5, 4], [5, 2],
                                                      False)
    segment_ccw = geo.ArcSegment.construct_with_points([3, 2], [5, 4], [5, 2],
                                                       True)

    segment_cw_2 = segment_cw.transform(reflection_matrix)
    segment_ccw_2 = segment_ccw.transform(reflection_matrix)

    # original segment not modified
    check_arc_segment_values(segment_cw, [3, 2], [5, 4], [5, 2],
                             False, 2, 0.5 * np.pi, np.pi)
    check_arc_segment_values(segment_ccw, [3, 2], [5, 4], [5, 2],
                             True, 2, 1.5 * np.pi, 3 * np.pi)

    # check new segment
    exp_start = [2, 3]
    exp_end = [4, 5]
    exp_center = [2, 5]

    # Reflection must change winding!
    check_arc_segment_values(segment_cw_2, exp_start, exp_end, exp_center,
                             True, exp_radius, exp_angle_cw, exp_arc_length_cw)
    check_arc_segment_values(segment_ccw_2, exp_start, exp_end, exp_center,
                             False, exp_radius, exp_angle_ccw,
                             exp_arc_length_ccw)

    # apply same transformation in place
    segment_cw.apply_transformation(reflection_matrix)
    segment_ccw.apply_transformation(reflection_matrix)
    check_segments_identical(segment_cw_2, segment_cw)
    check_segments_identical(segment_ccw_2, segment_ccw)

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
        segment.transform(scaling_matrix)
    with pytest.raises(Exception):
        segment.apply_transformation(scaling_matrix)

    # transformation results in length = 0
    segment = geo.ArcSegment.construct_with_points([3, 2], [5, 4], [5, 2],
                                                   False)
    zero_matrix = np.zeros((2, 2))
    with pytest.raises(Exception):
        segment.transform(zero_matrix)
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


def test_shape_line_segment_addition():
    shape_0 = geo.Shape()
    shape_0.add_line_segments([[0, 0], [1, 0]])
    assert shape_0.num_segments == 1

    shape_1 = geo.Shape()
    shape_1.add_line_segments([[0, 0], [1, 0], [2, 0]])
    assert shape_1.num_segments == 2

    # test possible formats to add single line segment ----

    shape_0.add_line_segments([2, 0])
    assert shape_0.num_segments == 2
    shape_0.add_line_segments([[3, 0]])
    assert shape_0.num_segments == 3
    shape_0.add_line_segments(np.array([4, 0]))
    assert shape_0.num_segments == 4
    shape_0.add_line_segments(np.array([[5, 0]]))
    assert shape_0.num_segments == 5

    # add multiple segments -------------------------------

    shape_0.add_line_segments([[6, 0], [7, 0], [8, 0]])
    assert shape_0.num_segments == 8
    shape_0.add_line_segments(np.array([[9, 0], [10, 0], [11, 0]]))
    assert shape_0.num_segments == 11

    for i in range(11):
        expected_segment = geo.LineSegment.construct_with_points([i, 0],
                                                                 [i + 1, 0])
        check_segments_identical(shape_0.segments[i], expected_segment)
        if i < 2:
            check_segments_identical(shape_1.segments[i], expected_segment)

    # exceptions ------------------------------------------

    shape_2 = geo.Shape()
    # invalid inputs
    with pytest.raises(Exception):
        shape_2.add_line_segments([])
    assert shape_2.num_segments == 0

    with pytest.raises(Exception):
        shape_2.add_line_segments(None)
    assert shape_2.num_segments == 0

    # single point with empty shape
    with pytest.raises(Exception):
        shape_2.add_line_segments([0, 1])
    assert shape_2.num_segments == 0

    # invalid point format
    with pytest.raises(Exception):
        shape_2.add_line_segments([[0, 1, 2], [1, 2, 3]])
    assert shape_2.num_segments == 0


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

    assert data.shape[1] == 4

    # no duplication if shape is closed -------------------

    shape.add_segments(
        geo.LineSegment.construct_with_points(points[3], points[0]))

    data = shape.rasterize(10)

    assert data.shape[1] == 4
    assert helpers.are_all_points_unique(data)

    # exceptions ------------------------------------------
    with pytest.raises(Exception):
        shape.rasterize(0)
    with pytest.raises(Exception):
        shape.rasterize(-3)
    # empty shape
    shape_empty = geo.Shape()
    with pytest.raises(Exception):
        shape_empty.rasterize(0.2)


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

    # apply translation
    shape = shape_ref.translate(translation)

    # original shape unchanged
    check_shapes_identical(shape_ref, default_test_shape())

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

    # apply same transformation in place
    shape_ref.apply_translation(translation)
    check_shapes_identical(shape_ref, shape)


def test_shape_transformation():
    # without reflection ----------------------------------
    def check_point_rotation(point, point_ref):
        assert point[0] == point_ref[1]
        assert point[1] == -point_ref[0]

    rotation_matrix = np.array([[0, 1], [-1, 0]])

    shape_ref = default_test_shape()

    # apply transformation
    shape = shape_ref.transform(rotation_matrix)

    # original shape unchanged
    check_shapes_identical(shape_ref, default_test_shape())

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

    # apply same transformation in place
    shape_ref.apply_transformation(rotation_matrix)
    check_shapes_identical(shape_ref, shape)

    # with reflection -------------------------------------
    def check_point_reflection(point, point_ref):
        assert point[0] == point_ref[1]
        assert point[1] == point_ref[0]

    reflection_matrix = np.array([[0, 1], [1, 0]])

    shape_ref = default_test_shape()

    # apply transformation
    shape = shape_ref.transform(reflection_matrix)

    # original shape unchanged
    check_shapes_identical(shape_ref, default_test_shape())

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

    # apply same transformation in place
    shape_ref.apply_transformation(reflection_matrix)
    check_shapes_identical(shape_ref, shape)


def check_reflected_point(point, reflected_point, axis_offset,
                          direction_reflection_axis):
    """Check if the midpoint lies on the reflection axis."""
    vec_original_reflected = reflected_point - point
    mid_point = point + 0.5 * vec_original_reflected
    shifted_mid_point = mid_point - axis_offset
    determinant = np.linalg.det(
        [shifted_mid_point, direction_reflection_axis])
    assert np.abs(determinant) < 1E-8


def shape_reflection_testcase(normal, distance_to_origin):
    direction_reflection_axis = np.array([normal[1], -normal[0]])
    normal_length = np.linalg.norm(normal)
    unit_normal = np.array(normal) / normal_length
    offset = distance_to_origin * unit_normal

    shape = default_test_shape()

    # create reflected shape
    shape_reflected = shape.reflect(normal, distance_to_origin)

    # original shape is not modified
    check_shapes_identical(shape, default_test_shape())

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

    # apply same reflection in place
    shape.apply_reflection(normal, distance_to_origin)
    check_shapes_identical(shape, shape_reflected)


def test_shape_reflection():
    shape_reflection_testcase([2, 1], np.linalg.norm([2, 1]))
    shape_reflection_testcase([0, 1], 5)
    shape_reflection_testcase([1, 0], 3)
    shape_reflection_testcase([1, 0], -3)
    shape_reflection_testcase([-7, 2], 4.12)
    shape_reflection_testcase([-7, -2], 4.12)
    shape_reflection_testcase([7, -2], 4.12)

    # exceptions ------------------------------------------
    shape = default_test_shape()

    with pytest.raises(Exception):
        shape.reflect([0, 0], 2)
    with pytest.raises(Exception):
        shape.apply_reflection([0, 0])


def check_point_reflected_across_line(point, reflected_point, point_start,
                                      point_end):
    """Check if the midpoint lies on the reflection axis."""
    vec_original_reflected = reflected_point - point
    mid_point = point + 0.5 * vec_original_reflected

    vec_start_mid = mid_point - point_start
    vec_start_end = point_end - point_start

    determinant = np.linalg.det([vec_start_end, vec_start_mid])
    assert np.abs(determinant) < 1E-8


def shape_reflection_across_line_testcase(point_start, point_end):
    point_start = np.array(point_start, float)
    point_end = np.array(point_end, float)

    shape = default_test_shape()

    # create reflected shape
    shape_reflected = shape.reflect_across_line(point_start, point_end)

    # original shape is not modified
    check_shapes_identical(shape, default_test_shape())

    arc_segment = shape.segments[0]
    arc_segment_ref = shape_reflected.segments[0]
    line_segment = shape.segments[1]
    line_segment_ref = shape_reflected.segments[1]

    # check reflected points
    check_point_reflected_across_line(arc_segment.point_start,
                                      arc_segment_ref.point_start,
                                      point_start,
                                      point_end)
    check_point_reflected_across_line(arc_segment.point_end,
                                      arc_segment_ref.point_end,
                                      point_start,
                                      point_end)
    check_point_reflected_across_line(arc_segment.point_center,
                                      arc_segment_ref.point_center,
                                      point_start,
                                      point_end)

    check_point_reflected_across_line(line_segment.point_start,
                                      line_segment_ref.point_start,
                                      point_start,
                                      point_end)
    check_point_reflected_across_line(line_segment.point_end,
                                      line_segment_ref.point_end,
                                      point_start,
                                      point_end)

    # apply same reflection in place
    shape.apply_reflection_across_line(point_start, point_end)
    check_shapes_identical(shape, shape_reflected)


def test_shape_reflection_across_line():
    shape_reflection_across_line_testcase([0, 0], [0, 1])
    shape_reflection_across_line_testcase([0, 0], [1, 0])
    shape_reflection_across_line_testcase([-3, 2.5], [31.53, -23.44])
    shape_reflection_across_line_testcase([7, 8], [9, 10])
    shape_reflection_across_line_testcase([-4.26, -23.1], [-8, -0.12])
    shape_reflection_across_line_testcase([-2, 1], [2, -4.5])

    # exceptions ------------------------------------------
    shape = default_test_shape()

    with pytest.raises(Exception):
        shape.reflect_across_line([2, 5], [2, 5])
    with pytest.raises(Exception):
        shape.apply_reflection_across_line([-3, 2], [-3, 2])


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


# Test profile class ----------------------------------------------------------

def test_profile_construction_and_shape_addition():
    segment0 = geo.LineSegment.construct_with_points([0, 0], [1, 0])
    segment1 = geo.LineSegment.construct_with_points([1, 0], [2, -1])
    segment2 = geo.LineSegment.construct_with_points([2, -1], [0, -1])

    shape = geo.Shape([segment0, segment1, segment2])

    # Check invalid types
    with pytest.raises(TypeError):
        geo.Profile(3)
    with pytest.raises(TypeError):
        geo.Profile("This is not right")
    with pytest.raises(TypeError):
        geo.Profile([2, 8, 1])

    # Check valid types
    profile = geo.Profile(shape)
    assert profile.num_shapes == 1
    profile = geo.Profile([shape, shape])
    assert profile.num_shapes == 2

    # Check invalid addition
    with pytest.raises(TypeError):
        profile.add_shapes([shape, 0.1])
    with pytest.raises(TypeError):
        profile.add_shapes(["shape"])
    with pytest.raises(TypeError):
        profile.add_shapes(0.1)

    # Check that invalid calls only raise an exception and do not invalidate
    # the internal data
    assert profile.num_shapes == 2

    # Check valid addition
    profile.add_shapes(shape)
    assert profile.num_shapes == 3
    profile.add_shapes([shape, shape])
    assert profile.num_shapes == 5

    # Check shapes
    shapes_profile = profile.shapes
    for shape_profile in shapes_profile:
        assert shape.num_segments == shape_profile.num_segments

        segments = shape.segments
        segments_profile = shape_profile.segments

        assert len(segments) == shape.num_segments
        assert len(segments) == len(segments_profile)

        for i in range(shape.num_segments):
            assert isinstance(segments_profile[i], type(segments[i]))
            points = segments[i].points
            points_profile = segments_profile[i].points
            for j in range(2):
                helpers.check_vectors_identical(points[:, j],
                                                points_profile[:, j])


def test_profile_rasterization():
    raster_width = 0.1
    shape0 = geo.Shape(
        geo.LineSegment.construct_with_points([-1, 0], [-raster_width, 0]))
    shape1 = geo.Shape(geo.LineSegment.construct_with_points([0, 0], [1, 0]))
    shape2 = geo.Shape(
        geo.LineSegment.construct_with_points([1 + raster_width, 0], [2, 0]))

    profile = geo.Profile([shape0, shape1])
    profile.add_shapes(shape2)

    # rasterize
    data = profile.rasterize(0.1)

    # no duplications
    assert helpers.are_all_points_unique(data)

    # check raster data size
    expected_number_raster_points = int(round(3 / raster_width)) + 1
    assert data.shape[1] == expected_number_raster_points

    # Check that all shapes are rasterized correct
    for i in range(int(round(3 / raster_width)) + 1):
        expected_raster_point_x = i * raster_width - 1
        assert data[0, i] - expected_raster_point_x < 1E-9
        assert data[1, i] == 0

    # exceptions
    with pytest.raises(Exception):
        profile.rasterize(0)
    with pytest.raises(Exception):
        profile.rasterize(-3)


# Test trace segment classes --------------------------------------------------

def check_trace_segment_length(segment, tolerance=1E-9):
    lcs = segment.local_coordinate_system(1)
    length_numeric_prev = np.linalg.norm(lcs.origin)

    # calculate numerical length by linearization
    num_segments = 2.
    num_iterations = 20

    # calculate numerical length with increasing number of segments until
    # the rate of change between 2 calculations is small enough
    for i in range(num_iterations):
        length_numeric = 0
        increment = 1. / num_segments

        ccs0 = segment.local_coordinate_system(0)
        for rel_pos in (np.arange(increment, 1. + increment / 2, increment)):
            ccs1 = segment.local_coordinate_system(rel_pos)
            length_numeric += np.linalg.norm(ccs1.origin - ccs0.origin)
            ccs0 = copy.deepcopy(ccs1)

        relative_change = length_numeric / length_numeric_prev

        length_numeric_prev = copy.deepcopy(length_numeric)
        num_segments *= 2

        if math.isclose(relative_change, 1, abs_tol=tolerance / 10):
            break
        assert i < num_iterations - 1, "Segment length could not be " \
                                       "determined numerically"

    assert math.isclose(length_numeric, segment.length, abs_tol=tolerance)


def check_trace_segment_orientation(segment):
    # The initial orientation of a segment must be [0, 1, 0]
    lcs = segment.local_coordinate_system(0)
    helpers.check_vectors_identical(lcs.basis[:, 1], np.array([0, 1, 0]))

    delta = 1E-9
    for rel_pos in np.arange(0.1, 1.01, 0.1):
        lcs = segment.local_coordinate_system(rel_pos)
        lcs_d = segment.local_coordinate_system(rel_pos - delta)
        trace_direction_numerical = tf.normalize(lcs.origin - lcs_d.origin)

        # Check that the y-axis is always aligned with the trace's direction
        helpers.check_vectors_identical(lcs.basis[:, 0],
                                        trace_direction_numerical, 1E-6)


def default_trace_segment_tests(segment, tolerance_length=1E-9):
    lcs = segment.local_coordinate_system(0)

    # test that function actually returns a coordinate system class
    assert isinstance(lcs, tf.LocalCoordinateSystem)

    # check that origin for weight 0 is at [0,0,0]
    for i in range(3):
        assert math.isclose(lcs.origin[i], 0)

    check_trace_segment_length(segment, tolerance_length)
    check_trace_segment_orientation(segment)


def test_linear_horizontal_trace_segment():
    length = 7.13
    segment = geo.LinearHorizontalTraceSegment(length)

    # default tests
    default_trace_segment_tests(segment)

    # getter tests
    assert math.isclose(segment.length, length)

    # invalid inputs
    with pytest.raises(ValueError):
        geo.LinearHorizontalTraceSegment(0)
    with pytest.raises(ValueError):
        geo.LinearHorizontalTraceSegment(-4.61)


def test_radial_horizontal_trace_segment():
    radius = 4.74
    angle = np.pi / 1.23
    segment_cw = geo.RadialHorizontalTraceSegment(radius, angle, True)
    segment_ccw = geo.RadialHorizontalTraceSegment(radius, angle, False)

    # default tests
    default_trace_segment_tests(segment_cw, 1E-4)
    default_trace_segment_tests(segment_ccw, 1E-4)

    # getter tests
    assert math.isclose(segment_cw.angle, angle)
    assert math.isclose(segment_ccw.angle, angle)
    assert math.isclose(segment_cw.radius, radius)
    assert math.isclose(segment_ccw.radius, radius)
    assert segment_cw.is_clockwise
    assert not segment_ccw.is_clockwise

    # check positions
    for weight in np.arange(0.1, 1, 0.1):
        current_angle = angle * weight
        x_exp = np.sin(current_angle) * radius
        y_exp = (1 - np.cos(current_angle)) * radius

        lcs_cw = segment_cw.local_coordinate_system(weight)
        lcs_ccw = segment_ccw.local_coordinate_system(weight)

        assert math.isclose(lcs_cw.origin[0], x_exp)
        assert math.isclose(lcs_cw.origin[1], -y_exp)
        assert math.isclose(lcs_ccw.origin[0], x_exp)
        assert math.isclose(lcs_ccw.origin[1], y_exp)

    # invalid inputs
    with pytest.raises(ValueError):
        geo.RadialHorizontalTraceSegment(0, np.pi)
    with pytest.raises(ValueError):
        geo.RadialHorizontalTraceSegment(-0.53, np.pi)
    with pytest.raises(ValueError):
        geo.RadialHorizontalTraceSegment(1, 0)
    with pytest.raises(ValueError):
        geo.RadialHorizontalTraceSegment(1, -np.pi)


# Test trace class ------------------------------------------------------------

def test_trace_construction():
    linear_segment = geo.LinearHorizontalTraceSegment(1)
    radial_segment = geo.RadialHorizontalTraceSegment(1, np.pi)
    ccs_origin = np.array([2, 3, -2])
    ccs = helpers.rotated_coordinate_system(origin=ccs_origin)

    # test single segment construction --------------------
    trace = geo.Trace(linear_segment, ccs)
    assert math.isclose(trace.length, linear_segment.length)
    assert trace.num_segments == 1

    segments = trace.segments
    assert len(segments) == 1
    assert isinstance(segments[0], type(linear_segment))
    assert math.isclose(linear_segment.length, segments[0].length)

    helpers.check_matrices_identical(ccs.basis, trace.coordinate_system.basis)
    helpers.check_vectors_identical(ccs.origin, trace.coordinate_system.origin)

    # test multi segment construction ---------------------
    trace = geo.Trace([radial_segment, linear_segment])
    assert math.isclose(trace.length,
                        linear_segment.length + radial_segment.length)
    assert trace.num_segments == 2

    segments = trace.segments
    assert len(segments) == 2
    assert isinstance(segments[0], type(radial_segment))
    assert isinstance(segments[1], type(linear_segment))

    assert math.isclose(radial_segment.radius, segments[0].radius)
    assert math.isclose(radial_segment.angle, segments[0].angle)
    assert math.isclose(radial_segment.is_clockwise, segments[0].is_clockwise)
    assert math.isclose(linear_segment.length, segments[1].length)

    helpers.check_matrices_identical(np.identity(3),
                                     trace.coordinate_system.basis)
    helpers.check_vectors_identical(np.array([0, 0, 0]),
                                    trace.coordinate_system.origin)

    # check invalid inputs --------------------------------
    with pytest.raises(TypeError):
        geo.Trace(radial_segment, linear_segment)
    with pytest.raises(TypeError):
        geo.Trace(radial_segment, 2)
    with pytest.raises(Exception):
        geo.Trace(None)

    # check construction with custom segment --------------
    class CustomSegment():
        def __init__(self):
            self.length = None

        @staticmethod
        def local_coordinate_system(*args):
            return tf.LocalCoordinateSystem()

    custom_segment = CustomSegment()
    custom_segment.length = 3
    geo.Trace(custom_segment)

    with pytest.raises(Exception):
        custom_segment.length = -12
        geo.Trace(custom_segment)
    with pytest.raises(Exception):
        custom_segment.length = 0
        geo.Trace(custom_segment)


def test_trace_local_coordinate_system():
    radial_segment = geo.RadialHorizontalTraceSegment(1, np.pi)
    linear_segment = geo.LinearHorizontalTraceSegment(1)

    # check with default coordinate system ----------------
    trace = geo.Trace([radial_segment, linear_segment])

    # check first segment
    for i in range(11):
        weight = i / 10
        position = radial_segment.length * weight
        cs_trace = trace.local_coordinate_system(position)
        cs_segment = radial_segment.local_coordinate_system(weight)

        helpers.check_matrices_identical(cs_trace.basis, cs_segment.basis)
        helpers.check_vectors_identical(cs_trace.origin, cs_segment.origin)

    # check second segment
    expected_basis = radial_segment.local_coordinate_system(1).basis
    for i in range(11):
        weight = i / 10
        position_on_segment = linear_segment.length * weight
        position = radial_segment.length + position_on_segment

        expected_origin = np.array([-position_on_segment, 2, 0])
        cs_trace = trace.local_coordinate_system(position)

        helpers.check_matrices_identical(cs_trace.basis, expected_basis)
        helpers.check_vectors_identical(cs_trace.origin, expected_origin)

    # check with arbitrary coordinate system --------------
    basis = tf.rotation_matrix_x(np.pi / 2)
    origin = np.array([-3, 2.5, 5])
    cs_base = tf.LocalCoordinateSystem(basis, origin)

    trace = geo.Trace([radial_segment, linear_segment], cs_base)

    # check first segment
    for i in range(11):
        weight = i / 10
        position = radial_segment.length * weight
        cs_trace = trace.local_coordinate_system(position)
        cs_segment = radial_segment.local_coordinate_system(weight)

        expected_basis = np.matmul(basis, cs_segment.basis)
        expected_origin = np.matmul(basis, cs_segment.origin) + origin

        helpers.check_matrices_identical(cs_trace.basis, expected_basis)
        helpers.check_vectors_identical(cs_trace.origin, expected_origin)

    # check second segment
    expected_basis = np.matmul(basis,
                               radial_segment.local_coordinate_system(1).basis)
    for i in range(11):
        weight = i / 10
        position_on_segment = linear_segment.length * weight
        position = radial_segment.length + position_on_segment

        expected_origin = np.array([-position_on_segment, 0, 2]) + origin
        cs_trace = trace.local_coordinate_system(position)

        helpers.check_matrices_identical(cs_trace.basis, expected_basis)
        helpers.check_vectors_identical(cs_trace.origin, expected_origin)


def test_trace_rasterization():
    radial_segment = geo.RadialHorizontalTraceSegment(1, np.pi)
    linear_segment = geo.LinearHorizontalTraceSegment(1)

    # check with default coordinate system ----------------
    trace = geo.Trace([linear_segment, radial_segment])
    data = trace.rasterize(0.1)

    # no duplications
    assert helpers.are_all_points_unique(data)

    raster_width_eff = trace.length / (data.shape[1] - 1)
    for i in range(data.shape[1]):
        trace_location = i * raster_width_eff
        if trace_location <= 1:
            helpers.check_vectors_identical([trace_location, 0, 0], data[:, i])
        else:
            arc_location = trace_location - 1
            angle = arc_location  # radius 1!
            x = np.sin(angle) + 1  # radius 1!
            y = 1 - np.cos(angle)
            helpers.check_vectors_identical([x, y, 0], data[:, i])

    # check with arbitrary coordinate system --------------
    basis = tf.rotation_matrix_y(np.pi / 2)
    origin = np.array([-3, 2.5, 5])
    cs_base = tf.LocalCoordinateSystem(basis, origin)

    trace = geo.Trace([linear_segment, radial_segment], cs_base)
    data = trace.rasterize(0.1)

    raster_width_eff = trace.length / (data.shape[1] - 1)

    for i in range(data.shape[1]):
        trace_location = i * raster_width_eff
        if trace_location <= 1:
            x = origin[0]
            y = origin[1]
            z = origin[2] - trace_location
        else:
            arc_location = trace_location - 1
            angle = arc_location  # radius 1!
            x = origin[0]
            y = origin[1] + 1 - np.cos(angle)
            z = origin[2] - 1 - np.sin(angle)

        helpers.check_vectors_identical([x, y, z], data[:, i])

    # check if raster width is clipped to valid range -----
    data = trace.rasterize(1000)

    assert data.shape[1] == 2
    helpers.check_vectors_identical([-3, 2.5, 5], data[:, 0])
    helpers.check_vectors_identical([-3, 4.5, 4], data[:, 1])

    # exceptions ------------------------------------------
    with pytest.raises(Exception):
        trace.rasterize(0)
    with pytest.raises(Exception):
        trace.rasterize(-23.1)


# Profile interpolation classes -----------------------------------------------


def check_interpolated_profile_points(profile, c_0, c_1, c_2):
    helpers.check_vectors_identical(profile.shapes[0].segments[0].point_start,
                                    c_0)
    helpers.check_vectors_identical(profile.shapes[0].segments[0].point_end,
                                    c_1)
    helpers.check_vectors_identical(profile.shapes[1].segments[0].point_start,
                                    c_1)
    helpers.check_vectors_identical(profile.shapes[1].segments[0].point_end,
                                    c_2)


def test_linear_profile_interpolation_sbs():
    a_0 = [0, 0]
    a_1 = [8, 16]
    a_2 = [16, 0]
    shape_a01 = geo.Shape(geo.LineSegment.construct_with_points(a_0, a_1))
    shape_a12 = geo.Shape(geo.LineSegment.construct_with_points(a_1, a_2))
    profile_a = geo.Profile([shape_a01, shape_a12])

    b_0 = [-4, 8]
    b_1 = [0, 8]
    b_2 = [16, -16]
    shape_b01 = geo.Shape(geo.LineSegment.construct_with_points(b_0, b_1))
    shape_b12 = geo.Shape(geo.LineSegment.construct_with_points(b_1, b_2))
    profile_b = geo.Profile([shape_b01, shape_b12])

    [profile_a, profile_b] = get_default_profiles()

    for i in range(5):
        weight = i / 4.
        profile_c = geo.linear_profile_interpolation_sbs(profile_a, profile_b,
                                                         weight)
        check_interpolated_profile_points(profile_c,
                                          [-i, 2 * i],
                                          [8 - 2 * i, 16 - 2 * i],
                                          [16, -4 * i])

    # check weight clipped to valid range -----------------

    profile_c = geo.linear_profile_interpolation_sbs(profile_a, profile_b, -3)

    check_interpolated_profile_points(profile_c, a_0, a_1, a_2)

    profile_c = geo.linear_profile_interpolation_sbs(profile_a, profile_b, 42)

    check_interpolated_profile_points(profile_c, b_0, b_1, b_2)

    # exceptions ------------------------------------------

    # number of shapes differ
    profile_d = geo.Profile([shape_b01, shape_b12, shape_a12])
    with pytest.raises(Exception):
        geo.linear_profile_interpolation_sbs(profile_d, profile_b, 0.5)

    # number of segments differ
    shape_b012 = geo.Shape([geo.LineSegment.construct_with_points(b_0, b_1),
                            geo.LineSegment.construct_with_points(b_1, b_2)])

    profile_b2 = geo.Profile([shape_b01, shape_b012])
    with pytest.raises(Exception):
        geo.linear_profile_interpolation_sbs(profile_a, profile_b2, 0.2)


# test variable profile -------------------------------------------------------

def check_variable_profile_state(variable_profile, locations):
    num_profiles = len(locations)
    assert variable_profile.num_interpolation_schemes == num_profiles - 1
    assert variable_profile.num_locations == num_profiles
    assert variable_profile.num_profiles == num_profiles

    for i in range(num_profiles):
        assert math.isclose(locations[i], variable_profile.locations[i])


def test_variable_profile_construction():
    interpol = geo.linear_profile_interpolation_sbs

    profile_a, profile_b = get_default_profiles()

    # construction with single location and interpolation
    variable_profile = geo.VariableProfile([profile_a, profile_b],
                                           1,
                                           interpol)
    check_variable_profile_state(variable_profile, [0, 1])
    variable_profile = geo.VariableProfile([profile_a, profile_b],
                                           [1],
                                           [interpol])
    check_variable_profile_state(variable_profile, [0, 1])

    # construction with location list
    variable_profile = geo.VariableProfile([profile_a, profile_b],
                                           [0, 1],
                                           interpol)
    check_variable_profile_state(variable_profile, [0, 1])

    variable_profile = geo.VariableProfile([profile_a, profile_b, profile_a],
                                           [1, 2],
                                           [interpol, interpol])
    check_variable_profile_state(variable_profile, [0, 1, 2])

    variable_profile = geo.VariableProfile([profile_a, profile_b, profile_a],
                                           [0, 1, 2],
                                           [interpol, interpol])
    check_variable_profile_state(variable_profile, [0, 1, 2])

    # exceptions ------------------------------------------

    # first location is not 0
    with pytest.raises(Exception):
        geo.VariableProfile([profile_a, profile_b], [1, 2], interpol)

    # number of locations is not correct
    with pytest.raises(Exception):
        geo.VariableProfile([profile_a, profile_b, profile_a], [1],
                            [interpol, interpol])
    with pytest.raises(Exception):
        geo.VariableProfile([profile_a, profile_b], [0, 1, 2],
                            interpol)

    # number of interpolations is not correct
    with pytest.raises(Exception):
        geo.VariableProfile([profile_a, profile_b, profile_a], [0, 1, 2],
                            [interpol])
    with pytest.raises(Exception):
        geo.VariableProfile([profile_a, profile_b, profile_a], [0, 1, 2],
                            [interpol, interpol, interpol])

    # locations not ordered
    with pytest.raises(Exception):
        geo.VariableProfile([profile_a, profile_b, profile_a], [0, 2, 1],
                            [interpol, interpol])


def test_variable_profile_local_profile():
    interpol = geo.linear_profile_interpolation_sbs

    profile_a, profile_b = get_default_profiles()
    variable_profile = geo.VariableProfile([profile_a, profile_b, profile_a],
                                           [0, 1, 2],
                                           [interpol, interpol])

    for i in range(5):
        # first segment
        location = i / 4.
        profile = variable_profile.local_profile(location)
        check_interpolated_profile_points(profile,
                                          [-i, 2 * i],
                                          [8 - 2 * i, 16 - 2 * i],
                                          [16, -4 * i])
        # second segment
        location += 1
        profile = variable_profile.local_profile(location)
        check_interpolated_profile_points(profile,
                                          [-4 + i, 8 - 2 * i],
                                          [2 * i, 8 + 2 * i],
                                          [16, -16 + 4 * i])

    # check if values are clipped to valid range

    profile = variable_profile.local_profile(177)
    check_interpolated_profile_points(profile, [0, 0], [8, 16], [16, 0])

    profile = variable_profile.local_profile(-2)
    check_interpolated_profile_points(profile, [0, 0], [8, 16], [16, 0])


# test geometry class ---------------------------------------------------------

def check_variable_profiles_identical(a, b):
    assert a.num_profiles == b.num_profiles
    assert a.num_locations == b.num_locations
    assert a.num_interpolation_schemes == b.num_interpolation_schemes

    for i in range(a.num_profiles):
        check_profiles_identical(a.profiles[i], b.profiles[i])
    for i in range(a.num_locations):
        assert math.isclose(a.locations[i], b.locations[i])
    for i in range(a.num_interpolation_schemes):
        assert isinstance(a.interpolation_schemes[i],
                          type(b.interpolation_schemes[i]))


def test_geometry_construction():
    profile_a, profile_b = get_default_profiles()
    variable_profile = \
        geo.VariableProfile([profile_a, profile_b],
                            [0, 1],
                            geo.linear_profile_interpolation_sbs)

    radial_segment = geo.RadialHorizontalTraceSegment(1, np.pi)
    linear_segment = geo.LinearHorizontalTraceSegment(1)
    trace = geo.Trace([radial_segment, linear_segment])

    # single profile construction
    geometry = geo.Geometry(profile_a, trace)
    check_profiles_identical(geometry.profile, profile_a)
    check_traces_identical(geometry.trace, trace)

    # variable profile construction
    geometry = geo.Geometry(variable_profile, trace)
    check_variable_profiles_identical(geometry.profile, variable_profile)
    check_traces_identical(geometry.trace, trace)

    # exceptions ------------------------------------------

    # wrong types
    with pytest.raises(TypeError):
        geo.Geometry(variable_profile, profile_b)
    with pytest.raises(TypeError):
        geo.Geometry(trace, trace)
    with pytest.raises(TypeError):
        geo.Geometry(trace, profile_b)
    with pytest.raises(TypeError):
        geo.Geometry(variable_profile, "a")
    with pytest.raises(TypeError):
        geo.Geometry("42", trace)


def test_geometry_rasterization_trace():
    a0 = [1, 0]
    a1 = [1, 1]
    a2 = [0, 1]
    a3 = [-1, 1]
    a4 = [-1, 0]

    shape_a012 = geo.Shape([geo.LineSegment.construct_with_points(a0, a1),
                            geo.LineSegment.construct_with_points(a1, a2)])
    shape_a234 = geo.Shape([geo.LineSegment.construct_with_points(a2, a3),
                            geo.LineSegment.construct_with_points(a3, a4)])

    profile_a = geo.Profile([shape_a012, shape_a234])

    radial_segment = geo.RadialHorizontalTraceSegment(1, np.pi / 2, False)
    linear_segment = geo.LinearHorizontalTraceSegment(1)
    trace = geo.Trace([linear_segment, radial_segment])

    geometry = geo.Geometry(profile_a, trace)

    # Note, if the raster width is larger than the segment, it is automatically
    # adjusted to the segment width. Hence each rasterized profile has 6
    # points, which were defined at the beginning of the test (a2 is
    # included twice)
    data = geometry.rasterize(7, 0.1)

    num_raster_profiles = int(np.round(data.shape[1] / 6))
    profile_points = np.array([a0, a1, a2, a2, a3, a4]).transpose()

    eff_raster_width = trace.length / (data.shape[1] / 6 - 1)
    arc_point_distance_on_trace = 2 * np.sin(eff_raster_width / 2)

    for i in range(num_raster_profiles):
        idx_0 = i * 6
        if data[0, idx_0 + 2] <= 1:
            x = data[0, idx_0]
            assert math.isclose(x, eff_raster_width * i, abs_tol=1E-6)
            for j in range(6):
                assert math.isclose(data[1, idx_0 + j], profile_points[0, j])
                assert math.isclose(data[2, idx_0 + j], profile_points[1, j])
                assert math.isclose(data[0, idx_0 + j], data[0, idx_0])
        else:
            assert math.isclose(data[0, idx_0], 1)
            assert math.isclose(data[1, idx_0], a0[0])
            assert math.isclose(data[2, idx_0], a0[1])
            assert math.isclose(data[0, idx_0 + 1], 1)
            assert math.isclose(data[1, idx_0 + 1], a1[0])
            assert math.isclose(data[2, idx_0 + 1], a1[1])

            # z-values are constant
            for j in np.arange(2, 6, 1):
                assert math.isclose(data[2, idx_0 + j], profile_points[1, j])

            # all profile points in a common x-y plane
            exp_radius = np.array([1, 1, 2, 2])
            vec_02 = data[0:2, idx_0 + 2] - data[0:2, idx_0]
            assert math.isclose(np.linalg.norm(vec_02), exp_radius[0])
            for j in np.arange(3, 6, 1):
                vec_0j = data[0:2, idx_0 + j] - data[0:2, idx_0]
                assert math.isclose(np.linalg.norm(vec_0j), exp_radius[j - 2])
                unit_vec_0j = tf.normalize(vec_0j)
                assert math.isclose(np.dot(unit_vec_0j, vec_02), 1)

            # check point distance between profiles
            if data[1, idx_0 - 4] > 1:
                exp_point_distance = arc_point_distance_on_trace * exp_radius
                for j in np.arange(2, 6, 1):
                    point_distance = np.linalg.norm(
                        data[:, idx_0 + j] - data[:, idx_0 + j - 6])
                    assert math.isclose(exp_point_distance[j - 2],
                                        point_distance)

    # check if raster width is clipped to valid range -----
    data = geometry.rasterize(7, 1000)

    assert data.shape[1] == 12

    for i in range(12):
        if i < 6:
            math.isclose(data[0, i], 0)
        else:
            assert math.isclose(data[1, i], 1)

    # exceptions ------------------------------------------
    with pytest.raises(Exception):
        geometry.rasterize(0, 1)
    with pytest.raises(Exception):
        geometry.rasterize(1, 0)
    with pytest.raises(Exception):
        geometry.rasterize(0, 0)
    with pytest.raises(Exception):
        geometry.rasterize(-2.3, 1)
    with pytest.raises(Exception):
        geometry.rasterize(1, -4.6)
    with pytest.raises(Exception):
        geometry.rasterize(-2.3, -4.6)


def test_geometry_rasterization_profile_interpolation():
    interpol = geo.linear_profile_interpolation_sbs

    a0 = [1, 0]
    a1 = [1, 1]
    a2 = [0, 1]
    a3 = [-1, 1]
    a4 = [-1, 0]

    shape_a012 = geo.Shape([geo.LineSegment.construct_with_points(a0, a1),
                            geo.LineSegment.construct_with_points(a1, a2)])
    shape_a234 = geo.Shape([geo.LineSegment.construct_with_points(a2, a3),
                            geo.LineSegment.construct_with_points(a3, a4)])

    shape_b012 = copy.deepcopy(shape_a012)
    shape_b234 = copy.deepcopy(shape_a234)
    shape_b012.apply_transformation([[2, 0], [0, 2]])
    shape_b234.apply_transformation([[2, 0], [0, 2]])

    profile_a = geo.Profile([shape_a012, shape_a234])
    profile_b = geo.Profile([shape_b012, shape_b234])

    variable_profile = geo.VariableProfile([profile_a, profile_b, profile_a],
                                           [0, 2, 6], [interpol, interpol])

    linear_segment_l1 = geo.LinearHorizontalTraceSegment(1)
    linear_segment_l2 = geo.LinearHorizontalTraceSegment(2)
    # Note: The profile in the middle is not located at the start of the
    # second segment
    trace = geo.Trace([linear_segment_l2, linear_segment_l1])

    geometry = geo.Geometry(variable_profile, trace)

    # Note: If the raster width is larger than the segment, it is automatically
    # adjusted to the segment width. Hence each rasterized profile has 6
    # points, which were defined at the beginning of the test (a2 is
    # included twice)
    data = geometry.rasterize(7, 0.1)
    assert data.shape[1] == 186

    profile_points = np.array([a0, a1, a2, a2, a3, a4]).transpose()

    # check first segment
    for i in range(11):
        idx_0 = i * 6
        for j in range(6):
            exp_point = np.array([i * 0.1,
                                  profile_points[0, j] * (1 + i * 0.1),
                                  profile_points[1, j] * (1 + i * 0.1)])
            helpers.check_vectors_identical(data[:, idx_0 + j], exp_point)

    # check second segment
    for i in range(20):
        idx_0 = (30 - i) * 6
        for j in range(6):
            exp_point = np.array([3 - i * 0.1,
                                  profile_points[0, j] * (1 + i * 0.05),
                                  profile_points[1, j] * (1 + i * 0.05)])
            helpers.check_vectors_identical(data[:, idx_0 + j], exp_point)
