import pytest
import mypackage.point_cloud_generator as pcg
import mypackage.geometry as geo
import mypackage.transformations as tf
import math
import numpy as np
import copy
import tests._helpers as helpers


# helpers ---------------------------------------------------------------------

def check_profiles_identical(a, b):
    assert a.num_shapes == b.num_shapes
    for i in range(a.num_shapes):
        check_shapes_identical(a.shapes[i], b.shapes[i])


def check_shapes_identical(a, b):
    assert a.num_segments == b.num_segments
    for i in range(a.num_segments):
        assert isinstance(a.segments[i], type(b.segments[i]))
        helpers.check_vectors_identical(a.segments[i].point_start,
                                        b.segments[i].point_start)
        helpers.check_vectors_identical(a.segments[i].point_end,
                                        b.segments[i].point_end)
        if isinstance(a.segments[i], geo.ArcSegment):
            helpers.check_vectors_identical(a.segments[i].point_center,
                                            b.segments[i].point_center)


def check_trace_segments_identical(a, b):
    assert isinstance(a, type(b))
    if isinstance(a, pcg.LinearHorizontalTraceSegment):
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
    shape_a01 = geo.Shape2D(geo.LineSegment.construct_from_points(a_0, a_1))
    shape_a12 = geo.Shape2D(geo.LineSegment.construct_from_points(a_1, a_2))
    profile_a = pcg.Profile([shape_a01, shape_a12])

    b_0 = [-4, 8]
    b_1 = [0, 8]
    b_2 = [16, -16]
    shape_b01 = geo.Shape2D(geo.LineSegment.construct_from_points(b_0, b_1))
    shape_b12 = geo.Shape2D(geo.LineSegment.construct_from_points(b_1, b_2))
    profile_b = pcg.Profile([shape_b01, shape_b12])
    return [profile_a, profile_b]


# Test profile class ----------------------------------------------------------

def test_profile_construction_and_shape_addition():
    segment0 = geo.LineSegment.construct_from_points([0, 0], [1, 0])
    segment1 = geo.LineSegment.construct_from_points([1, 0], [2, -1])
    segment2 = geo.LineSegment.construct_from_points([2, -1], [0, -1])

    shape = geo.Shape2D([segment0, segment1, segment2])

    # Check invalid types
    with pytest.raises(TypeError):
        pcg.Profile(3)
    with pytest.raises(TypeError):
        pcg.Profile("This is not right")
    with pytest.raises(TypeError):
        pcg.Profile([2, 8, 1])

    # Check valid types
    profile = pcg.Profile(shape)
    assert profile.num_shapes == 1
    profile = pcg.Profile([shape, shape])
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
    shape0 = geo.Shape2D(
        geo.LineSegment.construct_from_points([-1, 0], [-raster_width, 0]))
    shape1 = geo.Shape2D(geo.LineSegment.construct_from_points([0, 0], [1, 0]))
    shape2 = geo.Shape2D(
        geo.LineSegment.construct_from_points([1 + raster_width, 0], [2, 0]))

    profile = pcg.Profile([shape0, shape1])
    profile.add_shapes(shape2)

    # rasterize
    data = profile.rasterize(0.1)

    # check raster data size
    expected_number_raster_points = int(round(3 / raster_width)) + 1
    assert data.shape[1] == expected_number_raster_points

    # Check that all shapes are rasterized correct
    for i in range(int(round(3 / raster_width)) + 1):
        expected_raster_point_x = i * raster_width - 1
        assert data[0, i] - expected_raster_point_x < 1E-9
        assert data[1, i] == 0


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
        helpers.check_vectors_identical(lcs.basis[:, 1],
                                        trace_direction_numerical, 1E-6)


def default_trace_segment_tests(segment, tolerance_length=1E-9):
    lcs = segment.local_coordinate_system(0)

    # test that function actually returns a coordinate system class
    assert isinstance(lcs, tf.CartesianCoordinateSystem3d)

    # check that origin for weight 0 is at [0,0,0]
    for i in range(3):
        assert math.isclose(lcs.origin[i], 0)

    check_trace_segment_length(segment, tolerance_length)
    check_trace_segment_orientation(segment)


def test_linear_horizontal_trace_segment():
    length = 7.13
    segment = pcg.LinearHorizontalTraceSegment(length)

    # default tests
    default_trace_segment_tests(segment)

    # getter tests
    assert math.isclose(segment.length, length)

    # invalid inputs
    with pytest.raises(ValueError):
        pcg.LinearHorizontalTraceSegment(0)
    with pytest.raises(ValueError):
        pcg.LinearHorizontalTraceSegment(-4.61)


def test_radial_horizontal_trace_segment():
    radius = 4.74
    angle = np.pi / 1.23
    segment_cw = pcg.RadialHorizontalTraceSegment(radius, angle, True)
    segment_ccw = pcg.RadialHorizontalTraceSegment(radius, angle, False)

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
        x_exp = (1 - np.cos(current_angle)) * radius
        y_exp = np.sin(current_angle) * radius

        lcs_cw = segment_cw.local_coordinate_system(weight)
        lcs_ccw = segment_ccw.local_coordinate_system(weight)

        assert math.isclose(lcs_cw.origin[0], x_exp)
        assert math.isclose(lcs_cw.origin[1], y_exp)
        assert math.isclose(lcs_ccw.origin[0], -x_exp)
        assert math.isclose(lcs_ccw.origin[1], y_exp)

    # invalid inputs
    with pytest.raises(ValueError):
        pcg.RadialHorizontalTraceSegment(0, np.pi)
    with pytest.raises(ValueError):
        pcg.RadialHorizontalTraceSegment(-0.53, np.pi)
    with pytest.raises(ValueError):
        pcg.RadialHorizontalTraceSegment(1, 0)
    with pytest.raises(ValueError):
        pcg.RadialHorizontalTraceSegment(1, -np.pi)


# Test trace class ------------------------------------------------------------

def test_trace_construction():
    linear_segment = pcg.LinearHorizontalTraceSegment(1)
    radial_segment = pcg.RadialHorizontalTraceSegment(1, np.pi)
    ccs_origin = np.array([2, 3, -2])
    ccs = helpers.rotated_coordinate_system(origin=ccs_origin)

    # test single segment construction --------------------
    trace = pcg.Trace(linear_segment, ccs)
    assert math.isclose(trace.length, linear_segment.length)
    assert trace.num_segments == 1

    segments = trace.segments
    assert len(segments) == 1
    assert isinstance(segments[0], type(linear_segment))
    assert math.isclose(linear_segment.length, segments[0].length)

    helpers.check_matrices_identical(ccs.basis, trace.coordinate_system.basis)
    helpers.check_vectors_identical(ccs.origin, trace.coordinate_system.origin)

    # test multi segment construction ---------------------
    trace = pcg.Trace([radial_segment, linear_segment])
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
        pcg.Trace(radial_segment, linear_segment)
    with pytest.raises(TypeError):
        pcg.Trace(radial_segment, 2)
    with pytest.raises(Exception):
        pcg.Trace(None)

    # check construction with custom segment --------------
    class CustomSegment():
        @staticmethod
        def local_coordinate_system(*args):
            return tf.CartesianCoordinateSystem3d

    custom_segment = CustomSegment()
    custom_segment.length = 3
    pcg.Trace(custom_segment)

    with pytest.raises(Exception):
        custom_segment.length = -12
        pcg.Trace(custom_segment)
    with pytest.raises(Exception):
        custom_segment.length = 0
        pcg.Trace(custom_segment)


def test_trace_local_coordinate_system():
    radial_segment = pcg.RadialHorizontalTraceSegment(1, np.pi)
    linear_segment = pcg.LinearHorizontalTraceSegment(1)

    # check with default coordinate system ----------------
    trace = pcg.Trace([radial_segment, linear_segment])

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

        expected_origin = np.array([-2, -position_on_segment, 0])
        cs_trace = trace.local_coordinate_system(position)

        helpers.check_matrices_identical(cs_trace.basis, expected_basis)
        helpers.check_vectors_identical(cs_trace.origin, expected_origin)

    # check with arbitrary coordinate system --------------
    basis = tf.rotation_matrix_x(np.pi / 2)
    origin = np.array([-3, 2.5, 5])
    cs_base = tf.CartesianCoordinateSystem3d(basis, origin)

    trace = pcg.Trace([radial_segment, linear_segment], cs_base)

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

        expected_origin = np.array([-2, 0, -position_on_segment]) + origin
        cs_trace = trace.local_coordinate_system(position)

        helpers.check_matrices_identical(cs_trace.basis, expected_basis)
        helpers.check_vectors_identical(cs_trace.origin, expected_origin)


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
    shape_a01 = geo.Shape2D(geo.LineSegment.construct_from_points(a_0, a_1))
    shape_a12 = geo.Shape2D(geo.LineSegment.construct_from_points(a_1, a_2))
    profile_a = pcg.Profile([shape_a01, shape_a12])

    b_0 = [-4, 8]
    b_1 = [0, 8]
    b_2 = [16, -16]
    shape_b01 = geo.Shape2D(geo.LineSegment.construct_from_points(b_0, b_1))
    shape_b12 = geo.Shape2D(geo.LineSegment.construct_from_points(b_1, b_2))
    profile_b = pcg.Profile([shape_b01, shape_b12])

    [profile_a, profile_b] = get_default_profiles()

    for i in range(5):
        weight = i / 4.
        profile_c = pcg.LinearProfileInterpolationSBS.interpolate(profile_a,
                                                                  profile_b,
                                                                  weight)
        check_interpolated_profile_points(profile_c,
                                          [-i, 2 * i],
                                          [8 - 2 * i, 16 - 2 * i],
                                          [16, -4 * i])

    # check weight clipped to valid range -----------------

    profile_c = pcg.LinearProfileInterpolationSBS.interpolate(profile_a,
                                                              profile_b,
                                                              -3)

    check_interpolated_profile_points(profile_c, a_0, a_1, a_2)

    profile_c = pcg.LinearProfileInterpolationSBS.interpolate(profile_a,
                                                              profile_b,
                                                              42)

    check_interpolated_profile_points(profile_c, b_0, b_1, b_2)

    # exceptions ------------------------------------------

    # number of shapes differ
    profile_d = pcg.Profile([shape_b01, shape_b12, shape_a12])
    with pytest.raises(Exception):
        pcg.LinearProfileInterpolationSBS.interpolate(profile_d, profile_b,
                                                      0.5)

    # number of segments differ
    shape_b012 = geo.Shape2D([geo.LineSegment.construct_from_points(b_0, b_1),
                              geo.LineSegment.construct_from_points(b_1, b_2)])

    profile_b2 = pcg.Profile([shape_b01, shape_b012])
    with pytest.raises(Exception):
        pcg.LinearProfileInterpolationSBS.interpolate(profile_a,
                                                      profile_b2,
                                                      0.2)


# test variable profile -------------------------------------------------------

def check_variable_profile_state(variable_profile, locations):
    num_profiles = len(locations)
    assert variable_profile.num_interpolation_schemes == num_profiles - 1
    assert variable_profile.num_locations == num_profiles
    assert variable_profile.num_profiles == num_profiles

    for i in range(num_profiles):
        assert math.isclose(locations[i], variable_profile.locations[i])


def test_variable_profile_construction():
    interpol = pcg.LinearProfileInterpolationSBS

    profile_a, profile_b = get_default_profiles()

    # construction with single location and interpolation
    variable_profile = pcg.VariableProfile([profile_a, profile_b],
                                           1,
                                           interpol)
    check_variable_profile_state(variable_profile, [0, 1])
    variable_profile = pcg.VariableProfile([profile_a, profile_b],
                                           [1],
                                           [interpol])
    check_variable_profile_state(variable_profile, [0, 1])

    # construction with location list
    variable_profile = pcg.VariableProfile([profile_a, profile_b],
                                           [0, 1],
                                           interpol)
    check_variable_profile_state(variable_profile, [0, 1])

    variable_profile = pcg.VariableProfile([profile_a, profile_b, profile_a],
                                           [1, 2],
                                           [interpol, interpol])
    check_variable_profile_state(variable_profile, [0, 1, 2])

    variable_profile = pcg.VariableProfile([profile_a, profile_b, profile_a],
                                           [0, 1, 2],
                                           [interpol, interpol])
    check_variable_profile_state(variable_profile, [0, 1, 2])

    # exceptions ------------------------------------------

    # first location is not 0
    with pytest.raises(Exception):
        pcg.VariableProfile([profile_a, profile_b], [1, 2], interpol)

    # number of locations is not correct
    with pytest.raises(Exception):
        pcg.VariableProfile([profile_a, profile_b, profile_a], [1],
                            [interpol, interpol])
    with pytest.raises(Exception):
        pcg.VariableProfile([profile_a, profile_b], [0, 1, 2],
                            interpol)

    # number of interpolations is not correct
    with pytest.raises(Exception):
        pcg.VariableProfile([profile_a, profile_b, profile_a], [0, 1, 2],
                            [interpol])
    with pytest.raises(Exception):
        pcg.VariableProfile([profile_a, profile_b, profile_a], [0, 1, 2],
                            [interpol, interpol, interpol])

    # locations not ordered
    with pytest.raises(Exception):
        pcg.VariableProfile([profile_a, profile_b, profile_a], [0, 2, 1],
                            [interpol, interpol])


def test_variable_profile_local_profile():
    interpol = pcg.LinearProfileInterpolationSBS

    profile_a, profile_b = get_default_profiles()
    variable_profile = pcg.VariableProfile([profile_a, profile_b, profile_a],
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
    variable_profile = pcg.VariableProfile([profile_a, profile_b], [0, 1],
                                           pcg.LinearProfileInterpolationSBS)

    radial_segment = pcg.RadialHorizontalTraceSegment(1, np.pi)
    linear_segment = pcg.LinearHorizontalTraceSegment(1)
    trace = pcg.Trace([radial_segment, linear_segment])

    # single profile construction
    geometry = pcg.Geometry(profile_a, trace)
    check_profiles_identical(geometry.profile, profile_a)
    check_traces_identical(geometry.trace, trace)

    # variable profile construction
    geometry = pcg.Geometry(variable_profile, trace)
    check_variable_profiles_identical(geometry.profile, variable_profile)
    check_traces_identical(geometry.trace, trace)

    # exceptions ------------------------------------------

    # wrong types
    with pytest.raises(TypeError):
        pcg.Geometry(variable_profile, profile_b)
    with pytest.raises(TypeError):
        pcg.Geometry(trace, trace)
    with pytest.raises(TypeError):
        pcg.Geometry(trace, profile_b)
    with pytest.raises(TypeError):
        pcg.Geometry(variable_profile, "a")
    with pytest.raises(TypeError):
        pcg.Geometry("42", trace)


def test_geometry_rasterization_trace():
    a0 = [-1, 0]
    a1 = [-1, 1]
    a2 = [0, 1]
    a3 = [1, 1]
    a4 = [1, 0]

    shape_a012 = geo.Shape2D([geo.LineSegment.construct_from_points(a0, a1),
                              geo.LineSegment.construct_from_points(a1, a2)])
    shape_a234 = geo.Shape2D([geo.LineSegment.construct_from_points(a2, a3),
                              geo.LineSegment.construct_from_points(a3, a4)])

    profile_a = pcg.Profile([shape_a012, shape_a234])

    radial_segment = pcg.RadialHorizontalTraceSegment(1, np.pi / 2)
    linear_segment = pcg.LinearHorizontalTraceSegment(1)
    trace = pcg.Trace([linear_segment, radial_segment])

    geometry = pcg.Geometry(profile_a, trace)

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
        if data[1, idx_0 + 2] <= 1:
            y = data[1, idx_0]
            assert math.isclose(y, eff_raster_width * i, abs_tol=1E-6)

            for j in range(6):
                assert math.isclose(data[0, idx_0 + j], profile_points[0, j])
                assert math.isclose(data[2, idx_0 + j], profile_points[1, j])
                assert math.isclose(data[1, idx_0 + j], data[1, idx_0])
        else:
            assert math.isclose(data[0, idx_0], a0[0])
            assert math.isclose(data[1, idx_0], 1)
            assert math.isclose(data[2, idx_0], a0[1])
            assert math.isclose(data[0, idx_0 + 1], a1[0])
            assert math.isclose(data[1, idx_0 + 1], 1)
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


def test_geometry_rasterization_profile_interpolation():
    interpol = pcg.LinearProfileInterpolationSBS

    a0 = [-1, 0]
    a1 = [-1, 1]
    a2 = [0, 1]
    a3 = [1, 1]
    a4 = [1, 0]

    shape_a012 = geo.Shape2D([geo.LineSegment.construct_from_points(a0, a1),
                              geo.LineSegment.construct_from_points(a1, a2)])
    shape_a234 = geo.Shape2D([geo.LineSegment.construct_from_points(a2, a3),
                              geo.LineSegment.construct_from_points(a3, a4)])

    shape_b012 = copy.deepcopy(shape_a012)
    shape_b234 = copy.deepcopy(shape_a234)
    shape_b012.apply_transformation([[2, 0], [0, 2]])
    shape_b234.apply_transformation([[2, 0], [0, 2]])

    profile_a = pcg.Profile([shape_a012, shape_a234])
    profile_b = pcg.Profile([shape_b012, shape_b234])

    variable_profile = pcg.VariableProfile([profile_a, profile_b, profile_a],
                                           [0, 2, 6], [interpol, interpol])

    linear_segment_l1 = pcg.LinearHorizontalTraceSegment(1)
    linear_segment_l2 = pcg.LinearHorizontalTraceSegment(2)
    # Note: The profile in the middle is not located at the start of the
    # second segment
    trace = pcg.Trace([linear_segment_l2, linear_segment_l1])

    geometry = pcg.Geometry(variable_profile, trace)

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
            exp_point = np.array([profile_points[0, j] * (1 + i * 0.1),
                                  i * 0.1,
                                  profile_points[1, j] * (1 + i * 0.1)])
            helpers.check_vectors_identical(data[:, idx_0 + j], exp_point)

    # check second segment
    for i in range(20):
        idx_0 = (30 - i) * 6
        for j in range(6):
            exp_point = np.array([profile_points[0, j] * (1 + i * 0.05),
                                  3 - i * 0.1,
                                  profile_points[1, j] * (1 + i * 0.05)])
            helpers.check_vectors_identical(data[:, idx_0 + j], exp_point)
