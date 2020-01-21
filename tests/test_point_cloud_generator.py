import pytest
import mypackage.point_cloud_generator as pcg
import mypackage.geometry as geo
import mypackage.transformations as tf
import math
import numpy as np
import copy
import tests.helpers as helpers


# Test profile class ----------------------------------------------------------

def test_profile_construction_and_shape_addition():
    shape = geo.Shape2D([0, 0], [1, 0])
    shape.add_segment([2, -1])
    shape.add_segment([0, -1])

    # Check invalid types
    with pytest.raises(TypeError):
        pcg.Profile(3)
    with pytest.raises(TypeError):
        pcg.Profile("This is not right")
    with pytest.raises(TypeError):
        pcg.Profile([2, 8, 1])

    # Check valid types
    profile = pcg.Profile(shape)
    assert profile.num_shapes() == 1
    profile = pcg.Profile([shape, shape])
    assert profile.num_shapes() == 2

    # Check invalid addition
    with pytest.raises(TypeError):
        profile.add_shapes([shape, 0.1])
    with pytest.raises(TypeError):
        profile.add_shapes(["shape"])
    with pytest.raises(TypeError):
        profile.add_shapes(0.1)

    # Check that invalid calls only raise an exception and do not invalidate
    # the internal data
    assert profile.num_shapes() == 2

    # Check valid addition
    profile.add_shapes(shape)
    assert profile.num_shapes() == 3
    profile.add_shapes([shape, shape])
    assert profile.num_shapes() == 5

    # Check shapes
    shapes_profile = profile.shapes
    for shape_profile in shapes_profile:
        assert shape.num_segments() == shape_profile.num_segments()

        segments = shape.segments
        segments_profile = shape_profile.segments

        assert len(segments) == shape.num_segments()
        assert len(segments) == len(segments_profile)

        for i in range(shape.num_segments()):
            assert isinstance(segments_profile[i], type(segments[i]))

        points = shape.points
        points_profile = shape_profile.points

        assert points.shape == points_profile.shape
        for i in range(shape.num_points()):
            helpers.check_vectors_identical(points[i], points_profile[i])


def test_profile_rasterization():
    raster_width = 0.1
    shape0 = geo.Shape2D([-1, 0], [-raster_width, 0])
    shape1 = geo.Shape2D([0, 0], [1, 0])
    shape2 = geo.Shape2D([1 + raster_width, 0], [2, 0])

    profile = pcg.Profile([shape0, shape1])
    profile.add_shapes(shape2)

    # rasterize
    data = profile.rasterize(0.1)

    # check raster data size
    expected_number_raster_points = int(round(3 / raster_width)) + 1
    assert len(data[:, 0]) == expected_number_raster_points

    # Check that all shapes are rasterized correct
    for i in range(int(round(3 / raster_width)) + 1):
        expected_raster_point_x = i * raster_width - 1
        assert data[i, 0] - expected_raster_point_x < 1E-9
        assert data[i, 1] == 0


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
    assert segment_cw.is_clockwise() is True
    assert segment_ccw.is_clockwise() is False

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
    assert trace.num_segments() == 1

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
    assert trace.num_segments() == 2

    segments = trace.segments
    assert len(segments) == 2
    assert isinstance(segments[0], type(radial_segment))
    assert isinstance(segments[1], type(linear_segment))

    assert math.isclose(radial_segment.radius, segments[0].radius)
    assert math.isclose(radial_segment.angle, segments[0].angle)
    assert math.isclose(radial_segment.is_clockwise(),
                        segments[0].is_clockwise())
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
        def local_coordinate_system(self, *args):
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
