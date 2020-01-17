import pytest
import mypackage.point_cloud_generator as pcg
import mypackage.geometry as geo
import mypackage.transformations as tf
import math
import numpy as np
import copy


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

def check_trace_segment_length(segment):
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

        if math.isclose(relative_change, 1):
            break
        assert i < num_iterations - 1, "Segment length could not be " \
                                       "determined numerically"

    assert math.isclose(length_numeric, segment.length)


def are_vectors_identical(a, b, tolerance=1E-9):
    if not a.size == b.size:
        return False
    for i in range(a.size):
        if not math.isclose(a[i], b[i], abs_tol=tolerance):
            return False
    return True


def check_trace_segment_orientation(segment):
    # The initial orientation of a segment must be [0, 1, 0]
    lcs = segment.local_coordinate_system(0)
    assert are_vectors_identical(lcs.basis[1], np.array([0, 1, 0]))

    delta = 1E-9
    for rel_pos in np.arange(0.1, 1.01, 0.1):
        lcs = segment.local_coordinate_system(rel_pos)
        lcs_d = segment.local_coordinate_system(rel_pos - delta)
        trace_direction_numerical = tf.normalize(lcs.origin - lcs_d.origin)

        # Check that the y-axis is always aligned with the trace's direction
        assert are_vectors_identical(lcs.basis[1], trace_direction_numerical,
                                     1E-6)


def default_trace_segment_tests(segment):
    lcs = segment.local_coordinate_system(0)

    # test that function actually returns a coordinate system class
    assert isinstance(lcs, tf.CartesianCoordinateSystem3d)

    # check that origin for weight 0 is at [0,0,0]
    for i in range(3):
        assert math.isclose(lcs.origin[i], 0)

    check_trace_segment_length(segment)
    check_trace_segment_orientation(segment)


def test_linear_horizontal_trace_segment():
    length = 7.13
    segment = pcg.LinearHorizontalTraceSegment(length)

    assert math.isclose(segment.length, length)

    default_trace_segment_tests(segment)


test_linear_horizontal_trace_segment()
