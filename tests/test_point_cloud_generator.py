import pytest
import mypackage.point_cloud_generator as pcg
import mypackage.geometry as geo


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
