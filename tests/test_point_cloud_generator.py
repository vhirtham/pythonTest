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
