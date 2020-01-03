import pytest
import mypackage.point_cloud_generator as pcg
import mypackage.geometry as geo


def test_profile_construction():
    shape = geo.Shape2D([0, 0], [1, 0])
    shape.add_segment([2, -1])
    shape.add_segment([0, -1])

    # Check valid types
    pcg.Profile(shape)
    pcg.Profile([shape, shape])

    # Check invalid types
    with pytest.raises(ValueError):
        pcg.Profile(3)
    with pytest.raises(ValueError):
        pcg.Profile("This is not right")
    with pytest.raises(ValueError):
        pcg.Profile([2, 8, 1])
