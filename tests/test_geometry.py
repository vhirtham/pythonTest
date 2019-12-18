import mypackage.geometry as geo


def test_class_point_ctor():
    a = geo.Point2D(1, 2)
    assert a.coordinates[0] == 1
    assert a.coordinates[1] == 2
