import pytest
import mypackage.geometry as geo


def basic_segment_tests(segment_type, constructor_args):
    # Segment length too small
    with pytest.raises(Exception):
        geo.Shape2D([0, 0], [0, 0], segment_type(*constructor_args))


def test_class_arc_segment():
    arc_segment = geo.Shape2D.ArcSegment(3)
    shape = geo.Shape2D([0, 0], [1, 1], segment=arc_segment)


def test_class_shape2d_point_operations():
    shape = geo.Shape2D([0, 0], [0, 1])
    shape.add_segment(2, 2)
    shape.add_segment(1, 0)
    shape.add_segment(0, 0)

    assert (shape.is_point_included([0, 1]))
    assert (not shape.is_point_included([5, 1]))

    with pytest.raises(ValueError):
        shape.add_segment(1, 0)
