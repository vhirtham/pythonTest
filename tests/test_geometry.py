import pytest
import mypackage.geometry as geo


def test_shape2d_with_arc_segment():
    # Invalid center point
    with pytest.raises(ValueError):
        geo.Shape2D([0, 0], [1, 1], geo.Shape2D.ArcSegment([0, 1.1]))

    shape = geo.Shape2D([0, 0], [1, 1], segment=geo.Shape2D.ArcSegment([0, 1]))
    shape.add_segment([2, 2], segment=geo.Shape2D.ArcSegment([2, 1]))
    # Invalid center point
    with pytest.raises(ValueError):
        shape.add_segment([3, 1], segment=geo.Shape2D.ArcSegment([2.1, 1]))


def test_class_shape2d_point_operations():
    # Segment length too small
    with pytest.raises(Exception):
        geo.Shape2D([0, 0], [0, 0])

    # Create shape and add segments
    shape = geo.Shape2D([0, 0], [0, 1])
    shape.add_segment([2, 2])
    shape.add_segment([1, 0])
    shape.add_segment([0, 0])

    assert (shape.is_point_included([0, 1]))
    assert (not shape.is_point_included([5, 1]))

    # Shape is already closed
    with pytest.raises(ValueError):
        shape.add_segment([1, 0])
