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


def test_shape2d_construction():
    # Test Exception: Segment length too small
    with pytest.raises(Exception):
        geo.Shape2D([0, 0], [0, 0])

    # Test Exception: Invalid point format
    with pytest.raises(Exception):
        geo.Shape2D([0, 0], [1])
    with pytest.raises(Exception):
        geo.Shape2D([0, 0, 4], [1, 1])

    # Test Exception: Invalid shape type
    with pytest.raises(Exception):
        geo.Shape2D([0, 0], [0, 1], "wrong type")

    # Create shape
    geo.Shape2D([0, 0], [0, 1])


def test_shape2d_boolean_functions():
    shape = geo.Shape2D([0, 0], [0, 1])

    # Point included or not
    assert (shape.is_point_included([0, 1]))
    assert (not shape.is_point_included([5, 1]))


def test_shape2d_segment_addition():
    # Create shape and add segments
    shape = geo.Shape2D([0, 0], [0, 1])
    shape.add_segment([2, 2])
    shape.add_segment([1, 0])

    # Test Exception: Invalid point format
    with pytest.raises(Exception):
        shape.add_segment(0)
    # Test Exception: Invalid shape type
    with pytest.raises(Exception):
        shape.add_segment([0, 0], "wrong type")

    # Close segment
    shape.add_segment([0, 0])

    # Test Exception: Shape is already closed
    with pytest.raises(ValueError):
        shape.add_segment([1, 0])
