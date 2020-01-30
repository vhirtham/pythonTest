"""Test the internal utility functions."""

import numpy as np
import mypackage._utility as utils


def test_vector_is_close():
    vec_a = np.array([0, 1, 2])
    vec_b = np.array([3, 5, 1])

    assert utils.vector_is_close(vec_a, vec_a)
    assert utils.vector_is_close(vec_b, vec_b)
    assert not utils.vector_is_close(vec_a, vec_b)
    assert not utils.vector_is_close(vec_b, vec_a)

    # check tolerance
    vec_c = vec_a + 0.0001
    assert utils.vector_is_close(vec_a, vec_c, abs_tol=0.00011)
    assert not utils.vector_is_close(vec_a, vec_c, abs_tol=0.00009)

    # vectors have different size
    assert not utils.vector_is_close(vec_a, vec_a[0:2])
