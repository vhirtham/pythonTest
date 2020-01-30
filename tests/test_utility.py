"""Test the internal utility functions."""

import numpy as np
import mypackage._utility as utils


def test_is_column_in_matrix():
    c_0 = [1, 5, 2]
    c_1 = [3, 2, 2]
    c_2 = [1, 6, 1]
    c_3 = [1, 6, 0]
    matrix = np.array([c_0, c_1, c_2, c_3]).transpose()

    assert utils.is_column_in_matrix(c_0, matrix)
    assert utils.is_column_in_matrix(c_1, matrix)
    assert utils.is_column_in_matrix(c_2, matrix)
    assert utils.is_column_in_matrix(c_3, matrix)

    assert not utils.is_column_in_matrix([1, 6], matrix)
    assert not utils.is_column_in_matrix([1, 6, 2], matrix)
    assert not utils.is_column_in_matrix([1, 1, 3, 1], matrix)


def test_is_row_in_matrix():
    c_0 = [1, 5, 2]
    c_1 = [3, 2, 2]
    c_2 = [1, 6, 1]
    c_3 = [1, 6, 0]
    matrix = np.array([c_0, c_1, c_2, c_3])

    assert utils.is_row_in_matrix(c_0, matrix)
    assert utils.is_row_in_matrix(c_1, matrix)
    assert utils.is_row_in_matrix(c_2, matrix)
    assert utils.is_row_in_matrix(c_3, matrix)

    assert not utils.is_row_in_matrix([1, 6], matrix)
    assert not utils.is_row_in_matrix([1, 6, 2], matrix)
    assert not utils.is_row_in_matrix([1, 1, 3, 1], matrix)


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
