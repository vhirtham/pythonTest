"""Contains package internal utility functions."""

import math


def to_list(var):
    """
    Store the passed variable into a list and return it.

    If the variable is already a list, it is returned without modification.
    If 'None' is passed, the function returns an empty list.

    :param var: Arbitrary variable
    :return: List
    """
    if isinstance(var, list):
        return var
    if var is None:
        return []
    return [var]


def vector_is_close(vec_a, vec_b, abs_tol=1E-9):
    """
    Check if a vector is close or equal to another vector.

    :param vec_a: First vector
    :param vec_b: Second vector
    :param abs_tol: Absolute tolerance
    :return: True or False
    """
    if not vec_a.size == vec_b.size:
        return False
    for i in range(vec_a.size):
        if not math.isclose(vec_a[i], vec_b[i], abs_tol=abs_tol):
            return False

    return True
