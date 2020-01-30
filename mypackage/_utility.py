"""Contains package internal utility functions."""


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
