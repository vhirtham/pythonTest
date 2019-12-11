"""
Contains some test functions
"""


def my_func(bool):
    """
    Prints a message

    :param bool: Selects one of two possible messages
    :return: ---
    """
    if bool:
        print("The answer is 42")
        print("Tvis is bat inglish")
    else:
        print("This branch is not covered")


def add_numbers(a, b):
    """
    Adds 2 numbers

    :param a: First number
    :param b: Second number
    :return: Result of the addition
    """
    if not (isinstance(a, int) or isinstance(a, float)):
        raise TypeError("First argument must be a integer or float")
    if not (isinstance(b, int) or isinstance(b, float)):
        raise TypeError("Second argument must be a integer or float")
    return a + b
