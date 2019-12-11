"""Contains some test functions"""


def my_func(sel):
    """
    Prints a message

    :param sel: Selects one of two possible messages
    :return: ---
    """
    if bool:
        print("The answer is 42")
        print("Tvis is bat inglish")
    else:
        print("This branch is not covered")


def add_numbers(in0, in1):
    """
    Adds 2 numbers

    :param in0: First number
    :param in1: Second number
    :return: Result of the addition
    """
    if not isinstance(in0, (int, float)):
        raise TypeError("First argument must be a integer or float")
    if not isinstance(in1, (int, float)):
        raise TypeError("Second argument must be a integer or float")
    return in0 + in1
