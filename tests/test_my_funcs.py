# import mypackage.my_funcs as mf
from mypackage.my_funcs import *
import pytest


def test_print_hello_world():
    print('Hello world')


def test_my_func():
    my_func(True)
    my_func(False)


def test_add_numbers():
    assert add_numbers(1, 2) == 3
    with pytest.raises(TypeError):
        add_numbers("1", 3)
    with pytest.raises(TypeError):
        add_numbers(1, "3")
