import mypackage.my_funcs as mf
import pytest


def test_print_hello_world():
    print('Hello world')


def test_my_func():
    mf.my_func(True)
    mf.my_func(False)


def test_add_numbers():
    assert mf.addNumbers(1, 2) == 3
    with pytest.raises(TypeError):
        mf.addNumbers("1", 3)
    with pytest.raises(TypeError):
        mf.addNumbers(1, "3")
