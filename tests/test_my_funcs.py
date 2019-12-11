import os
import sys
import pytest

filePath = os.path.dirname(os.path.abspath(__file__))
sys.path.append(filePath + '\\..\\MyPackages')
sys.path.append(filePath + '/../MyPackages')


from my_funcs import *


def test_PrintHelloWorld():
    print('Hello world')

def test_MyFunc():
    my_func(True)

def testAddNumbers():
    assert add_numbers(1,2) == 3
    with pytest.raises(TypeError):
        add_numbers("1",2)

