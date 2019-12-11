import os
import sys
import pytest



filePath = os.path.dirname(os.path.abspath(__file__))
sys.path.append(filePath + '\\..\\MyPackages')
sys.path.append(filePath + '/../MyPackages')

import my_funcs as mf




def test_PrintHelloWorld():
    print('Hello world')

def test_MyFunc():
    mf.my_func(True)

def testAddNumbers():
    assert mf.add_numbers(1,2) == 3
    with pytest.raises(TypeError):
        mf.add_numbers("1",2)

