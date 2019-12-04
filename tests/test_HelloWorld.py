import os
import sys
import pytest

filePath = os.path.dirname(os.path.abspath(__file__))
sys.path.append(filePath + '\\..\\MyPackages')
sys.path.append(filePath + '/../MyPackages')


from MyFuncs import *


def test_PrintHelloWorld():
    print('Hello world')

def test_MyFunc():
    MyFunc(True)
    with pytest.raises(TypeError):
        MyFunc(True)

def testAddNumbers():
    assert AddNumbers(1,2) == 3
    with pytest.raises(TypeError):
        AddNumbers("1",2)

