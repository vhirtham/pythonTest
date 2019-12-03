import os
import sys


filePath = os.path.dirname(os.path.abspath(__file__))
sys.path.append(filePath + '\\..\\MyPackages')
sys.path.append(filePath + '/../MyPackages')


from MyFuncs import *


def test_PrintHelloWorld():
    print('Hello world')

def test_MyFunc():
    MyFunc(True)

