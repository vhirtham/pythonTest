"""
Test Functions by Markus.
"""
import xarray as xr
import numpy as np


def function_one():
    """
    Print "Hello World"!
    
    :param: ---
    :return: ---
    """
    print("Hello World")

def function_two():
    """
    Some stuff with Xarrays
    
    :param: ---
    :return: ---
    """
    xcoords=np.linspace(0,1,2)
    ycoords=np.linspace(0,1,2)
    a = xr.DataArray(
        [[True, False],[False,True]],
        name=["xcoords", "ycoords"],
        dims=["x","y"],
        coords=dict(x=xcoords, y=ycoords),
    )
    a.sel(x=1, y=1, method="nearest", tolerance=0)