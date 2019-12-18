"""Test Functions by Markus."""
import xarray as xr
import numpy as np


def function_one():
    """
    Print "Hello World".

    :param: ---
    :return: ---
    """
    print("Hello World")


def function_two():
    """
     Working with Xarrays.

    :param: ---
    :return: ---
    """
    xcoords = np.linspace(0, 1, 2)
    ycoords = np.linspace(0, 1, 2)
    a = xr.DataArray(
        [[True, False], [False, True]],
        name=["xcoords", "ycoords"],
        dims=["x", "y"],
        coords=dict(x=xcoords, y=ycoords),
    )
    try:
        if a.sel(x=1, y=1, method="nearest", tolerance=0):
            raise KeyError
        else:
            a.loc[dict(x=1, y=1)] = True
    except KeyError:
        print("Punkt schon vorhanden!")
