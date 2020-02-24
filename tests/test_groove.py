from mypackage.all_groove import singleVGrooveButtWeld, singleUGrooveButtWeld

from astropy.units import Quantity
from math import isclose


def test_v_groove():
    t = Quantity(6, unit="millimeter")
    alpha = Quantity(73.73979529168803, unit="deg")
    b = Quantity(2, unit="millimeter")
    c = Quantity(2, unit="millimeter")
    width = Quantity(6, unit="millimeter")
    v_naht_dict = dict(t=t, alpha=alpha, b=b, c=c, width_default=width)
    profile = singleVGrooveButtWeld(**v_naht_dict)
    data = profile.rasterize(10)
    test_data = [[-7, -1, -1, -4, -7, 7, 1, 1, 4, 7],
                 [0, 0, 2, 6, 6, 0, 0, 2, 6, 6]]
    for i in range((len(test_data[0]))):
        assert isclose(data[0][i], test_data[0][i])
        assert isclose(data[1][i], test_data[1][i])

def test_u_groove():
    t = Quantity(7, unit="millimeter")
    beta = Quantity(9, unit="deg")
    R = Quantity(2, unit="millimeter")
    b = Quantity(2, unit="millimeter")
    c = Quantity(2, unit="millimeter")
    width = Quantity(6, unit="millimeter")
    u_naht_dict = dict(t=t, beta=beta, R=R, b=b, c=c, width_default=width)
    profile = singleUGrooveButtWeld(**u_naht_dict)
    data = profile.rasterize(10)
    test_data = [[-7, -1, -1, -2.97537668, -3.50008357, -7, 7, 1, 1,
                  2.97537668, 3.50008357, 7],
                 [0, 0, 2, 3.68713107, 7, 7, 0, 0, 2, 3.68713107, 7, 7]]
    for i in range((len(test_data[0]))):
        assert isclose(data[0][i], test_data[0][i])
        assert isclose(data[1][i], test_data[1][i])