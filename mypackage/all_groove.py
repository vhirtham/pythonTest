"""provides the calculation of all Groove-Types"""

from astropy.units import Quantity
import numpy as np

import mypackage.geometry as geo


def singleVGrooveButtWeld(
    t, alpha, b, c, width_default=Quantity(5, unit="millimeter")
):
    """
    The calculation of a Single-V Groove Butt Weld.
    Required variables are in Quantity(astropy):

    :param t: the workpiece thickness
    :param alpha: the groove angle
    :param b: the root opening
    :param c: the root face
    :param width_default: the width of the workpiece

    :return: point_could_generator.Profile
    """
    t = t.to_value("millimeter")
    alpha = alpha.to_value("rad")
    b = b.to_value("millimeter")
    c = c.to_value("millimeter")
    width = width_default.to_value("millimeter")

    segment_list = []

    bottom = geo.LineSegment([[-width, 0], [0, 0]])
    segment_list.append(bottom)

    if c != 0:
        root_face = geo.LineSegment([[0, 0], [0, c]])
        segment_list.append(root_face)

    s = np.tan(alpha / 2) * (t - c)
    groove_face = geo.LineSegment([[0, -s], [c, t]])
    segment_list.append(groove_face)

    top = geo.LineSegment([[-s, -width], [t, t]])
    segment_list.append(top)

    shape = geo.Shape(segment_list)

    shape = shape.translate([-b / 2, 0])
    # y Achse als Spiegelachse
    shape_r = shape.reflect_across_line([0, 0], [0, 1])

    profile = geo.Profile([shape, shape_r])

    return profile


def singleUGrooveButtWeld(
    t, beta, R, b, c, width_default=Quantity(15, unit="millimeter")
):
    """
    The calculation of a Single-U Groove Butt Weld.
    Required variables are in Quantity(astropy):

    :param t: the workpiece thickness
    :param beta: the bevel angle
    :param R: radius
    :param b: the root opening
    :param c: the root face
    :param width_default: the width of the workpiece

    :return: point_could_generator.Profile
    """

    t = t.to_value("millimeter")
    beta = beta.to_value("rad")
    R = R.to_value("millimeter")
    b = b.to_value("millimeter")
    c = c.to_value("millimeter")
    width = width_default.to_value("millimeter")

    segment_list = []

    bottom = geo.LineSegment([[-width, 0], [0, 0]])
    segment_list.append(bottom)

    if c != 0:
        root_face = geo.LineSegment([[0, 0], [0, c]])
        segment_list.append(root_face)

    # vom nächsten Punkt zum Kreismittelpunkt ist der Vektor (x,y)
    x = R * np.cos(beta)
    y = R * np.sin(beta)
    # m = [0,c+R] Kreismittelpunkt
    # => [-x,c+R-y] ist der nächste Punkt
    groove_face_arc = geo.ArcSegment([[0, -x, 0], [c, c + R - y, c + R]],
                                     False)
    segment_list.append(groove_face_arc)

    s = np.tan(beta) * (t - (c + R - y))
    groove_face_line = geo.LineSegment([[-x, -x - s], [c + R - y, t]])
    segment_list.append(groove_face_line)

    top = geo.LineSegment([[-x - s, -width], [t, t]])
    segment_list.append(top)

    shape = geo.Shape(segment_list)

    shape = shape.translate([-b / 2, 0])
    # y Achse als Spiegelachse
    shape_r = shape.reflect_across_line([0, 0], [0, 1])

    profile = geo.Profile([shape, shape_r])

    return profile
