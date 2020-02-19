"""provides the calculation of all Groove-Types."""

from astropy.units import Quantity
import numpy as np

import mypackage.geometry as geo


def singleVGrooveButtWeld(
    t, alpha, b, c, width_default=Quantity(2, unit="millimeter")
):
    """
    Calculate a Single-V Groove Butt Weld.

    :param t: the workpiece thickness, as Astropy unit
    :param alpha: the groove angle, as Astropy unit
    :param b: the root opening, as Astropy unit
    :param c: the root face, as Astropy unit
    :param width_default: the width of the workpiece, as Astropy unit
    :return: geo.Profile
    """
    t = t.to_value("millimeter")
    alpha = alpha.to_value("rad")
    b = b.to_value("millimeter")
    c = c.to_value("millimeter")
    width = width_default.to_value("millimeter")

    # calculations:
    s = np.tan(alpha / 2) * (t - c)

    # Rand breiten Berechnung
    edge = np.min([-s, 0])
    if width <= -edge + 1:
        # zu Kleine Breite für die Naht wird angepasst
        width = width - edge

    segment_list = []

    bottom = geo.LineSegment([[-width, 0], [0, 0]])
    segment_list.append(bottom)

    if c != 0:
        root_face = geo.LineSegment([[0, 0], [0, c]])
        segment_list.append(root_face)

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
    t, beta, R, b, c, width_default=Quantity(3, unit="millimeter")
):
    """
    Calculate a Single-U Groove Butt Weld.

    :param t: the workpiece thickness, as Astropy unit
    :param beta: the bevel angle, as Astropy unit
    :param R: radius, as Astropy unit
    :param b: the root opening, as Astropy unit
    :param c: the root face, as Astropy unit
    :param width_default: the width of the workpiece, as Astropy unit
    :return: geo.Profile
    """
    t = t.to_value("millimeter")
    beta = beta.to_value("rad")
    R = R.to_value("millimeter")
    b = b.to_value("millimeter")
    c = c.to_value("millimeter")
    width = width_default.to_value("millimeter")

    # calculations:
    # vom nächsten Punkt zum Kreismittelpunkt ist der Vektor (x,y)
    x = R * np.cos(beta)
    y = R * np.sin(beta)
    # m = [0,c+R] Kreismittelpunkt
    # => [-x,c+R-y] ist der nächste Punkt

    s = np.tan(beta) * (t - (c + R - y))

    # Rand breiten Berechnung
    edge = np.min([-x - s, 0])
    if width <= -edge + 1:
        # zu Kleine Breite für die Naht wird angepasst
        width = width - edge

    segment_list = []

    bottom = geo.LineSegment([[-width, 0], [0, 0]])
    segment_list.append(bottom)

    if c != 0:
        root_face = geo.LineSegment([[0, 0], [0, c]])
        segment_list.append(root_face)

    groove_face_arc = geo.ArcSegment([[0, -x, 0], [c, c + R - y, c + R]],
                                     False)
    segment_list.append(groove_face_arc)

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
