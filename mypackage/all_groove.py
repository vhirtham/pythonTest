"""provides the calculation of all Groove-Types."""

from astropy.units import Quantity
import numpy as np

import mypackage.geometry as geo


def grooveType(dictionary, groove_type):
    """
    Calculate a Groove type.

    :param dictionary: dictionary with the needed groove parameters
    :param groove_type: string, string corresponding to the groove type.
    """
    if groove_type == "v":
        return singleVGrooveButtWeld(**dictionary)

    if groove_type == "u":
        return singleUGrooveButtWeld(**dictionary)


def singleVGrooveButtWeld(t, alpha, b, c,
                          width_default=Quantity(2, unit="millimeter")):
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

    # Rand breite
    edge = np.min([-s, 0])
    if width <= -edge + 1:
        # zu Kleine Breite für die Naht wird angepasst
        width = width - edge

    # x-values
    x_value = []
    # y-values
    y_value = []
    segment_list = []

    # bottom segment
    x_value.append(-width)
    y_value.append(0)
    x_value.append(0)
    y_value.append(0)
    segment_list.append("line")

    # root face
    if c != 0:
        x_value.append(0)
        y_value.append(c)
        segment_list.append("line")

    # groove face
    x_value.append(-s)
    y_value.append(t)
    segment_list.append("line")

    # top segment
    x_value.append(-width)
    y_value.append(t)
    segment_list.append("line")

    shape = _helperfunction(segment_list, [x_value, y_value])

    shape = shape.translate([-b / 2, 0])
    # y Achse als Spiegelachse
    shape_r = shape.reflect_across_line([0, 0], [0, 1])

    return geo.Profile([shape, shape_r])


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

    # Rand breite
    edge = np.min([-x - s, 0])
    if width <= -edge + 1:
        # zu Kleine Breite für die Naht wird angepasst
        width = width - edge

    # x-values
    x_value = []
    # y-values
    y_value = []
    segment_list = []

    # bottom segment
    x_value.append(-width)
    y_value.append(0)
    x_value.append(0)
    y_value.append(0)
    segment_list.append("line")

    # root face
    if c != 0:
        x_value.append(0)
        y_value.append(c)
        segment_list.append("line")

    # groove face arc kreismittelpunkt
    x_value.append(0)
    y_value.append(c + R)

    # groove face arc
    x_value.append(-x)
    y_value.append(c + R - y)
    segment_list.append("arc")

    # groove face line
    x_value.append(-x - s)
    y_value.append(t)
    segment_list.append("line")

    # top segment
    x_value.append(-width)
    y_value.append(t)
    segment_list.append("line")

    shape = _helperfunction(segment_list, [x_value, y_value])

    shape = shape.translate([-b / 2, 0])
    # y Achse als Spiegelachse
    shape_r = shape.reflect_across_line([0, 0], [0, 1])

    return geo.Profile([shape, shape_r])


def _helperfunction(liste, array):
    """
    Calculate a shape from input.
    Input liste der aufeinanderfolgenden Segmente als strings.
    Input array der Punkte ich richtiger Reichenfolge. BSP:
    array = [[x-werte], [y-werte]]

    :param liste: list of String, segment names ("line", "arc")
    :param array: array of 2 array,
        first array are x-values
        second array are y-values
    :return: geo.Shape
    """
    segment_list = []
    counter = 0
    for elem in liste:
        if elem == "line":
            seg = geo.LineSegment(
                [array[0][counter: counter + 2],
                 array[1][counter: counter + 2]]
            )
            segment_list.append(seg)
            counter += 1
        if elem == "arc":
            arr0 = [
                # anfang
                array[0][counter],
                # ende
                array[0][counter + 2],
                # mittelpunkt
                array[0][counter + 1],
            ]
            arr1 = [
                # anfang
                array[1][counter],
                # ende
                array[1][counter + 2],
                # mittelpunkt
                array[1][counter + 1],
            ]
            seg = geo.ArcSegment([arr0, arr1], False)
            segment_list.append(seg)
            counter += 2

    return geo.Shape(segment_list)
