import numpy as np
import math
from scipy.spatial.transform import Rotation as Rot


def rotation_matrix_x(angle):
    """
    Create a rotation matrix that rotates around the x-axis.

    :param angle: Rotation angle
    :return: Rotation matrix
    """
    return Rot.from_euler("x", angle).as_dcm()


def rotation_matrix_y(angle):
    """
    Create a rotation matrix that rotates around the y-axis.

    :param angle: Rotation angle
    :return: Rotation matrix
    """
    return Rot.from_euler("y", angle).as_dcm()


def rotation_matrix_z(angle):
    """
    Create a rotation matrix that rotates around the z-axis.

    :param angle: Rotation angle
    :return: Rotation matrix
    """
    return Rot.from_euler("z", angle).as_dcm()


def normalize(u):
    """
    Normalize a vector.

    :param u: Vector
    :return: Normalized vector
    """
    norm = np.linalg.norm(u)
    if norm == 0.:
        raise Exception("Vector length is 0.")
    return u / norm


def orientation_point_plane_containing_origin(point, a, b):
    """
    Determine a points orientation relative to a plane containing the origin.

    The side is defined by the winding order of the triangle 'origin - A -
    B'. When looking at it from the left-hand side, the ordering is clockwise
    and counter-clockwise when looking from the right-hand side.

    The function returns 1 if the point lies left of the plane, -1 if it is
    on the right and 0 if it lies on the plane.

    Note, that this function is not appropriate to check if a point lies on
    a plane since it has no tolerance to compensate for numerical errors.

    Additional note: The points A and B can also been considered as two
    vectors spanning the plane.

    :param point: Point
    :param a: Second point of the triangle 'origin - A - B'.
    :param b: Third point of the triangle 'origin - A - B'.
    :return: 1, -1 or 0 (see description)
    """
    if (math.isclose(np.linalg.norm(a), 0) or
            math.isclose(np.linalg.norm(b), 0) or
            math.isclose(np.linalg.norm(b - a), 0)):
        raise Exception(
            "One or more points describing the plane are identical.")

    return np.sign(np.linalg.det([a, b, point]))


def point_left_of_plane_by_points(point, a, b, c):
    """
    Determine a points orientation relative to a plane.

    The side is defined by the winding order of the triangle 'A -B - C'.
    When looking at it from the left-hand side, the ordering is clockwise
    and counter-clockwise when looking from the right-hand side.

    The function returns 1 if the point lies left of the plane, -1 if it is
    on the right and 0 if it lies on the plane.

    Note, that this function is not appropriate to check if a point lies on
    a plane since it has no tolerance to compensate for numerical errors.

    :param a: First point of the triangle 'A - B - C'.
    :param b: Second point of the triangle 'A - B - C'.
    :param b: Third point of the triangle 'A - B - C'.
    :return: 1, -1 or 0 (see description)
    """
    vec_a_b = b - a
    vec_a_c = c - a
    vec_a_point = point - a
    return orientation_point_plane_containing_origin(vec_a_b, vec_a_c,
                                                     vec_a_point)


def is_orthogonal(u, v, tolerance=1E-9):
    return np.abs(np.dot(normalize(u), normalize(v))) < tolerance


def change_of_base_rotation(css_from, css_to):
    return np.linalg.solve(css_from.basis, css_to.basis)


def change_of_base_translation(css_from, css_to):
    return css_from.origin - css_to.origin


class CartesianCoordinateSystem3d:
    """Defines a 3d cartesian coordinate system."""

    def __init__(self, basis=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                 origin=np.array([0, 0, 0])):
        """
        Construct a cartesian coordinate system.

        :param basis: List of basis vectors
        :param origin: Position of the origin
        :return: Cartesian coordinate system
        """
        basis[0] = normalize(basis[0])
        basis[1] = normalize(basis[1])
        basis[2] = normalize(basis[2])

        if not (is_orthogonal(basis[0], basis[1]) and
                is_orthogonal(basis[1], basis[2]) and
                is_orthogonal(basis[2], basis[0])):
            raise Exception("Basis vectors must be orthogonal")

        self._basis = basis

        self._origin = np.array(origin)

    @classmethod
    def construct_from_basis(cls, basis, origin=np.array([0, 0, 0])):
        """
        Construct a cartesian coordinate system.

        :param basis: List of basis vectors
        :param origin: Position of the origin
        :return: Cartesian coordinate system
        """
        return cls(basis, origin=origin)

    @classmethod
    def construct_from_xyz(cls, x, y, z, origin=np.array([0, 0, 0])):
        """
        Construct a cartesian coordinate system.

        :param x: Vector defining the x-axis
        :param y: Vector defining the y-axis
        :param z: Vector defining the z-axis
        :param origin: Position of the origin
        :return: Cartesian coordinate system
        """
        basis = [x, y, z]
        return cls(basis, origin=origin)

    @classmethod
    def construct_from_xy_and_orientation(cls, x, y, positive_orientation=True,
                                          origin=np.array([0, 0, 0])):
        """
        Construct a cartesian coordinate system.

        :param x: Vector defining the x-axis
        :param y: Vector defining the y-axis
        :param positive_orientation: Set to True if the orientation should
        be positive and to False if not
        :param origin: Position of the origin
        :return: Cartesian coordinate system
        """
        z = cls._calcualte_orthogonal_axis(x, y) * cls._sign_orientation(
            positive_orientation)
        basis = [x, y, z]
        return cls(basis, origin=origin)

    @classmethod
    def construct_from_yz_and_orientation(cls, y, z, positive_orientation=True,
                                          origin=np.array([0, 0, 0])):
        """
        Construct a cartesian coordinate system.

        :param y: Vector defining the y-axis
        :param z: Vector defining the z-axis
        :param positive_orientation: Set to True if the orientation should
        be positive and to False if not
        :param origin: Position of the origin
        :return: Cartesian coordinate system
        """
        x = cls._calcualte_orthogonal_axis(y, z) * cls._sign_orientation(
            positive_orientation)
        basis = [x, y, z]
        return cls(basis, origin=origin)

    @classmethod
    def construct_from_xz_and_orientation(cls, x, z, positive_orientation=True,
                                          origin=np.array([0, 0, 0])):
        """
        Construct a cartesian coordinate system.

        :param x: Vector defining the x-axis
        :param z: Vector defining the z-axis
        :param positive_orientation: Set to True if the orientation should
        be positive and to False if not
        :param origin: Position of the origin
        :return: Cartesian coordinate system
        """
        y = cls._calcualte_orthogonal_axis(z, x) * cls._sign_orientation(
            positive_orientation)
        basis = [x, y, z]
        return cls(basis, origin=origin)

    @staticmethod
    def _sign_orientation(positive_orientation):
        """
        Get -1 or 1 depending on the coordinate systems orientation.

        :param positive_orientation: Set to True if the orientation should
        be positive and to False if not
        :return: 1 if the coordinate system has positive orientation,
        -1 otherwise
        """
        if positive_orientation:
            return 1
        return -1

    @staticmethod
    def _calcualte_orthogonal_axis(a0, a1):
        """
        Calculate an axis which is orthogonal to two other axes.

        The calculated axis has a positive orientation towards the other 2
        axes.

        :param a0: First axis
        :param a1: Second axis
        :return: Orthogonal axis
        """
        return np.cross(a0, a1)

    @property
    def basis(self):
        """
        Get the coordinate systems basis.

        :return: Basis of the coordinate system
        """
        return self._basis

    @property
    def origin(self):
        """
        Get the coordinate systems origin.

        :return: Origin of the coordinate system
        """
        return self._origin

# def vector_to_vector_transformation(u, v):
#    r = np.cross(u, v)
#    w = np.sqrt(np.dot(u, u) * np.dot(v, v)) + np.dot(u, v)
#    quaternion = np.concatenate((r, [w]))
#    unit_quaternion = quaternion / np.linalg.norm(quaternion)

#    return R.from_quat(unit_quaternion).as_dcm()
