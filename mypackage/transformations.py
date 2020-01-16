import numpy as np
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
    norm = np.linalg.norm(u)
    if norm == 0.:
        raise Exception("Vector length is 0.")
    return u / norm


def point_left_of_plane_by_vectors(point, plane_vec0, plane_vec1):
    # right is the direction of the cross product
    print([plane_vec0, plane_vec1, point])
    print(np.sign(np.linalg.det([plane_vec0, plane_vec1, point])))
    return np.sign(np.linalg.det([plane_vec0, plane_vec1, point]))


def point_left_of_plane_by_points(point, plane_a, plane_b, plane_c):
    vec_a_b = plane_b - plane_a
    vec_a_c = plane_c - plane_a
    vec_a_point = point - plane_a
    return point_left_of_plane_by_vectors(vec_a_b, vec_a_c, vec_a_point)


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
