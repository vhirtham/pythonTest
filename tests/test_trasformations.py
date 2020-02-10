import mypackage.transformations as tf
import numpy as np
import pytest
import random
import math
import copy
import tests._helpers as helper


# helpers for tests -----------------------------------------------------------

def check_coordinate_system(ccs, basis_expected, origin_expected,
                            positive_orientation_expected):
    # check orientation is as expected
    assert is_orientation_positive(ccs) == positive_orientation_expected

    # check basis vectors are orthogonal
    assert tf.is_orthogonal(ccs.basis[0], ccs.basis[1])
    assert tf.is_orthogonal(ccs.basis[1], ccs.basis[2])
    assert tf.is_orthogonal(ccs.basis[2], ccs.basis[0])

    for i in range(3):
        unit_vec = tf.normalize(basis_expected[:, i])

        # check axis orientations match
        assert np.abs(np.dot(ccs.basis[:, i], unit_vec) - 1) < 1E-9

        # check origin correct
        assert np.abs(origin_expected[i] - ccs.origin[i]) < 1E-9


def check_matrix_does_not_reflect(matrix):
    assert np.linalg.det(matrix) >= 0


def check_matrix_orthogonal(matrix):
    transposed = np.transpose(matrix)

    product = np.matmul(transposed, matrix)
    unit = np.identity(3)
    for i in range(3):
        for j in range(3):
            assert np.abs(product[i][j] - unit[i][j]) < 1E-9


def check_matrix_identical(a, b):
    for i in range(3):
        for j in range(3):
            assert math.isclose(a[i, j], b[i, j], abs_tol=1E-9)


def is_orientation_positive(ccs):
    return tf.orientation_point_plane_containing_origin(ccs.basis[2],
                                                        ccs.basis[0],
                                                        ccs.basis[1]) > 0


def random_non_unit_vector():
    vec = np.array([random.random(), random.random(),
                    random.random()]) * 10 * random.random()
    while math.isclose(np.linalg.norm(vec), 1) or math.isclose(
            np.linalg.norm(vec), 0):
        vec = np.array([random.random(), random.random(),
                        random.random()]) * 10 * random.random()
    return vec


def rotated_positive_orthogonal_base(angle_x=np.pi / 3, angle_y=np.pi / 4,
                                     angle_z=np.pi / 5):
    x = [1, 0, 0]
    y = [0, 1, 0]
    z = [0, 0, 1]

    # rotate axes to produce a more general test case
    r_x = tf.rotation_matrix_x(angle_x)
    r_y = tf.rotation_matrix_y(angle_y)
    r_z = tf.rotation_matrix_z(angle_z)

    r_tot = np.matmul(r_z, np.matmul(r_y, r_x))

    x = np.matmul(r_tot, x)
    y = np.matmul(r_tot, y)
    z = np.matmul(r_tot, z)

    return np.transpose([x, y, z])


# test functions --------------------------------------------------------------

def test_single_axis_rotation_matrices():
    matrix_funcs = [tf.rotation_matrix_x, tf.rotation_matrix_y,
                    tf.rotation_matrix_z]
    vec = np.array([1, 1, 1])

    for i in range(3):
        for j in range(36):
            angle = j / 18 * np.pi
            matrix = matrix_funcs[i](angle)

            # rotation matrices are orthogonal
            check_matrix_orthogonal(matrix)

            # matrix should not reflect
            check_matrix_does_not_reflect(matrix)

            # rotate vector
            res = np.matmul(matrix, vec)

            # check component of rotation axis
            assert np.abs(res[i] - 1) < 1E-9

            # check other components
            i_1 = (i + 1) % 3
            i_2 = (i + 2) % 3

            exp_1 = np.cos(angle) - np.sin(angle)
            exp_2 = np.cos(angle) + np.sin(angle)

            assert np.abs(res[i_1] - exp_1) < 1E-9
            assert np.abs(res[i_2] - exp_2) < 1E-9


def test_normalize():
    for i in range(20):
        vec = random_non_unit_vector()

        unit = tf.normalize(vec)

        # check that vector is modified
        for i in range(vec.size):
            assert not math.isclose(unit[i], vec[i])

        # check length is 1
        assert math.isclose(np.linalg.norm(unit), 1)

        # check that both vectors point into the same direction
        vec2 = unit * np.linalg.norm(vec)
        for i in range(vec.size):
            assert math.isclose(vec2[i], vec[i])

    # check exception if length is 0
    with pytest.raises(Exception):
        tf.normalize(np.array([0, 0, 0]))


def test_orientation_point_plane_containing_origin():
    [a, b, n] = rotated_positive_orthogonal_base()
    a *= 2.3
    b /= 1.5

    for length in np.arange(-9.5, 9.51, 1):
        orientation = tf.orientation_point_plane_containing_origin(n * length,
                                                                   a, b)
        assert np.sign(length) == orientation

    # check exceptions
    with pytest.raises(Exception):
        tf.orientation_point_plane_containing_origin(n, a, a)
    with pytest.raises(Exception):
        tf.orientation_point_plane_containing_origin(n, np.zeros(3), b)
    with pytest.raises(Exception):
        tf.orientation_point_plane_containing_origin(n, a, np.zeros(3))

    # check special case point on plane
    a = np.array([1, 0, 0])
    b = np.array([0, 1, 0])
    orientation = tf.orientation_point_plane_containing_origin(a, a, b)
    assert orientation == 0


def test_orientation_point_plane():
    [b, c, n] = rotated_positive_orthogonal_base()
    a = [3.2, -2.1, 5.4]
    b = b * 6.5 + a
    c = c * 0.3 + a

    for length in np.arange(-9.5, 9.51, 1):
        orientation = tf.orientation_point_plane(n * length + a, a, b, c)
        assert np.sign(length) == orientation

    # check exceptions
    with pytest.raises(Exception):
        tf.orientation_point_plane(n, a, a, c)
    with pytest.raises(Exception):
        tf.orientation_point_plane(n, a, b, b)
    with pytest.raises(Exception):
        tf.orientation_point_plane(n, c, b, c)
    with pytest.raises(Exception):
        tf.orientation_point_plane(n, a, a, a)

    # check special case point on plane
    a = np.array([1, 0, 0])
    b = np.array([0, 1, 0])
    c = np.array([0, 0, 1])
    orientation = tf.orientation_point_plane(a, a, b, c)
    assert orientation == 0


def test_is_orthogonal():
    basis = rotated_positive_orthogonal_base()
    x = basis[:, 0]
    y = basis[:, 1]
    z = basis[:, 2]

    assert tf.is_orthogonal(x, y)
    assert tf.is_orthogonal(y, x)
    assert tf.is_orthogonal(y, z)
    assert tf.is_orthogonal(z, y)
    assert tf.is_orthogonal(z, x)
    assert tf.is_orthogonal(x, z)

    assert not tf.is_orthogonal(x, x)
    assert not tf.is_orthogonal(y, y)
    assert not tf.is_orthogonal(z, z)

    # check tolerance is working
    assert not tf.is_orthogonal(x + 0.00001, z, 1E-6)
    assert tf.is_orthogonal(x + 0.00001, z, 1E-4)

    # check zero length vectors cause exception
    with pytest.raises(Exception):
        tf.is_orthogonal([0, 0, 0], z)
    with pytest.raises(Exception):
        tf.is_orthogonal(x, [0, 0, 0])
    with pytest.raises(Exception):
        tf.is_orthogonal([0, 0, 0], [0, 0, 0])


def test_change_of_basis_rotation():
    diff_angle = np.pi / 2
    ref_mat = [tf.rotation_matrix_x(-diff_angle),
               tf.rotation_matrix_y(-diff_angle),
               tf.rotation_matrix_z(-diff_angle)]

    for i in range(3):
        angles_from = np.pi * np.array([1 / 3., 1 / 5., 1 / 4])
        for j in np.arange(i + 1, 3):
            angles_from[j] = 0

        angles_to = copy.deepcopy(angles_from)
        angles_to[i] += diff_angle

        base_from = rotated_positive_orthogonal_base(*angles_from)
        base_to = rotated_positive_orthogonal_base(*angles_to)

        ccs_from = tf.CartesianCoordinateSystem3d(base_from,
                                                  random_non_unit_vector())
        ccs_to = tf.CartesianCoordinateSystem3d(base_to,
                                                random_non_unit_vector())

        matrix = tf.change_of_basis_rotation(ccs_from, ccs_to)

        check_matrix_identical(matrix, ref_mat[i])


def test_change_of_basis_translation():
    for i in range(20):
        origin_from = random_non_unit_vector()
        origin_to = random_non_unit_vector()
        base_from = rotated_positive_orthogonal_base(*random_non_unit_vector())
        base_to = rotated_positive_orthogonal_base(*random_non_unit_vector())

        ccs_from = tf.CartesianCoordinateSystem3d(base_from, origin_from)
        ccs_to = tf.CartesianCoordinateSystem3d(base_to, origin_to)

        diff = tf.change_of_basis_translation(ccs_from, ccs_to)

        expected_diff = origin_from - origin_to
        for j in range(3):
            assert math.isclose(diff[j], expected_diff[j])


# test cartesian coordinate system class --------------------------------------

def test_cartesian_coordinate_system_construction():
    # alias name for class - name is too long :)
    cls_ccs = tf.CartesianCoordinateSystem3d

    # setup -----------------------------------------------
    origin = [4, -2, 6]
    basis_pos = rotated_positive_orthogonal_base()

    x = basis_pos[:, 0]
    y = basis_pos[:, 1]
    z = basis_pos[:, 2]

    basis_neg = np.transpose([x, y, -z])

    # construction with basis -----------------------------

    ccs_basis_pos = cls_ccs.construct_from_basis(basis_pos, origin)
    ccs_basis_neg = cls_ccs.construct_from_basis(basis_neg, origin)

    check_coordinate_system(ccs_basis_pos, basis_pos, origin, True)
    check_coordinate_system(ccs_basis_neg, basis_neg, origin, False)

    # construction with x,y,z-vectors ---------------------

    ccs_xyz_pos = cls_ccs.construct_from_xyz(x, y, z, origin)
    ccs_xyz_neg = cls_ccs.construct_from_xyz(x, y, -z, origin)

    check_coordinate_system(ccs_xyz_pos, basis_pos, origin, True)
    check_coordinate_system(ccs_xyz_neg, basis_neg, origin, False)

    # construction with x,y-vectors and orientation -------
    ccs_xyo_pos = cls_ccs.construct_from_xy_and_orientation(x, y, True, origin)
    ccs_xyo_neg = cls_ccs.construct_from_xy_and_orientation(x, y, False,
                                                            origin)

    check_coordinate_system(ccs_xyo_pos, basis_pos, origin, True)
    check_coordinate_system(ccs_xyo_neg, basis_neg, origin, False)

    # construction with y,z-vectors and orientation -------
    ccs_yzo_pos = cls_ccs.construct_from_yz_and_orientation(y, z, True, origin)
    ccs_yzo_neg = cls_ccs.construct_from_yz_and_orientation(y, -z, False,
                                                            origin)

    check_coordinate_system(ccs_yzo_pos, basis_pos, origin, True)
    check_coordinate_system(ccs_yzo_neg, basis_neg, origin, False)

    # construction with x,z-vectors and orientation -------
    ccs_xzo_pos = cls_ccs.construct_from_xz_and_orientation(x, z, True, origin)
    ccs_xzo_neg = cls_ccs.construct_from_xz_and_orientation(x, -z, False,
                                                            origin)

    check_coordinate_system(ccs_xzo_pos, basis_pos, origin, True)
    check_coordinate_system(ccs_xzo_neg, basis_neg, origin, False)

    # test integers as inputs -----------------------------
    x_i = [1, 1, 0]
    y_i = [-1, 1, 0]
    z_i = [0, 0, 1]

    cls_ccs.construct_from_xyz(x_i, y_i, z_i, origin)
    cls_ccs.construct_from_xy_and_orientation(x_i, y_i)
    cls_ccs.construct_from_yz_and_orientation(y_i, z_i)
    cls_ccs.construct_from_xz_and_orientation(z_i, x_i)

    # check exceptions ------------------------------------
    with pytest.raises(Exception):
        cls_ccs([x, y, [0, 0, 1]])


def test_cartesian_coordinate_system_addition():
    cls_ccs = tf.CartesianCoordinateSystem3d

    orientation0 = tf.rotation_matrix_z(np.pi / 2)
    origin0 = [1, 3, 2]
    ccs0 = cls_ccs(orientation0, origin0)

    orientation1 = tf.rotation_matrix_y(np.pi / 2)
    origin1 = [4, -2, 1]
    ccs1 = cls_ccs(orientation1, origin1)

    orientation2 = tf.rotation_matrix_x(np.pi / 2)
    origin2 = [-3, 4, 2]
    ccs2 = cls_ccs(orientation2, origin2)

    # check i
    ccs_tot_0 = ccs0 + (ccs1 + ccs2)
    ccs_tot_1 = (ccs0 + ccs1) + ccs2
    helper.check_matrices_identical(ccs_tot_0.basis, ccs_tot_1.basis)
    helper.check_vectors_identical(ccs_tot_0.origin, ccs_tot_1.origin)

    expected_origin = np.array([-1, 9, 6])
    expected_orientation = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])

    helper.check_matrices_identical(ccs_tot_0.basis, expected_orientation)
    helper.check_vectors_identical(ccs_tot_0.origin, expected_origin)
