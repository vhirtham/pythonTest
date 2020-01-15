import mypackage.transformations as tf
import numpy as np
import pytest


# cartesian coordinate system class -------------------------------------------

def is_orientation_positive(ccs):
    return tf.point_left_of_plane_by_vectors(ccs.basis[2], ccs.basis[0],
                                             ccs.basis[1]) > 0


def check_coordinate_system(ccs, basis_expected, origin_expected,
                            positive_orientation_expected):
    # check orientation is as expected
    assert is_orientation_positive(ccs) == positive_orientation_expected

    # check basis vectors are orthogonal
    assert tf.is_orthogonal(ccs.basis[0], ccs.basis[1])
    assert tf.is_orthogonal(ccs.basis[1], ccs.basis[2])
    assert tf.is_orthogonal(ccs.basis[2], ccs.basis[0])

    for i in range(3):
        unit_vec = tf.normalize(basis_expected[i])

        # check axis orientations match
        assert np.abs(np.dot(ccs.basis[i], unit_vec) - 1) < 1E-9

        # check origin correct
        assert np.abs(origin_expected[i] - ccs.origin[i]) < 1E-9


def test_cartesian_coordinate_system_construction():
    # alias name for class - name is too long :)
    cls_ccs = tf.CartesianCoordinateSystem3d

    # setup -----------------------------------------------
    origin = [4, -2, 6]
    x = [1, 0, 0]
    y = [0, 1, 0]
    z = [0, 0, 1]

    # rotate axes to produce a more general test case
    angle_x = np.pi / 3
    angle_y = np.pi / 4
    angle_z = np.pi / 5
    r_x = tf.rotation_matrix_x(angle_x)
    r_y = tf.rotation_matrix_y(angle_y)
    r_z = tf.rotation_matrix_z(angle_z)

    r_tot = np.matmul(r_z, np.matmul(r_y, r_x))

    x = np.matmul(r_tot, x)
    y = np.matmul(r_tot, y)
    z = np.matmul(r_tot, z)

    basis_pos = [x, y, z]
    basis_neg = [x, y, -z]

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

    # check exceptions ------------------------------------
    with pytest.raises(Exception):
        cls_ccs([x, y, [0, 0, 1]])
