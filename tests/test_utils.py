import numpy as np

from lieink import utils


def test_skew_and_vee_roundtrip_for_single_and_batch_values():
    vector = np.array([[1.0], [-2.0], [3.5]])
    matrix = utils.skew3(vector)
    np.testing.assert_allclose(utils.veeskew3(matrix), vector)

    batch = np.array([vector, vector * 2])
    batch_matrix = utils.skew3b(batch)
    np.testing.assert_allclose(utils.veeskew3b(batch_matrix), batch)


def test_se3_and_adjoint_conversions_roundtrip():
    twist = np.array([[0.2], [-0.3], [0.4], [1.0], [2.0], [-1.0]])
    se3 = utils.skew4(twist)
    np.testing.assert_allclose(utils.veeskew4(se3), twist)

    transform = np.eye(4)
    transform[:3, :3] = utils.rot3c("y", 0.3)
    transform[:3, 3:] = twist[3:]
    adjoint = utils.SE3_matrix_to_Ad_matrix(transform)
    checked, recovered = utils.check_Ad_matrix(adjoint, return_t=True)
    np.testing.assert_allclose(checked, adjoint)
    np.testing.assert_allclose(recovered, twist[3:])


def test_batch_coadjoint_validation_accepts_valid_matrices():
    rotations = np.stack([utils.rot3c("x", 0.2), utils.rot3c("z", -0.4)])
    translations = np.array([[[1.0], [2.0], [3.0]], [[-1.0], [0.5], [2.0]]])
    transforms = np.zeros((2, 4, 4))
    transforms[:, :3, :3] = rotations
    transforms[:, :3, 3:] = translations
    transforms[:, 3, 3] = 1.0
    coadjoint = utils.SE3_matrixb_to_coAd_matrixb(transforms)

    result, recovered = utils.check_coAd_matrixb(coadjoint, return_t=True)
    np.testing.assert_allclose(result, coadjoint)
    np.testing.assert_allclose(recovered, translations)
