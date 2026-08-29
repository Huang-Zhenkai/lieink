from typing import Any, Literal, cast

import numpy as np
from beartype import beartype
from rich import print

from lieink.annotations import (
    NDArray,
    NDArray_1D,
    NDArray_2D,
    NDArray_3,
    NDArray_3_1,
    NDArray_3_3,
    NDArray_3_N,
    NDArray_3D,
    NDArray_4_4,
    NDArray_4D,
    NDArray_6,
    NDArray_6_1,
    NDArray_6_6,
    NDArray_6_N,
    NDArray_N_3_1,
    NDArray_N_3_3,
    NDArray_N_4_4,
    NDArray_N_6_1,
    NDArray_N_6_6,
    NDArrayBool,
    RealScalar,
    get_NDArrayAnnotation,
    get_NDArrayAnnotationShape,
    is_NDArrayAnnotation,
)

ATOL = 1e-7  # 绝对精度
DELTA = np.array(
    [
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
    ]
)


@beartype
def check_equal(
    x: NDArray | RealScalar, y: NDArray | RealScalar, tol: RealScalar = ATOL
) -> NDArray | RealScalar | Literal[False]:

    if np.allclose(x, y, atol=float(tol)):
        return x
    else:
        return False


@beartype
def check_symmetry(x: NDArray, tol: RealScalar = ATOL) -> NDArray | Literal[False]:
    if np.allclose(x, x.T, atol=float(tol)):
        return x
    else:
        return False


@beartype
def check_symmetryb(
    x: NDArray_3D, tol: RealScalar = ATOL
) -> NDArray_3D | Literal[False]:
    if np.allclose(x, x.transpose(0, 2, 1), atol=float(tol)):
        return x
    else:
        return False


@beartype
def find_zeros(x: NDArray, tol: RealScalar = ATOL) -> NDArrayBool:

    mask = np.isclose(x, 0, atol=float(tol))

    return mask


@beartype
def find_inf(x: NDArray, tol: RealScalar = ATOL) -> NDArrayBool:

    mask = np.isclose(x, np.inf, float(tol))

    return mask


@beartype
def is_realscaler(x: NDArray | RealScalar) -> RealScalar | Literal[False]:

    if isinstance(x, (float, int, np.floating, np.integer)):
        return x
    else:
        x = x.flatten()
        if x.shape[0] == 1:
            return x[0]
        else:
            return False


@beartype
def check_rotation_matrix(R: NDArray_3_3) -> NDArray_3_3 | Literal[False]:

    if (
        check_equal(R.T @ R, np.eye(3)) is False
        or check_equal(np.linalg.det(R), 1.0) is False
    ):
        return False
    else:
        return R


@beartype
def check_rotation_matrixb(Rs: NDArray_N_3_3) -> NDArray_N_3_3 | Literal[False]:

    E = cast(NDArray_N_3_3, np.einsum("nik,njk->nij", Rs, Rs))  # type: ignore
    if (
        check_equal(E, np.eye(3)) is False
        or check_equal(np.linalg.det(Rs), 1.0) is False
    ):
        return False
    else:
        return Rs


@beartype
def skew3(vector3: NDArray_3_1 | NDArray_3) -> NDArray_3_3:

    if vector3.ndim == 1:
        vector3 = vector3.reshape(-1, 1)

    skew_matrix = np.array(
        [
            [0, -vector3[2, 0], vector3[1, 0]],
            [vector3[2, 0], 0, -vector3[0, 0]],
            [-vector3[1, 0], vector3[0, 0], 0],
        ],
        dtype=np.float64,
    )

    return skew_matrix


@beartype
def skew3b(vector3b: NDArray_N_3_1 | NDArray_3_N) -> NDArray_N_3_3:

    if vector3b.ndim == 3:
        skew3_matrixb = np.zeros((vector3b.shape[0], 3, 3))
        skew3_matrixb[:, 0, 1] = -vector3b[:, 2, 0]
        skew3_matrixb[:, 1, 0] = vector3b[:, 2, 0]
        skew3_matrixb[:, 0, 2] = vector3b[:, 1, 0]
        skew3_matrixb[:, 2, 0] = -vector3b[:, 1, 0]
        skew3_matrixb[:, 1, 2] = -vector3b[:, 0, 0]
        skew3_matrixb[:, 2, 1] = vector3b[:, 0, 0]
    else:
        skew3_matrixb = np.zeros((vector3b.shape[1], 3, 3))
        skew3_matrixb[:, 0, 1] = -vector3b[2, :]
        skew3_matrixb[:, 1, 0] = vector3b[2, :]
        skew3_matrixb[:, 0, 2] = vector3b[1, :]
        skew3_matrixb[:, 2, 0] = -vector3b[1, :]
        skew3_matrixb[:, 1, 2] = -vector3b[0, :]
        skew3_matrixb[:, 2, 1] = vector3b[0, :]

    return skew3_matrixb


@beartype
def check_skew3_matrix(skew3_matrix: NDArray_3_3) -> NDArray_3_3 | Literal[False]:

    diag = np.diag(skew3_matrix)
    if (
        check_equal(diag, 0) is False
        or check_equal(skew3_matrix, -skew3_matrix.T) is False
    ):
        return False

    return skew3_matrix


@beartype
def check_skew3_matrixb(skew3_matrixb: NDArray_N_3_3) -> NDArray_N_3_3 | Literal[False]:

    if check_equal(skew3_matrixb[:, 0, 0], 0) is False:
        return False
    if check_equal(skew3_matrixb[:, 1, 1], 0) is False:
        return False
    if check_equal(skew3_matrixb[:, 2, 2], 0) is False:
        return False
    v_ = skew3_matrixb + skew3_matrixb.transpose(0, 2, 1)
    if check_equal(v_, 0) is False:
        return False

    return skew3_matrixb


@beartype
def veeskew3(skew3_matrix: NDArray_3_3, skip_check: bool = False) -> NDArray_3_1:

    if not skip_check and check_skew3_matrix(skew3_matrix) is False:
        raise ValueError("skew3_matrix is not skew3 matrix")

    vector3 = np.array(
        [
            [(skew3_matrix[2, 1] - skew3_matrix[1, 2]) / 2],
            [(skew3_matrix[0, 2] - skew3_matrix[2, 0]) / 2],
            [(skew3_matrix[1, 0] - skew3_matrix[0, 1]) / 2],
        ]
    )

    return vector3


@beartype
def veeskew3b(skew3_matrixb: NDArray_N_3_3, skip_check: bool = False) -> NDArray_N_3_1:

    if not skip_check and check_skew3_matrixb(skew3_matrixb) is False:
        raise ValueError("skew3_matrixb is not skew3 matrix")

    vector3b = np.zeros((skew3_matrixb.shape[0], 3, 1))
    vector3b[:, 0, 0] = skew3_matrixb[:, 2, 1]
    vector3b[:, 1, 0] = skew3_matrixb[:, 0, 2]
    vector3b[:, 2, 0] = skew3_matrixb[:, 1, 0]

    return vector3b


@beartype
def skew4(vector6: NDArray_6_1 | NDArray_6) -> NDArray_4_4:

    if vector6.ndim == 1:
        vector6 = vector6.reshape(-1, 1)

    skew_matrix = np.zeros((4, 4))
    skew_matrix[:3, :3] = skew3(vector6[:3])
    skew_matrix[:3, 3:] = vector6[3:]

    return skew_matrix


@beartype
def skew4b(vectorb6: NDArray_N_6_1 | NDArray_6_N) -> NDArray_N_4_4:

    if vectorb6.ndim == 3:
        skew4_matrixb = np.zeros((vectorb6.shape[0], 4, 4))
        skew4_matrixb[:, :3, :3] = skew3b(vectorb6[:, :3])
        skew4_matrixb[:, :3, 3:] = vectorb6[:, 3:]
    else:
        skew4_matrixb = np.zeros((vectorb6.shape[1], 4, 4))
        skew4_matrixb[:, :3, :3] = skew3b(vectorb6[:3])
        skew4_matrixb[:, :3, 3] = vectorb6[3:].T

    return skew4_matrixb


@beartype
def check_skew4_matrix(skew4_matrix: NDArray_4_4) -> NDArray_4_4 | Literal[False]:

    skew3_matrix = check_skew3_matrix(skew4_matrix[:3, :3])
    if skew3_matrix is False:
        return False
    if check_equal(skew4_matrix[-1, :], 0) is False:
        return False

    return skew4_matrix


@beartype
def check_skew4_matrixb(skew4_matrixb: NDArray_N_4_4) -> NDArray_N_4_4 | Literal[False]:

    skew3_matrixb = check_skew3_matrixb(skew4_matrixb[:, :3, :3])
    if skew3_matrixb is False:
        return False
    if check_equal(skew4_matrixb[:, -1, :], 0) is False:
        return False

    return skew4_matrixb


@beartype
def veeskew4(skew4_matrix: NDArray_4_4, skip_check: bool = False) -> NDArray_6_1:

    if not skip_check and check_skew4_matrix(skew4_matrix) is False:
        raise ValueError("skew4_matrix is not skew4 matrix")

    vector6 = np.zeros((6, 1))
    vector6[:3] = veeskew3(skew4_matrix[:3, :3], True)
    vector6[3:] = skew4_matrix[:3, 3:]

    return vector6


@beartype
def veeskew4b(skew4_matrixb: NDArray_N_4_4, skip_check: bool = False) -> NDArray_N_6_1:

    if not skip_check and check_skew4_matrixb(skew4_matrixb) is False:
        raise ValueError("skew4_matrixb is not skew4 matrixb")

    vectorb6 = np.zeros((skew4_matrixb.shape[0], 6, 1))
    vectorb6[:, :3] = veeskew3b(skew4_matrixb[:, :3, :3], True)
    vectorb6[:, 3:] = skew4_matrixb[:, :3, 3:]

    return vectorb6


@beartype
def skew6(vector6: NDArray_6_1 | NDArray_6) -> NDArray_6_6:

    if vector6.ndim == 1:
        vector6 = vector6.reshape(-1, 1)

    skew6_matrix = np.zeros((6, 6))
    w = skew3(vector6[:3])
    v = skew3(vector6[3:])
    skew6_matrix[:3, :3] = w
    skew6_matrix[3:, :3] = v
    skew6_matrix[3:, 3:] = w

    return skew6_matrix


@beartype
def skew6b(vectorb6: NDArray_N_6_1 | NDArray_6_N) -> NDArray_N_6_6:

    if vectorb6.ndim == 3:
        skew6_matrixb = np.zeros((vectorb6.shape[0], 6, 6))
        skew6_matrixb[:, :3, :3] = skew3b(vectorb6[:, :3])
        skew6_matrixb[:, 3:, :3] = skew3b(vectorb6[:, 3:])
        skew6_matrixb[:, 3:, 3:] = skew3b(vectorb6[:, :3])
    else:
        skew6_matrixb = np.zeros((vectorb6.shape[1], 6, 6))
        skew6_matrixb[:, :3, :3] = skew3b(vectorb6[:3])
        skew6_matrixb[:, 3:, :3] = skew3b(vectorb6[3:])
        skew6_matrixb[:, 3:, 3:] = skew3b(vectorb6[:3])

    return skew6_matrixb


@beartype
def check_skew6_matrix(skew6_matrix: NDArray_6_6) -> NDArray_6_6 | Literal[False]:

    skew3_w = check_skew3_matrix(skew6_matrix[:3, :3])
    skew3_w2 = check_skew3_matrix(skew6_matrix[3:, 3:])
    skew3_v = check_skew3_matrix(skew6_matrix[3:, :3])

    if skew3_w is False:
        return False
    if skew3_v is False:
        return False
    if check_equal(skew3_w, skew3_w2) is False:
        return False
    if check_equal(skew6_matrix[:3, 3:], 0) is False:
        return False

    return skew6_matrix


@beartype
def check_skew6_matrixb(skew6_matrixb: NDArray_N_6_6) -> NDArray_N_6_6 | Literal[False]:

    skew3_w = check_skew3_matrixb(skew6_matrixb[:, :3, :3])
    skew3_w2 = check_skew3_matrixb(skew6_matrixb[:, 3:, 3:])
    skew3_v = check_skew3_matrixb(skew6_matrixb[:, 3:, :3])

    if skew3_w is False:
        return False
    if skew3_v is False:
        return False
    if check_equal(skew3_w, skew3_w2) is False:
        return False
    if check_equal(skew6_matrixb[:, :3, 3:], 0) is False:
        return False

    return skew6_matrixb


@beartype
def coskew6(vector6: NDArray_6_1 | NDArray_6) -> NDArray_6_6:

    if vector6.ndim == 1:
        vector6 = vector6.reshape(-1, 1)

    skew6_matrix = np.zeros((6, 6))
    w = skew3(vector6[:3])
    v = skew3(vector6[3:])
    skew6_matrix[:3, :3] = w
    skew6_matrix[:3, 3:] = v
    skew6_matrix[3:, 3:] = w

    return skew6_matrix


@beartype
def coskew6b(vectorb6: NDArray_N_6_1 | NDArray_6_N) -> NDArray_N_6_6:

    if vectorb6.ndim == 3:
        coskew6_matrixb = np.zeros((vectorb6.shape[0], 6, 6))
        coskew6_matrixb[:, :3, :3] = skew3b(vectorb6[:, :3])
        coskew6_matrixb[:, :3, 3:] = skew3b(vectorb6[:, 3:])
        coskew6_matrixb[:, 3:, 3:] = skew3b(vectorb6[:, :3])
    else:
        coskew6_matrixb = np.zeros((vectorb6.shape[1], 6, 6))
        coskew6_matrixb[:, :3, :3] = skew3b(vectorb6[:3])
        coskew6_matrixb[:, :3, 3:] = skew3b(vectorb6[3:])
        coskew6_matrixb[:, 3:, 3:] = skew3b(vectorb6[:3])

    return coskew6_matrixb


@beartype
def check_coskew6_matrix(skew6_matrix: NDArray_6_6) -> NDArray_6_6 | Literal[False]:

    skew3_w = check_skew3_matrix(skew6_matrix[:3, :3])
    skew3_w2 = check_skew3_matrix(skew6_matrix[3:, 3:])
    skew3_v = check_skew3_matrix(skew6_matrix[:3, 3:])

    if skew3_w is False:
        return False
    if skew3_v is False:
        return False
    if check_equal(skew3_w, skew3_w2) is False:
        return False
    if check_equal(skew6_matrix[3:, :3], 0) is False:
        return False

    return skew6_matrix


@beartype
def check_coskew6_matrixb(
    skew6_matrixb: NDArray_N_6_6,
) -> NDArray_N_6_6 | Literal[False]:

    skew3_w = check_skew3_matrixb(skew6_matrixb[:, :3, :3])
    skew3_w2 = check_skew3_matrixb(skew6_matrixb[:, 3:, 3:])
    skew3_v = check_skew3_matrixb(skew6_matrixb[:, :3, 3:])

    if skew3_w is False:
        return False
    if skew3_v is False:
        return False
    if check_equal(skew3_w, skew3_w2) is False:
        return False
    if check_equal(skew6_matrixb[:, 3:, :3], 0) is False:
        return False

    return skew6_matrixb


@beartype
def veecoskew6(coskew6_matrix: NDArray_6_6, skip_check: bool = False) -> NDArray_6_1:

    if not skip_check and check_coskew6_matrix(coskew6_matrix) is False:
        raise ValueError("coskew6_matrix is not coskew6 matrix")

    vector6 = np.zeros((6, 1))
    vector6[:3] = veeskew3(coskew6_matrix[:3, :3], True)
    vector6[3:] = veeskew3(coskew6_matrix[:3, 3:], True)

    return vector6


@beartype
def veecoskew6b(
    coskew6_matrixb: NDArray_N_6_6, skip_check: bool = False
) -> NDArray_N_6_1:

    if not skip_check and check_coskew6_matrixb(coskew6_matrixb) is False:
        raise ValueError("coskew6_matrixb is not coskew6 matrix")

    vector6b = np.zeros((coskew6_matrixb.shape[0], 6, 1))
    vector6b[:, :3] = veeskew3b(coskew6_matrixb[:, :3, :3], True)
    vector6b[:, 3:] = veeskew3b(coskew6_matrixb[:, :3, 3:], True)

    return vector6b


@beartype
def veeskew6(skew6_matrix: NDArray_6_6, skip_check: bool = False) -> NDArray_6_1:

    if not skip_check and check_skew6_matrix(skew6_matrix) is False:
        raise ValueError("skew6_matrix is not skew6 matrix")

    vector6 = np.zeros((6, 1))
    vector6[:3] = veeskew3(skew6_matrix[:3, :3], True)
    vector6[3:] = veeskew3(skew6_matrix[3:, :3], True)

    return vector6


@beartype
def veeskew6b(skew6_matrixb: NDArray_N_6_6, skip_check: bool = False) -> NDArray_N_6_1:

    if not skip_check and check_skew6_matrixb(skew6_matrixb) is False:
        raise ValueError("skew6_matrixb is not skew6 matrix")

    vector6b = np.zeros((skew6_matrixb.shape[0], 6, 1))
    vector6b[:, :3] = veeskew3b(skew6_matrixb[:, :3, :3], True)
    vector6b[:, 3:] = veeskew3b(skew6_matrixb[:, 3:, :3], True)

    return vector6b


@beartype
def skew4_matrix_to_skew6_matrix(
    skew4_matrix: NDArray_4_4, skip_check: bool = False
) -> NDArray_6_6:

    if not skip_check and check_skew4_matrix(skew4_matrix) is False:
        raise ValueError("skew4_matrix is not skew4 matrix.")

    skew6_matrix = np.zeros((6, 6))
    skew6_matrix[:3, :3] = skew4_matrix[:3, :3]
    skew6_matrix[3:, 3:] = skew4_matrix[:3, :3]
    skew6_matrix[3:, :3] = skew3(skew4_matrix[:3, 3:])

    return skew6_matrix


@beartype
def skew6_matrix_to_skew4_matrix(
    skew6_matrix: NDArray_6_6, skip_check: bool = False
) -> NDArray_4_4:

    if not skip_check and check_skew6_matrix(skew6_matrix) is False:
        raise ValueError("skew6_matrix is not skew6 matrix.")

    skew4_matrix = np.zeros((4, 4))
    skew4_matrix[:3, :3] = skew6_matrix[:3, :3]
    skew4_matrix[:3, 3:] = veeskew3(skew6_matrix[3:, :3], True)

    return skew4_matrix


@beartype
def skew4_matrix_to_coskew6_matrix(
    skew4_matrix: NDArray_4_4, skip_check: bool = False
) -> NDArray_6_6:

    if not skip_check and check_skew4_matrix(skew4_matrix) is False:
        raise ValueError("skew4_matrix is not skew4 matrix.")

    coskew6_matrix = np.zeros((6, 6))
    coskew6_matrix[:3, :3] = skew4_matrix[:3, :3]
    coskew6_matrix[3:, 3:] = skew4_matrix[:3, :3]
    coskew6_matrix[:3, 3:] = skew3(skew4_matrix[:3, 3:])

    return coskew6_matrix


@beartype
def coskew6_matrix_to_skew4_matrix(
    coskew6_matrix: NDArray_6_6, skip_check: bool = False
) -> NDArray_4_4:

    if not skip_check and check_coskew6_matrix(coskew6_matrix) is False:
        raise ValueError("coskew6_matrix is not coskew6 matrix.")

    skew4_matrix = np.zeros((4, 4))
    skew4_matrix[:3, :3] = coskew6_matrix[:3, :3]
    skew4_matrix[:3, 3:] = veeskew3(coskew6_matrix[:3, 3:], True)

    return skew4_matrix


@beartype
def check_SE3_matrix(SE3_matrix: NDArray_4_4) -> NDArray_4_4 | Literal[False]:

    if check_rotation_matrix(SE3_matrix[:3, :3]) is False:
        return False
    if check_equal(SE3_matrix[-1, -1], 1) is False:
        return False
    if check_equal(SE3_matrix[-1, :-1], 0) is False:
        return False

    return SE3_matrix


@beartype
def check_SE3_matrixb(SE3_matrixb: NDArray_N_4_4) -> NDArray_N_4_4 | Literal[False]:

    if check_rotation_matrixb(SE3_matrixb[:, :3, :3]) is False:
        return False
    if check_equal(SE3_matrixb[:, -1, -1], 1) is False:
        return False
    if check_equal(SE3_matrixb[:, -1, :-1], 0) is False:
        return False

    return SE3_matrixb


@beartype
def check_Ad_matrix(
    Ad_matrix: NDArray_6_6, return_t: bool = False
) -> NDArray_6_6 | tuple[NDArray_6_6, NDArray_3_1] | Literal[False]:

    if check_equal(Ad_matrix[:3, :3], Ad_matrix[3:, 3:]) is False:
        return False
    if check_rotation_matrix(Ad_matrix[:3, :3]) is False:
        return False

    skew3t = Ad_matrix[3:, :3] @ Ad_matrix[:3, :3].T
    if check_skew3_matrix(skew3t) is False:
        return False

    if check_equal(Ad_matrix[:3, 3:], 0) is False:
        return False

    if return_t:
        return Ad_matrix, veeskew3(skew3t, True)
    return Ad_matrix


@beartype
def check_Ad_matrixb(
    Ad_matrixb: NDArray_N_6_6, return_t: bool = False
) -> NDArray_N_6_6 | tuple[NDArray_N_6_6, NDArray_N_3_1] | Literal[False]:

    if check_equal(Ad_matrixb[:, :3, :3], Ad_matrixb[:, 3:, 3:]) is False:
        return False
    if check_rotation_matrixb(Ad_matrixb[:, :3, :3]) is False:
        return False

    skew3tb = np.einsum("nik,njk->nij", Ad_matrixb[:, 3:, :3], Ad_matrixb[:, :3, :3])  # type: ignore
    skew3tb = cast(NDArray_N_3_3, skew3tb)
    if check_skew3_matrixb(skew3tb) is False:
        return False

    if check_equal(Ad_matrixb[:, :3, 3:], 0) is False:
        return False

    if return_t:
        return Ad_matrixb, veeskew3b(skew3tb, True)
    return Ad_matrixb


@beartype
def check_coAd_matrix(
    coAd_matrix: NDArray_6_6, return_t: bool = False
) -> NDArray_6_6 | tuple[NDArray_6_6, NDArray_3_1] | Literal[False]:

    if check_equal(coAd_matrix[:3, :3], coAd_matrix[3:, 3:]) is False:
        return False
    if check_rotation_matrix(coAd_matrix[:3, :3]) is False:
        return False

    skew3t = coAd_matrix[:3, 3:] @ coAd_matrix[:3, :3].T
    if check_skew3_matrix(skew3t) is False:
        return False

    if check_equal(coAd_matrix[3:, :3], 0) is False:
        return False

    if return_t:
        return coAd_matrix, veeskew3(skew3t, True)
    return coAd_matrix


@beartype
def check_coAd_matrixb(
    coAd_matrixb: NDArray_N_6_6, return_t: bool = False
) -> NDArray_N_6_6 | tuple[NDArray_N_6_6, NDArray_N_3_1] | Literal[False]:

    if check_equal(coAd_matrixb[:, :3, :3], coAd_matrixb[:, 3:, 3:]) is False:
        return False
    if check_rotation_matrix(coAd_matrixb[:, :3, :3]) is False:
        return False

    skew3tb = np.einsum(  # type: ignore
        "nik,njk->nij", coAd_matrixb[:, :3, 3:], coAd_matrixb[:, :3, :3]
    )
    skew3tb = cast(NDArray_N_3_3, skew3tb)
    if check_skew3_matrixb(skew3tb) is False:
        return False

    if check_equal(coAd_matrixb[:, 3:, :3], 0) is False:
        return False

    if return_t:
        return coAd_matrixb, veeskew3b(skew3tb, True)
    return coAd_matrixb


@beartype
def SE3_matrix_to_Ad_matrix(
    SE3_matrix: NDArray_4_4, skip_check: bool = False
) -> NDArray_6_6:

    if not skip_check and check_SE3_matrix(SE3_matrix) is False:
        raise ValueError("SE3_matrix is not SE3 matrix.")

    Ad_matrix = np.zeros((6, 6))
    skew_t = skew3(SE3_matrix[:3, 3:])

    Ad_matrix[:3, :3] = SE3_matrix[:3, :3]
    Ad_matrix[3:, 3:] = SE3_matrix[:3, :3]
    Ad_matrix[3:, :3] = skew_t @ SE3_matrix[:3, :3]

    return Ad_matrix


@beartype
def SE3_matrixb_to_Ad_matrixb(
    SE3_matrixb: NDArray_N_4_4, skip_check: bool = False
) -> NDArray_N_6_6:

    if not skip_check and check_SE3_matrixb(SE3_matrixb) is False:
        raise ValueError("SE3_matrixb is not SE3 matrixb.")

    Ad_matrixb = np.zeros((SE3_matrixb.shape[0], 6, 6))
    Ad_matrixb = cast(NDArray_N_6_6, Ad_matrixb)
    skew_tb = skew3b(SE3_matrixb[:, :3, 3:])

    Ad_matrixb[:, :3, :3] = SE3_matrixb[:, :3, :3]
    Ad_matrixb[:, 3:, 3:] = SE3_matrixb[:, :3, :3]
    Ad_matrixb[:, 3:, :3] = skew_tb @ SE3_matrixb[:, :3, :3]

    return Ad_matrixb


@beartype
def SE3_matrix_to_coAd_matrix(
    SE3_matrix: NDArray_4_4, skip_check: bool = False
) -> NDArray_6_6:

    if not skip_check and check_SE3_matrix(SE3_matrix) is False:
        raise ValueError("SE3_matrix is not SE3 matrix.")

    coAd_matrix = np.zeros((6, 6))
    skew_t = skew3(SE3_matrix[:3, 3:])

    coAd_matrix[:3, :3] = SE3_matrix[:3, :3]
    coAd_matrix[3:, 3:] = SE3_matrix[:3, :3]
    coAd_matrix[:3, 3:] = skew_t @ SE3_matrix[:3, :3]

    return coAd_matrix


@beartype
def SE3_matrixb_to_coAd_matrixb(
    SE3_matrixb: NDArray_N_4_4, skip_check: bool = False
) -> NDArray_N_6_6:

    if not skip_check and check_SE3_matrixb(SE3_matrixb) is False:
        raise ValueError("SE3_matrixb is not SE3 matrixb.")

    coAd_matrixb = np.zeros((SE3_matrixb.shape[0], 6, 6))
    coAd_matrixb = cast(NDArray_N_6_6, coAd_matrixb)
    skew_tb = skew3b(SE3_matrixb[:, :3, 3:])

    coAd_matrixb[:, :3, :3] = SE3_matrixb[:, :3, :3]
    coAd_matrixb[:, 3:, 3:] = SE3_matrixb[:, :3, :3]
    coAd_matrixb[:, :3, 3:] = skew_tb @ SE3_matrixb[:, :3, :3]

    return coAd_matrixb


@beartype
def Ad_matrix_to_SE3_matrix(
    Ad_matrix: NDArray_6_6, skip_check: bool = False
) -> NDArray_4_4:

    if not skip_check:
        check_result = check_Ad_matrix(Ad_matrix)
        if check_result is False:
            raise ValueError("Ad_matrix is not Ad matrix.")
        t = check_result[1]
    else:
        t = veeskew3(Ad_matrix[3:, :3] @ Ad_matrix[:3, :3].T, True)

    SE3_matrix = np.eye(4)
    SE3_matrix[:3, :3] = Ad_matrix[:3, :3]
    SE3_matrix[:3, 3:] = t

    return SE3_matrix


@beartype
def Ad_matrixb_to_SE3_matrixb(
    Ad_matrixb: NDArray_N_6_6, skip_check: bool = False
) -> NDArray_N_4_4:

    if not skip_check:
        check_result = check_Ad_matrixb(Ad_matrixb)
        if check_result is False:
            raise ValueError("Ad_matrixb is not Ad matrixb.")
        tb = check_result[1]
    else:
        tb = veeskew3b(
            np.einsum(  # type: ignore
                "nik,njk->nij", Ad_matrixb[:, 3:, :3], Ad_matrixb[:, :3, :3]
            ),
            True,
        )

    SE3_matrixb = np.zeros((Ad_matrixb.shape[0], 4, 4))
    SE3_matrixb[:, -1, -1] = 1
    SE3_matrixb[:, :3, :3] = Ad_matrixb[:, :3, :3]
    SE3_matrixb[:, :3, 3:] = tb

    return SE3_matrixb


@beartype
def coAd_matrix_to_SE3_matrix(
    coAd_matrix: NDArray_6_6, skip_check: bool = False
) -> NDArray_4_4:

    if not skip_check:
        check_result = check_coAd_matrix(coAd_matrix)
        if check_result is False:
            raise ValueError("coAd_matrix is not coAd matrix.")
        t = check_result[1]
    else:
        t = veeskew3(coAd_matrix[:3, 3:] @ coAd_matrix[:3, :3].T, True)

    SE3_matrix = np.eye(4)
    SE3_matrix[:3, :3] = coAd_matrix[:3, :3]
    SE3_matrix[:3, 3:] = t

    return SE3_matrix


@beartype
def coAd_matrixb_to_SE3_matrixb(
    coAd_matrixb: NDArray_N_6_6, skip_check: bool = False
) -> NDArray_N_4_4:

    if not skip_check:
        check_result = check_coAd_matrixb(coAd_matrixb)
        if check_result is False:
            raise ValueError("coAd_matrixb is not coAd matrixb.")
        tb = check_result[1]
    else:
        tb = veeskew3b(
            np.einsum(  # type: ignore
                "nik,njk->nij", coAd_matrixb[:, :3, 3:], coAd_matrixb[:, :3, :3]
            ),
            True,
        )

    SE3_matrixb = np.zeros((coAd_matrixb.shape[0], 4, 4))
    SE3_matrixb[:, -1, -1] = 1
    SE3_matrixb[:, :3, :3] = coAd_matrixb[:, :3, :3]
    SE3_matrixb[:, :3, 3:] = tb

    return SE3_matrixb


@beartype
def rot3c(
    axis: Literal["x", "y", "z"],
    theta: RealScalar,
) -> NDArray_3_3:

    sin_t = np.sin(theta)
    cos_t = np.cos(theta)

    toArray: Any = np.array

    if axis == "x":
        v = toArray(
            [
                [1, 0, 0],
                [0, cos_t, -sin_t],
                [0, sin_t, cos_t],
            ]
        )
    elif axis == "y":
        v = toArray(
            [
                [cos_t, 0, sin_t],
                [0, 1, 0],
                [-sin_t, 0, cos_t],
            ]
        )
    elif axis == "z":
        v = toArray(
            [
                [cos_t, -sin_t, 0],
                [sin_t, cos_t, 0],
                [0, 0, 1],
            ]
        )

    return v


@beartype
def rot3cb(
    axis: Literal["x", "y", "z"], theta: RealScalar | NDArray_1D
) -> NDArray_N_3_3:

    if is_realscaler(theta):
        num = 1
    else:
        theta = cast(NDArray_1D, theta)
        num = theta.shape[0]

    v = np.zeros((num, 3, 3))
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    if axis == "x":
        v[:, 0, 0] = 1.0
        v[:, 1, 1] = cos_t
        v[:, 1, 2] = -sin_t
        v[:, 2, 1] = sin_t
        v[:, 2, 2] = cos_t
    elif axis == "y":
        v[:, 0, 0] = cos_t
        v[:, 0, 2] = sin_t
        v[:, 1, 1] = 1.0
        v[:, 2, 0] = -sin_t
        v[:, 2, 2] = cos_t
    else:
        v[:, 0, 0] = cos_t
        v[:, 0, 1] = -sin_t
        v[:, 1, 0] = sin_t
        v[:, 1, 1] = cos_t
        v[:, 2, 2] = 1.0

    return v


@beartype
def rot4c(
    axis: Literal["x", "y", "z"],
    theta: RealScalar,
) -> NDArray_4_4:

    sin_t = np.sin(theta)
    cos_t = np.cos(theta)

    toArray: Any = np.array

    if axis == "x":
        v = toArray(
            [
                [1, 0, 0, 0],
                [0, cos_t, -sin_t, 0],
                [0, sin_t, cos_t, 0],
                [0, 0, 0, 1],
            ]
        )
    elif axis == "y":
        v = toArray(
            [
                [cos_t, 0, sin_t, 0],
                [0, 1, 0, 0],
                [-sin_t, 0, cos_t, 0],
                [0, 0, 0, 1],
            ]
        )
    elif axis == "z":
        v = toArray(
            [
                [cos_t, -sin_t, 0, 0],
                [sin_t, cos_t, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )

    return v


@beartype
def rot4cb(
    axis: Literal["x", "y", "z"], theta: RealScalar | NDArray_1D
) -> NDArray_N_4_4:

    if is_realscaler(theta):
        num = 1
    else:
        theta = cast(NDArray_1D, theta)
        num = theta.shape[0]

    v = np.zeros((num, 4, 4))
    v[:, -1, -1] = 1
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    if axis == "x":
        v[:, 0, 0] = 1.0
        v[:, 1, 1] = cos_t
        v[:, 1, 2] = -sin_t
        v[:, 2, 1] = sin_t
        v[:, 2, 2] = cos_t
    elif axis == "y":
        v[:, 0, 0] = cos_t
        v[:, 0, 2] = sin_t
        v[:, 1, 1] = 1.0
        v[:, 2, 0] = -sin_t
        v[:, 2, 2] = cos_t
    else:
        v[:, 0, 0] = cos_t
        v[:, 0, 1] = -sin_t
        v[:, 1, 0] = sin_t
        v[:, 1, 1] = cos_t
        v[:, 2, 2] = 1.0

    return v


@beartype
def rot6c(
    axis: Literal["x", "y", "z"],
    theta: RealScalar,
) -> NDArray_6_6:

    sin_t = np.sin(theta)
    cos_t = np.cos(theta)

    toArray: Any = np.array

    if axis == "x":
        v = toArray(
            [
                [1, 0, 0, 0, 0, 0],
                [0, cos_t, -sin_t, 0, 0, 0],
                [0, sin_t, cos_t, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, cos_t, -sin_t],
                [0, 0, 0, 0, sin_t, cos_t],
            ]
        )
    elif axis == "y":
        v = toArray(
            [
                [cos_t, 0, sin_t, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [-sin_t, 0, cos_t, 0, 0, 0],
                [0, 0, 0, cos_t, 0, sin_t],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, -sin_t, 0, cos_t],
            ]
        )
    elif axis == "z":
        v = toArray(
            [
                [cos_t, -sin_t, 0, 0, 0, 0],
                [sin_t, cos_t, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, cos_t, -sin_t, 0],
                [0, 0, 0, sin_t, cos_t, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )

    return v


@beartype
def rot6cb(
    axis: Literal["x", "y", "z"], theta: RealScalar | NDArray_1D
) -> NDArray_N_6_6:

    if is_realscaler(theta):
        num = 1
    else:
        theta = cast(NDArray_1D, theta)
        num = theta.shape[0]

    v = np.zeros((num, 6, 6))
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    if axis == "x":
        v[:, 0, 0] = 1.0
        v[:, 1, 1] = cos_t
        v[:, 1, 2] = -sin_t
        v[:, 2, 1] = sin_t
        v[:, 2, 2] = cos_t
    elif axis == "y":
        v[:, 0, 0] = cos_t
        v[:, 0, 2] = sin_t
        v[:, 1, 1] = 1.0
        v[:, 2, 0] = -sin_t
        v[:, 2, 2] = cos_t
    else:
        v[:, 0, 0] = cos_t
        v[:, 0, 1] = -sin_t
        v[:, 1, 0] = sin_t
        v[:, 1, 1] = cos_t
        v[:, 2, 2] = 1.0

    v[:, 3:, 3:] = v[:, :3, :3]

    return v


@beartype
def get_Type(instance: object) -> Any:

    if isinstance(instance, (float, int, np.floating, np.integer)):
        return RealScalar

    elif isinstance(instance, np.ndarray):
        return get_NDArrayAnnotation(cast(NDArray, instance))

    elif instance is None:
        return None

    else:
        return type(instance)


@beartype
def get_TypeName(type_: Any) -> str:

    if type_ is RealScalar:
        return "RealScalar"
    elif is_NDArrayAnnotation(type_):
        shape = get_NDArrayAnnotationShape(type_)
        shape = "×".join([str(i) for i in shape])
        return f"shape-{shape} NDArray"
    elif type_ is None:
        return "None"
    else:
        return type_.__name__


@beartype
def print4(x: NDArray) -> None:
    print(np.round(x, 4))


@beartype
def __check_input_shape(num: int, shape: tuple[int, int]) -> tuple[int, int]:

    if shape[0] > 0 and shape[1] > 0:
        if shape[0] * shape[1] != num:
            raise ValueError(
                f"The input shape {shape} can not be satisfied by the input array {num}."
            )
        group_num, each_num = shape
    elif shape[0] < 0 and shape[1] < 0:
        raise ValueError("The input shape can not be negative at the same time.")
    elif shape[0] > 0 and shape[1] < 0:
        group_num = shape[0]
        if num % group_num != 0:
            raise ValueError(
                f"The input shape {shape} can not be satisfied by the input array {num}."
            )
        each_num = num // group_num
    else:
        each_num = shape[1]
        if num % each_num != 0:
            raise ValueError(
                f"The input shape {shape} can not be satisfied by the input array {num}."
            )
        group_num = num // each_num

    return group_num, each_num


@beartype
def NDArray_3Dto4D(array3D: NDArray_3D, shape: tuple[int, int]) -> NDArray_4D:
    num = array3D.shape[0]
    group_num, each_num = __check_input_shape(num, shape)
    H, W = array3D.shape[1], array3D.shape[2]
    return array3D.reshape(group_num, each_num, H, W)


@beartype
def NDArray_4Dto3D(array4D: NDArray_4D) -> NDArray_3D:
    return array4D.reshape(-1, array4D.shape[2], array4D.shape[3])


@beartype
def NDArray_3Dto2D(
    array3D: NDArray_3D, shape: tuple[int, int] | None = None
) -> NDArray_2D:
    if shape is None:
        return array3D.transpose(1, 0, 2).reshape(array3D.shape[1], -1)
    else:
        num = array3D.shape[0]
        row_num, col_num = __check_input_shape(num, shape)
        r, c = array3D.shape[1], array3D.shape[2]
        blocks = array3D.reshape(row_num, col_num, r, c)

        return blocks.transpose(0, 2, 1, 3).reshape(row_num * r, col_num * c)
