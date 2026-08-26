from __future__ import annotations

from typing import Any, Literal, Self, cast

import numpy as np
from beartype import beartype

import lieink.utils as utils
from lieink.annotations import (
    NDArray_1D,
    NDArray_3,
    NDArray_3_1,
    NDArray_3_3,
    NDArray_3_N,
    NDArray_4,
    NDArray_4_1,
    NDArray_4_4,
    NDArray_4_N,
    NDArray_6,
    NDArray_6_1,
    NDArray_6_6,
    NDArray_6_N,
    NDArray_N_3_1,
    NDArray_N_3_3,
    NDArray_N_4_1,
    NDArray_N_4_4,
    NDArray_N_6_1,
    NDArray_N_6_6,
    RealScalar,
)
from lieink.basic_object import BasicLie as _BasicLie
from lieink.basic_object import BasicPoint as _BasicPoint
from lieink.basic_object import BasicVector as _BasicVector
from lieink.basic_object import Lie
from lieink.basic_object import LieAlgebra as _LieAlgebra
from lieink.basic_object import LieAlgebra3 as _LieAlgebra3
from lieink.basic_object import LieGroup as _LieGroup
from lieink.basic_object import LieGroup3 as _LieGroup3


@beartype
class Vector3(
    _BasicVector[
        NDArray_3_1 | NDArray_3,
        NDArray_3_1,
        NDArray_N_3_1 | NDArray_3_N,
        NDArray_N_3_1,
    ]
):
    _r = 3
    _c = 1

    @classmethod
    def set_initial_pointc(
        cls,
        v: NDArray_3_1 | NDArray_3,
        initial_point: NDArray_3_1 | NDArray_3,
        skip_check: bool = False,
    ) -> NDArray_3_1:
        v = cls._check_shape_and_value(v, skip_check=skip_check)
        initial_point = Point3._check_shape_and_value(
            initial_point, skip_check=skip_check
        )
        return initial_point + v

    @classmethod
    def set_initial_pointcb(
        cls,
        v: NDArray_N_3_1 | NDArray_3_N,
        initial_point: NDArray_N_3_1 | NDArray_3_N,
        skip_check: bool = False,
    ) -> NDArray_N_3_1:
        v = cls._check_shapeb_and_valueb(v, skip_check=skip_check)
        initial_point = Point3._check_shapeb_and_valueb(
            initial_point, skip_check=skip_check
        )
        return Lie.addcb(initial_point, v)

    def set_initial_point(
        self, initial_point: Point3 | NDArray_3_1 | NDArray_3, skip_check: bool = False
    ) -> Point3:
        v = self._check_assigned()
        if isinstance(initial_point, Point3):
            initial_point = initial_point._check_assigned()
        else:
            initial_point = Point3._check_shape_and_value(
                initial_point, skip_check=skip_check
            )
        return Point3(initial_point + v)

    @classmethod
    def toVector4c(
        cls, v: NDArray_3_1 | NDArray_3, skip_check: bool = False
    ) -> NDArray_4_1:
        v = cls._check_shape_and_value(v, skip_check=skip_check)
        Vector4_v = np.zeros((4, 1))
        Vector4_v[:3] = v
        return Vector4_v

    @classmethod
    def toVector4cb(
        cls, v: NDArray_N_3_1 | NDArray_3_N, skip_check: bool = False
    ) -> NDArray_N_4_1:
        v = cls._check_shapeb_and_valueb(v, skip_check=skip_check)
        Vector4_v = np.zeros((v.shape[0], 4, 1))
        Vector4_v[:, :3] = v
        return Vector4_v

    def toVector4(self) -> Vector4:
        v = self._check_assigned()
        Vector4_v = np.zeros((4, 1))
        Vector4_v[:3] = v
        return Vector4(Vector4_v)


@beartype
class Vector4(
    _BasicVector[
        NDArray_4_1 | NDArray_4,
        NDArray_4_1,
        NDArray_N_4_1 | NDArray_4_N,
        NDArray_N_4_1,
    ]
):
    _r = 4
    _c = 1

    @classmethod
    def _check_value(cls, v: NDArray_4_1 | NDArray_4) -> NDArray_4_1:
        v = super()._check_value(v)
        if utils.check_equal(v[3], 0) is False:
            raise ValueError("v[3] is not 0")
        return v

    @classmethod
    def _check_valueb(cls, v: NDArray_N_4_1 | NDArray_4_N) -> NDArray_N_4_1:
        v = super()._check_valueb(v)
        if not utils.find_zeros(v[:, 3]).all():
            raise ValueError("v[:, 3] is not zero vector")
        return v

    @classmethod
    def set_initial_pointc(
        cls,
        v: NDArray_4_1 | NDArray_4,
        initial_point: NDArray_4_1 | NDArray_4,
        skip_check: bool = False,
    ) -> NDArray_4_1:
        v = cls._check_shape_and_value(v, skip_check=skip_check)
        initial_point = Point4._check_shape_and_value(
            initial_point, skip_check=skip_check
        )
        initial_point = Point4.normalizec(initial_point, skip_check=True)
        return initial_point + v

    @classmethod
    def set_initial_pointcb(
        cls,
        v: NDArray_N_4_1 | NDArray_4_N,
        initial_point: NDArray_N_4_1 | NDArray_4_N,
        skip_check: bool = False,
    ) -> NDArray_N_4_1:
        v = cls._check_shapeb_and_valueb(v, skip_check=skip_check)
        initial_point = Point4._check_shapeb_and_valueb(
            initial_point, skip_check=skip_check
        )
        initial_point = Point4.normalizecb(initial_point, skip_check=True)
        return Lie.addcb(initial_point, v)

    def set_initial_point(
        self, initial_point: Point4 | NDArray_4_1 | NDArray_4, skip_check: bool = False
    ) -> Point4:
        v = self._check_assigned()
        if isinstance(initial_point, Point4):
            initial_point = initial_point._check_assigned()
        else:
            initial_point = Point4._check_shape_and_value(
                initial_point, skip_check=skip_check
            )
        return Point4(initial_point + v)

    @classmethod
    def toVector3c(
        cls, v: NDArray_4_1 | NDArray_4, skip_check: bool = False
    ) -> NDArray_3_1:
        v = cls._check_shape_and_value(v, skip_check=skip_check)
        return v[:3]

    @classmethod
    def toVector3cb(
        cls, v: NDArray_N_4_1 | NDArray_4_N, skip_check: bool = False
    ) -> NDArray_N_3_1:
        v = cls._check_shapeb_and_valueb(v, skip_check=skip_check)
        return v[:, :3]

    def toVector3(self) -> Vector3:
        v = self._check_assigned()
        return Vector3(v[:3])


@beartype
class Point3(
    _BasicPoint[
        NDArray_3_1 | NDArray_3,
        NDArray_3_1,
        NDArray_N_3_1 | NDArray_3_N,
        NDArray_N_3_1,
    ]
):
    _r = 3
    _c = 1

    @classmethod
    def scalec(cls, v: NDArray_3_1 | NDArray_3, skip_check: bool = False) -> RealScalar:
        return 1.0

    @classmethod
    def scalecb(
        cls, v: NDArray_N_3_1 | NDArray_3_N, skip_check: bool = False
    ) -> NDArray_1D:
        v = cls._check_shapeb_and_valueb(v, skip_check=skip_check)
        return np.ones(v.shape[0]).astype(np.float64)

    @classmethod
    def toPoint4c(
        cls, v: NDArray_3_1 | NDArray_3, skip_check: bool = False
    ) -> NDArray_4_1:
        v = cls._check_shape_and_value(v, skip_check=skip_check)
        Point4_v = np.zeros((4, 1))
        Point4_v[:3] = v
        Point4_v[3] = 1
        return Point4_v

    @classmethod
    def toPoint4cb(
        cls, v: NDArray_N_3_1 | NDArray_3_N, skip_check: bool = False
    ) -> NDArray_N_4_1:
        v = cls._check_shapeb_and_valueb(v, skip_check=skip_check)
        Point4_v = np.zeros((v.shape[0], 4, 1))
        Point4_v[:, :3] = v
        Point4_v[:, 3] = 1
        return Point4_v

    def toPoint4(self) -> Point4:
        v = self._check_assigned()
        return Point4(self.toPoint4c(v, skip_check=True))

    @classmethod
    def get_Vector3c(
        cls,
        v: NDArray_3_1 | NDArray_3,
        normalize: bool = False,
        skip_check: bool = False,
    ) -> NDArray_3_1:
        v = cls._check_shape_and_value(v, skip_check=skip_check)
        if normalize:
            v = Vector3.normalizec(v, skip_check=True)
        return v

    @classmethod
    def get_Vector3cb(
        cls,
        v: NDArray_N_3_1 | NDArray_3_N,
        normalize: bool = False,
        skip_check: bool = False,
    ) -> NDArray_N_3_1:
        v = cls._check_shapeb_and_valueb(v, skip_check=skip_check)
        if normalize:
            v = Vector3.normalizecb(v, skip_check=True)
        return v

    def get_Vector3(self, normalize: bool = False) -> Vector3:
        v = self._check_assigned()
        return Vector3(self.get_Vector3c(v, normalize=normalize, skip_check=True))

    @classmethod
    def get_Vector4c(
        cls,
        v: NDArray_4_1 | NDArray_4,
        normalize: bool = False,
        skip_check: bool = False,
    ) -> NDArray_4_1:
        v = cls._check_shape_and_value(v, skip_check=skip_check)
        vector4_v = np.zeros((4, 1))
        vector4_v[:3] = v
        if normalize:
            vector4_v = Vector4.normalizec(vector4_v, skip_check=True)
        return vector4_v

    @classmethod
    def get_Vector4cb(
        cls,
        v: NDArray_N_4_1 | NDArray_4_N,
        normalize: bool = False,
        skip_check: bool = False,
    ) -> NDArray_N_4_1:
        v = cls._check_shapeb_and_valueb(v, skip_check=skip_check)
        vector4_v = np.zeros((v.shape[0], 4, 1))
        vector4_v[:, :3] = v
        if normalize:
            vector4_v = Vector4.normalizecb(vector4_v, skip_check=True)
        return vector4_v

    def get_Vector4(self, normalize: bool = False) -> Vector4:
        v = self._check_assigned()
        return Vector4(self.get_Vector4c(v, normalize=normalize, skip_check=True))


@beartype
class Point4(
    _BasicPoint[
        NDArray_4_1 | NDArray_4,
        NDArray_4_1,
        NDArray_N_4_1 | NDArray_4_N,
        NDArray_N_4_1,
    ]
):
    _r = 4
    _c = 1

    def __init__(
        self,
        v: NDArray_4_1 | NDArray_4 | _BasicLie[Any, Any, Any, Any] | None = None,
        normalize: bool = False,
    ) -> None:
        super().__init__(v)
        if normalize:
            self.normalize(True)

    def __add__(self, other: Any) -> _BasicLie[Any, Any, Any, Any]:
        if isinstance(other, Point4):
            self_v = self._check_assigned()
            other_v = other._check_assigned()
            self_v = self_v / self_v[3]
            other_v = other_v / other_v[3]
            new_v = self_v + other_v
            new_v[-1] = 1
            return Point4(self_v + other_v)
        else:
            return super().__add__(other)

    @classmethod
    def _check_value(cls, v: NDArray_4_1 | NDArray_4) -> NDArray_4_1:
        v = super()._check_value(v)
        if utils.check_equal(v[3], 0) is not False:
            raise ValueError("v[3] is 0")
        return v

    @classmethod
    def _check_valueb(cls, v: NDArray_N_4_1 | NDArray_4_N) -> NDArray_N_4_1:
        v = super()._check_valueb(v)
        if utils.find_zeros(v[:, 3]).any():
            raise ValueError("v[:, 3] is 0")
        return v

    @classmethod
    def scalec(cls, v: NDArray_4_1 | NDArray_4, skip_check: bool = False) -> RealScalar:
        v = cls._check_shape_and_value(v, skip_check=skip_check)
        return v[3, 0]

    @classmethod
    def scalecb(
        cls, v: NDArray_N_4_1 | NDArray_4_N, skip_check: bool = False
    ) -> NDArray_1D:
        v = cls._check_shapeb_and_valueb(v, skip_check=skip_check)
        return v[:, 3, 0]

    @classmethod
    def set_scalec(
        cls, scale: RealScalar, v: NDArray_4_1 | NDArray_4, skip_check: bool = False
    ) -> NDArray_4_1:
        v = cls._check_shape_and_value(v, skip_check=skip_check)
        if utils.check_equal(scale, 0) is not False:
            raise ValueError("scale is 0")
        return scale * v / v[3, 0]

    @classmethod
    def set_scalecb(
        cls,
        scale: RealScalar | NDArray_1D,
        v: NDArray_N_4_1 | NDArray_4_N,
        skip_check: bool = False,
    ) -> NDArray_N_4_1:
        v = cls._check_shapeb_and_valueb(v, skip_check=skip_check)
        if utils.is_realscaler(scale):
            if utils.check_equal(scale, 0) is not False:
                raise ValueError("scale is 0")
        else:
            scale = cast(NDArray_1D, scale)
            scale = scale.reshape(-1, 1, 1)
            if utils.find_zeros(scale).any():
                raise ValueError("scale contains 0")
            if scale.shape[0] != v.shape[0]:
                raise ValueError("scale and v have different shape")
        return scale * v / v[:, 3:, 0:1]

    def set_scale(self, scale: RealScalar, inplace: bool = False) -> Self:
        self_v = self._check_assigned()
        v = self.set_scalec(scale, self_v, skip_check=True)
        return self._inplace_or_not(v, inplace)

    @classmethod
    def toPoint3c(
        cls, v: NDArray_4_1 | NDArray_4, skip_check: bool = False
    ) -> NDArray_3_1:
        v = cls._check_shape_and_value(v, skip_check=skip_check)
        return cls.normalizec(v, skip_check=True)[:3]

    @classmethod
    def toPoint3cb(
        cls, v: NDArray_N_3_1 | NDArray_3_N, skip_check: bool = False
    ) -> NDArray_N_4_1:
        v = cls._check_shapeb_and_valueb(v, skip_check=skip_check)
        return cls.normalizecb(v, skip_check=True)[:, :3]

    def toPoint3(self) -> Point3:
        v = self._check_assigned()
        return Point3(self.toPoint3c(v, skip_check=True))

    @classmethod
    def get_Vector3c(
        cls,
        v: NDArray_4_1 | NDArray_4,
        normalize: bool = False,
        skip_check: bool = False,
    ) -> NDArray_3_1:

        v = cls.normalizec(v, skip_check=skip_check)[:3]
        if normalize:
            v = Vector3.normalizec(v, skip_check=True)
        return v

    @classmethod
    def get_Vector3cb(
        cls,
        v: NDArray_N_4_1 | NDArray_4_N,
        normalize: bool = False,
        skip_check: bool = False,
    ) -> NDArray_N_4_1:

        v = cls.normalizecb(v, skip_check=skip_check)[:, :3]
        if normalize:
            v = Vector3.normalizecb(v, skip_check=True)
        return v

    def get_Vector3(self, normalize: bool = False) -> Vector3:
        v = self._check_assigned()
        v = self.normalizec(v, skip_check=True)
        return Vector3(self.get_Vector3c(v, normalize=normalize, skip_check=True))

    @classmethod
    def get_Vector4c(
        cls,
        v: NDArray_4_1 | NDArray_4,
        normalize: bool = False,
        skip_check: bool = False,
    ) -> NDArray_4_1:

        v = cls.normalizec(v, skip_check=skip_check)
        vector4_v = np.zeros((4, 1))
        vector4_v[:3] = v
        if normalize:
            vector4_v = Vector4.normalizec(vector4_v, skip_check=True)
        return vector4_v

    @classmethod
    def get_Vector4cb(
        cls,
        v: NDArray_N_4_1 | NDArray_4_N,
        normalize: bool = False,
        skip_check: bool = False,
    ) -> NDArray_N_4_1:

        v = cls.normalizecb(v, skip_check=skip_check)
        vector4_v = np.zeros((v.shape[0], 4, 1))
        vector4_v[:, :3] = v
        if normalize:
            vector4_v = Vector4.normalizecb(vector4_v, skip_check=True)
        return vector4_v

    def get_Vector4(self, normalize: bool = False) -> Vector4:
        v = self._check_assigned()
        return Vector4(self.get_Vector4c(v, normalize=normalize, skip_check=True))


@beartype
class so3(
    _LieAlgebra3[
        NDArray_3_3,
        NDArray_3_3,
        NDArray_N_3_3,
        NDArray_N_3_3,
    ]
):
    _r = 3
    _c = 3

    @classmethod
    def _check_value(cls, v: NDArray_3_3) -> NDArray_3_3:

        v = super()._check_value(v)
        if utils.check_skew3_matrix(v) is False:
            raise ValueError("v is not a skew3 matrix")
        return v

    @classmethod
    def _check_valueb(cls, v: NDArray_N_3_3) -> NDArray_N_3_3:

        v = super()._check_valueb(v)
        if utils.check_skew3_matrixb(v) is False:
            raise ValueError("v is not skew3 matrixb")
        return v


@beartype
class Twist3(
    _LieAlgebra3[
        NDArray_3_1 | NDArray_3,
        NDArray_3_1,
        NDArray_N_3_1 | NDArray_3_N,
        NDArray_N_3_1,
    ]
):
    _r = 3
    _c = 1


@beartype
class SO3(
    _LieGroup3[
        NDArray_3_3,
        NDArray_3_3,
        NDArray_N_3_3,
        NDArray_N_3_3,
    ]
):
    _r = 3
    _c = 3

    @classmethod
    def _check_value(cls, v: NDArray_3_3) -> NDArray_3_3:
        v = super()._check_value(v)
        if utils.check_rotation_matrix(v) is False:
            raise ValueError("v is not a rotation matrix")
        return v

    @classmethod
    def _check_valueb(cls, v: NDArray_N_3_3) -> NDArray_N_3_3:
        v = super()._check_valueb(v)
        if utils.check_rotation_matrixb(v) is False:
            raise ValueError("Not all blocks in v are a rotation matrixb")
        return v

    @classmethod
    def invc(cls, v: NDArray_3_3, skip_check: bool = False) -> NDArray_3_3:
        v = cls._check_shape_and_value(v, skip_check=skip_check)
        return v.T

    @classmethod
    def invcb(cls, v: NDArray_N_3_3, skip_check: bool = False) -> NDArray_N_3_3:
        v = cls._check_shapeb_and_valueb(v, skip_check=skip_check)
        return v.transpose(0, 2, 1)

    @classmethod
    def _rot_basec(cls, axis: Literal["x", "y", "z"], theta: RealScalar) -> NDArray_3_3:
        return utils.rot3c(axis, theta)

    @classmethod
    def _rot_basecb(
        cls, axis: Literal["x", "y", "z"], theta: NDArray_1D | RealScalar
    ) -> NDArray_N_3_3:
        return utils.rot3cb(axis, theta)

    @property
    def T(self) -> Self:  # type: ignore
        v = self._check_assigned()
        return SO3(v.T)  # type: ignore


@beartype
class se3(
    _LieAlgebra[
        NDArray_4_4,
        NDArray_4_4,
        NDArray_N_4_4,
        NDArray_N_4_4,
    ]
):
    _r = 4
    _c = 4

    @classmethod
    def _check_value(cls, v: NDArray_4_4) -> NDArray_4_4:
        v = super()._check_value(v)
        if utils.check_skew4_matrix(v) is False:
            raise ValueError("v is not skew4 matrix.")
        return v

    @classmethod
    def _check_valueb(cls, v: NDArray_N_4_4) -> NDArray_N_4_4:
        v = super()._check_valueb(v)
        if utils.check_skew4_matrixb(v) is False:
            raise ValueError("v is not skew4 matrixb.")
        return v

    @classmethod
    def angular_part_skewc(
        cls, v: NDArray_4_4, skip_check: bool = False
    ) -> NDArray_3_3:
        v = cls._check_shape_and_value(v, skip_check=skip_check)
        return v[:3, :3]

    @classmethod
    def angular_part_skewcb(
        cls, v: NDArray_N_4_4, skip_check: bool = False
    ) -> NDArray_N_3_3:
        v = cls._check_shapeb_and_valueb(v, skip_check=skip_check)
        return v[:, :3, :3]

    @classmethod
    def angular_part_veec(cls, v: NDArray_4_4, skip_check: bool = False) -> NDArray_3_1:
        v = cls._check_shape_and_value(v, skip_check=skip_check)
        return utils.veeskew3(v[:3, :3], skip_check=True)

    @classmethod
    def angular_part_veecb(
        cls, v: NDArray_N_4_4, skip_check: bool = False
    ) -> NDArray_N_3_1:
        v = cls._check_shapeb_and_valueb(v, skip_check=skip_check)
        return utils.veeskew3b(v[:, :3, :3], skip_check=True)

    @classmethod
    def linear_part3c(cls, v: NDArray_4_4, skip_check: bool = False) -> NDArray_3_1:
        v = cls._check_shape_and_value(v, skip_check=skip_check)
        return v[:3, 3:]

    @classmethod
    def linear_part3cb(
        cls, v: NDArray_N_4_4, skip_check: bool = False
    ) -> NDArray_N_3_1:
        v = cls._check_shapeb_and_valueb(v, skip_check=skip_check)
        return v[:, :3, 3:]


@beartype
class Twist(
    _LieAlgebra[
        NDArray_6_1 | NDArray_6,
        NDArray_6_1,
        NDArray_N_6_1 | NDArray_6_N,
        NDArray_N_6_1,
    ]
):
    _r = 6
    _c = 1

    @classmethod
    def angular_part_skewc(
        cls, v: NDArray_6_1 | NDArray_6, skip_check: bool = False
    ) -> NDArray_3_3:
        v = cls._check_shape_and_value(v, skip_check=skip_check)
        return utils.skew3(v[:3])

    @classmethod
    def angular_part_skewcb(
        cls, v: NDArray_N_6_1 | NDArray_6_N, skip_check: bool = False
    ) -> NDArray_N_3_3:
        v = cls._check_shapeb_and_valueb(v, skip_check=skip_check)
        return utils.skew3b(v[:, :3])

    @classmethod
    def angular_part_veec(
        cls, v: NDArray_6_1 | NDArray_6, skip_check: bool = False
    ) -> NDArray_3_1:
        v = cls._check_shape_and_value(v, skip_check=skip_check)
        return v[:3]

    @classmethod
    def angular_part_veecb(
        cls, v: NDArray_N_6_1 | NDArray_6_N, skip_check: bool = False
    ) -> NDArray_N_3_1:
        v = cls._check_shapeb_and_valueb(v, skip_check=skip_check)
        return v[:, :3]

    @classmethod
    def linear_part3c(
        cls, v: NDArray_6_1 | NDArray_6, skip_check: bool = False
    ) -> NDArray_3_1:
        v = cls._check_shape_and_value(v, skip_check=skip_check)
        return v[3:]

    @classmethod
    def linear_part3cb(
        cls, v: NDArray_N_6_1 | NDArray_6_N, skip_check: bool = False
    ) -> NDArray_N_3_1:
        v = cls._check_shapeb_and_valueb(v, skip_check=skip_check)
        return v[:, 3:]


@beartype
class Wrench(
    _LieAlgebra[
        NDArray_6_1 | NDArray_6,
        NDArray_6_1,
        NDArray_N_6_1 | NDArray_6_N,
        NDArray_N_6_1,
    ]
):
    _r = 6
    _c = 1

    @classmethod
    def angular_part_skewc(
        cls, v: NDArray_6_1 | NDArray_6, skip_check: bool = False
    ) -> NDArray_3_3:
        v = cls._check_shape_and_value(v, skip_check=skip_check)
        return utils.skew3(v[:3])

    @classmethod
    def angular_part_skewcb(
        cls, v: NDArray_N_6_1 | NDArray_6_N, skip_check: bool = False
    ) -> NDArray_N_3_3:
        v = cls._check_shapeb_and_valueb(v, skip_check=skip_check)
        return utils.skew3b(v[:, :3])

    @classmethod
    def angular_part_veec(
        cls, v: NDArray_6_1 | NDArray_6, skip_check: bool = False
    ) -> NDArray_3_1:
        v = cls._check_shape_and_value(v, skip_check=skip_check)
        return v[:3]

    @classmethod
    def angular_part_veecb(
        cls, v: NDArray_N_6_1 | NDArray_6_N, skip_check: bool = False
    ) -> NDArray_N_3_1:
        v = cls._check_shapeb_and_valueb(v, skip_check=skip_check)
        return v[:, :3]

    @classmethod
    def linear_part3c(
        cls, v: NDArray_6_1 | NDArray_6, skip_check: bool = False
    ) -> NDArray_3_1:
        v = cls._check_shape_and_value(v, skip_check=skip_check)
        return v[3:]

    @classmethod
    def linear_part3cb(
        cls, v: NDArray_N_6_1 | NDArray_6_N, skip_check: bool = False
    ) -> NDArray_N_3_1:
        v = cls._check_shapeb_and_valueb(v, skip_check=skip_check)
        return v[:, 3:]

    @classmethod
    def expc(
        cls,
        v: NDArray_6_1 | NDArray_6,
        intensity: RealScalar,
        expto: Literal["SE3", "Ad", "coAd"] = "SE3",
        normalize: bool = False,
        skip_check: bool = False,
    ) -> NDArray_4_4 | NDArray_6_6:
        raise NotImplementedError("The function expc is not implemented for Wrench.")

    @classmethod
    def expcb(
        cls,
        v: NDArray_N_6_1 | NDArray_6_N,
        intensity: NDArray_1D | RealScalar,
        expto: Literal["SE3", "Ad", "coAd"] = "SE3",
        normalize: bool = False,
        skip_check: bool = False,
    ) -> NDArray_N_4_4 | NDArray_N_6_6:
        raise NotImplementedError("The function expcb is not implemented for Wrench.")

    def exp(
        self,
        intensity: RealScalar,
        expto: Literal["SE3", "Ad", "coAd"] = "SE3",
        normalize: bool = False,
    ) -> Any:
        raise NotImplementedError("The function exp is not implemented for Wrench.")


@beartype
class ad(_LieAlgebra[NDArray_6_6, NDArray_6_6, NDArray_N_6_6, NDArray_N_6_6]):
    _r = 6
    _c = 6

    @classmethod
    def _check_value(cls, v: NDArray_6_6) -> NDArray_6_6:
        v = super()._check_value(v)
        if utils.check_skew6_matrix(v) is False:
            raise ValueError("v is not skew6 matrix.")
        return v

    @classmethod
    def _check_valueb(cls, v: NDArray_N_6_6) -> NDArray_N_6_6:
        v = super()._check_valueb(v)
        if utils.check_skew6_matrixb(v) is False:
            raise ValueError("v is not skew6 matrixb.")
        return v

    @classmethod
    def angular_part_skewc(
        cls, v: NDArray_6_6, skip_check: bool = False
    ) -> NDArray_3_3:
        v = cls._check_shape_and_value(v, skip_check=skip_check)
        return v[:3, :3]

    @classmethod
    def angular_part_skewcb(
        cls, v: NDArray_N_6_6, skip_check: bool = False
    ) -> NDArray_N_3_3:
        v = cls._check_shapeb_and_valueb(v, skip_check=skip_check)
        return v[:, :3, :3]

    @classmethod
    def angular_part_veec(cls, v: NDArray_6_6, skip_check: bool = False) -> NDArray_3_1:
        v = cls._check_shape_and_value(v, skip_check=skip_check)
        return utils.veeskew3(v[:3, :3])

    @classmethod
    def angular_part_veecb(
        cls, v: NDArray_N_6_6, skip_check: bool = False
    ) -> NDArray_N_3_1:
        v = cls._check_shapeb_and_valueb(v, skip_check=skip_check)
        return utils.veeskew3b(v[:, :3, :3])

    @classmethod
    def linear_part3c(cls, v: NDArray_6_6, skip_check: bool = False) -> NDArray_3_1:
        v = cls._check_shape_and_value(v, skip_check=skip_check)
        return utils.veeskew3(v[3:, :3])

    @classmethod
    def linear_part3cb(
        cls, v: NDArray_N_6_6, skip_check: bool = False
    ) -> NDArray_N_3_1:
        v = cls._check_shapeb_and_valueb(v, skip_check=skip_check)
        return utils.veeskew3b(v[:, 3:, :3])


@beartype
class coad(_LieAlgebra[NDArray_6_6, NDArray_6_6, NDArray_N_6_6, NDArray_N_6_6]):
    _r = 6
    _c = 6

    @classmethod
    def _check_value(cls, v: NDArray_6_6) -> NDArray_6_6:
        v = super()._check_value(v)
        if utils.check_coskew6_matrix(v) is False:
            raise ValueError("v is not coskew6 matrix.")
        return v

    @classmethod
    def _check_valueb(cls, v: NDArray_N_6_6) -> NDArray_N_6_6:
        v = super()._check_valueb(v)
        if utils.check_coskew6_matrixb(v) is False:
            raise ValueError("v is not coskew6 matrixb.")
        return v

    @classmethod
    def angular_part_skewc(
        cls, v: NDArray_6_6, skip_check: bool = False
    ) -> NDArray_3_3:
        v = cls._check_shape_and_value(v, skip_check=skip_check)
        return v[:3, :3]

    @classmethod
    def angular_part_skewcb(
        cls, v: NDArray_N_6_6, skip_check: bool = False
    ) -> NDArray_N_3_3:
        v = cls._check_shapeb_and_valueb(v, skip_check=skip_check)
        return v[:, :3, :3]

    @classmethod
    def angular_part_veec(cls, v: NDArray_6_6, skip_check: bool = False) -> NDArray_3_1:
        v = cls._check_shape_and_value(v, skip_check=skip_check)
        return utils.veeskew3(v[:3, :3])

    @classmethod
    def angular_part_veecb(
        cls, v: NDArray_N_6_6, skip_check: bool = False
    ) -> NDArray_N_3_1:
        v = cls._check_shapeb_and_valueb(v, skip_check=skip_check)
        return utils.veeskew3b(v[:, :3, :3])

    @classmethod
    def linear_part3c(cls, v: NDArray_6_6, skip_check: bool = False) -> NDArray_3_1:
        v = cls._check_shape_and_value(v, skip_check=skip_check)
        return utils.veeskew3(v[:3, 3:])

    @classmethod
    def linear_part3cb(
        cls, v: NDArray_N_6_6, skip_check: bool = False
    ) -> NDArray_N_3_1:
        v = cls._check_shapeb_and_valueb(v, skip_check=skip_check)
        return utils.veeskew3b(v[:, :3, 3:])


@beartype
class SE3(_LieGroup[NDArray_4_4, NDArray_4_4, NDArray_N_4_4, NDArray_N_4_4]):
    _r = 4
    _c = 4

    @classmethod
    def _check_value(cls, v: NDArray_4_4) -> NDArray_4_4:
        v = super()._check_value(v)
        if utils.check_rotation_matrix(v[:3, :3]) is False:
            raise ValueError("v[:3, :3] is not rotation matrix.")
        if utils.check_equal(v[3, :3], 0) is False:
            raise ValueError("v[3, :3] is not zero vector.")
        if utils.check_equal(v[3, 3], 1) is False:
            raise ValueError("v[3, 3] is not 1.")
        return v

    @classmethod
    def _check_valueb(cls, v: NDArray_N_4_4) -> NDArray_N_4_4:
        v = super()._check_valueb(v)
        if utils.check_rotation_matrixb(v) is False:
            raise ValueError("v[:, :3, :3] is not rotation matrixb.")
        if utils.check_equal(v[:, 3, :3], 0) is False:
            raise ValueError("v[:, 3, :3] is not zero vector.")
        if utils.check_equal(v[:, 3, 3], 1) is False:
            raise ValueError("v[:, 3, 3] is not 1.")
        return v

    @classmethod
    def invc(cls, v: NDArray_4_4, skip_check: bool = False) -> NDArray_4_4:
        v = cls._check_shape_and_value(v, skip_check=skip_check)
        v_inv = np.eye(4)
        v_inv[:3, :3] = v[:3, :3].T
        v_inv[:3, 3:] = -v[:3, :3].T @ v[:3, 3:]
        return v_inv

    @classmethod
    def invcb(cls, v: NDArray_N_4_4, skip_check: bool = False) -> NDArray_N_4_4:
        v = cls._check_shapeb_and_valueb(v, skip_check=skip_check)
        v_inv = np.zeros(v.shape)
        v_inv[:, :3, :3] = v[:, :3, :3].transpose(0, 2, 1)
        v_inv[:, :3, 3:] = -v[:, :3, :3].transpose(0, 2, 1) @ v[:, :3, 3:]
        return v_inv

    @classmethod
    def Rc(cls, v: NDArray_4_4, skip_check: bool = False) -> NDArray_3_3:
        v = cls._check_shape_and_value(v, skip_check=skip_check)
        return v[:3, :3]

    @classmethod
    def Rcb(cls, v: NDArray_N_4_4, skip_check: bool = False) -> NDArray_N_3_3:
        v = cls._check_shapeb_and_valueb(v, skip_check=skip_check)
        return v[:, :3, :3]

    @classmethod
    def t3c(cls, v: NDArray_4_4, skip_check: bool = False) -> NDArray_3_1:
        v = cls._check_shape_and_value(v, skip_check=skip_check)
        return v[:3, 3:]

    @classmethod
    def t3cb(cls, v: NDArray_N_4_4, skip_check: bool = False) -> NDArray_N_3_1:
        v = cls._check_shapeb_and_valueb(v, skip_check=skip_check)
        return v[:, :3, 3:]

    @classmethod
    def _rot_basec(cls, axis: Literal["x", "y", "z"], theta: RealScalar) -> NDArray_4_4:
        return utils.rot4c(axis, theta)

    @classmethod
    def _rot_basecb(
        cls, axis: Literal["x", "y", "z"], theta: NDArray_1D | RealScalar
    ) -> NDArray_N_4_4:
        return utils.rot4cb(axis, theta)

    @classmethod
    def _trans_basec(
        cls, axis: Literal["x", "y", "z"], theta: RealScalar
    ) -> NDArray_4_4:
        v = np.eye(4)
        if axis == "x":
            v[0, -1] = theta
        elif axis == "y":
            v[1, -1] = theta
        else:
            v[2, -1] = theta
        return v

    @classmethod
    def _trans_basecb(
        cls, axis: Literal["x", "y", "z"], theta: NDArray_1D | RealScalar
    ) -> NDArray_N_4_4:
        if utils.is_realscaler(theta):
            num = 1
        else:
            theta = cast(NDArray_1D, theta)
            num = theta.shape[0]
        v = np.zeros((num, 4, 4))
        v[:] = np.eye(4)
        if axis == "x":
            v[:, 0, -1] = theta
        elif axis == "y":
            v[:, 1, -1] = theta
        else:
            v[:, 2, -1] = theta
        return v


@beartype
class Ad(_LieGroup[NDArray_6_6, NDArray_6_6, NDArray_N_6_6, NDArray_N_6_6]):
    _r = 6
    _c = 6

    @classmethod
    def _check_value(cls, v: NDArray_6_6) -> NDArray_6_6:
        v = super()._check_value(v)
        if utils.check_rotation_matrix(v[:3, :3]) is False:
            raise ValueError("v[:3, :3] is not rotation matrix")
        if utils.check_equal(v[:3, :3], v[3:, 3:]) is False:
            raise ValueError("v[:3,:3] != v[3:,3:]")
        skew_t = v[3:, :3] @ v[:3, :3].T
        if utils.check_skew3_matrix(skew_t) is False:
            raise ValueError("v[3:,:3]@v[:3,:3].T is not skew3 matrix")
        return v

    @classmethod
    def _check_valueb(cls, v: NDArray_N_6_6) -> NDArray_N_6_6:
        v = super()._check_valueb(v)
        if utils.check_rotation_matrixb(v[:, :3, :3]) is False:
            raise ValueError("v[:, :3, :3] is not rotation matrixb")
        if utils.check_equal(v[:, :3, :3], v[:, 3:, 3:]) is False:
            raise ValueError("v[:, :3, :3] != v[:, 3:, 3:]")
        skew_tb = v[:, 3:, :3] @ v[:, :3, :3].transpose(0, 2, 1)
        if utils.check_skew3_matrixb(skew_tb) is False:
            raise ValueError("v[:, 3:, :3]@v[:, :3, :3].T is not skew3 matrixb")
        return v

    @classmethod
    def invc(cls, v: NDArray_6_6, skip_check: bool = False) -> NDArray_6_6:
        v = cls._check_shape_and_value(v, skip_check=skip_check)
        v_inv = np.zeros(v.shape)
        v_inv[:3, :3] = v[:3, :3].T
        v_inv[3:, 3:] = v[:3, :3].T
        v_inv[3:, :3] = v[3:, :3].T
        return v_inv

    @classmethod
    def invcb(cls, v: NDArray_N_6_6, skip_check: bool = False) -> NDArray_N_6_6:
        v = cls._check_shapeb_and_valueb(v, skip_check=skip_check)
        v_inv = np.zeros(v.shape)
        v_inv[:, :3, :3] = v[:, :3, :3].transpose(0, 2, 1)
        v_inv[:, 3:, 3:] = v[:, :3, :3].transpose(0, 2, 1)
        v_inv[:, 3:, :3] = v[:, 3:, :3].transpose(0, 2, 1)
        return v_inv

    @classmethod
    def Rc(cls, v: NDArray_6_6, skip_check: bool = False) -> NDArray_3_3:
        v = cls._check_shape_and_value(v, skip_check=skip_check)
        return v[:3, :3]

    @classmethod
    def Rcb(cls, v: NDArray_N_6_6, skip_check: bool = False) -> NDArray_N_3_3:
        v = cls._check_shapeb_and_valueb(v, skip_check=skip_check)
        return v[:, :3, :3]

    @classmethod
    def t3c(cls, v: NDArray_6_6, skip_check: bool = False) -> NDArray_3_1:
        v = cls._check_shape_and_value(v, skip_check=skip_check)
        skew_t = v[3:, :3] @ v[:3, :3].T
        return utils.veeskew3(skew_t)

    @classmethod
    def t3cb(cls, v: NDArray_N_6_6, skip_check: bool = False) -> NDArray_N_3_1:
        v = cls._check_shapeb_and_valueb(v, skip_check=skip_check)
        skew_tb = v[:, 3:, :3] @ v[:, :3, :3].transpose(0, 2, 1)
        return utils.veeskew3b(skew_tb)

    @classmethod
    def _rot_basec(cls, axis: Literal["x", "y", "z"], theta: RealScalar) -> NDArray_6_6:
        return utils.rot6c(axis, theta)

    @classmethod
    def _rot_basecb(
        cls, axis: Literal["x", "y", "z"], theta: NDArray_1D | RealScalar
    ) -> NDArray_N_6_6:
        return utils.rot6cb(axis, theta)

    @classmethod
    def _trans_basec(
        cls, axis: Literal["x", "y", "z"], theta: RealScalar
    ) -> NDArray_6_6:
        v = np.eye(6)
        t = np.zeros((3, 1))
        if axis == "x":
            t[0] = theta
        elif axis == "y":
            t[1] = theta
        else:
            t[2] = theta
        v[3:, :3] = utils.skew3(t)
        return v

    @classmethod
    def _trans_basecb(
        cls, axis: Literal["x", "y", "z"], theta: NDArray_1D | RealScalar
    ) -> NDArray_N_6_6:
        if utils.is_realscaler(theta):
            num = 1
        else:
            theta = cast(NDArray_1D, theta)
            num = theta.shape[0]
        v = np.zeros((num, 6, 6))
        v[:] = np.eye(6)
        tb = np.zeros((num, 3, 1))
        if axis == "x":
            tb[:, 0, 0] = theta
        elif axis == "y":
            tb[:, 1, 0] = theta
        else:
            tb[:, 2, 0] = theta
        v[:, 3:, :3] = utils.skew3b(tb)
        return v


@beartype
class coAd(_LieGroup[NDArray_6_6, NDArray_6_6, NDArray_N_6_6, NDArray_N_6_6]):
    _r = 6
    _c = 6

    @classmethod
    def _check_value(cls, v: NDArray_6_6) -> NDArray_6_6:
        v = super()._check_value(v)
        if utils.check_rotation_matrix(v[:3, :3]) is False:
            raise ValueError("v[:3, :3] is not rotation matrix")
        if utils.check_equal(v[:3, :3], v[3:, 3:]) is False:
            raise ValueError("v[:3,:3] != v[3:,3:]")
        skew_t = v[:3, 3:] @ v[:3, :3].T
        if utils.check_skew3_matrix(skew_t) is False:
            raise ValueError("v[:3,3:]@v[:3,:3].T is not skew3 matrix")
        return v

    @classmethod
    def _check_valueb(cls, v: NDArray_N_6_6) -> NDArray_N_6_6:
        v = super()._check_valueb(v)
        if utils.check_rotation_matrixb(v[:, :3, :3]) is False:
            raise ValueError("v[:, :3, :3] is not rotation matrixb")
        if utils.check_equal(v[:, :3, :3], v[:, 3:, 3:]) is False:
            raise ValueError("v[:, :3, :3] != v[:, 3:, 3:]")
        skew_tb = v[:, :3, 3:] @ v[:, :3, :3].transpose(0, 2, 1)
        if utils.check_skew3_matrixb(skew_tb) is False:
            raise ValueError("v[:, :3, 3:]@v[:, :3, :3].T is not skew3 matrixb")
        return v

    @classmethod
    def invc(cls, v: NDArray_6_6, skip_check: bool = False) -> NDArray_6_6:
        v = cls._check_shape_and_value(v, skip_check=skip_check)
        v_inv = np.zeros(v.shape)
        v_inv[:3, :3] = v[:3, :3].T
        v_inv[3:, 3:] = v[:3, :3].T
        v_inv[:3, 3:] = v[:3, 3:].T
        return v_inv

    @classmethod
    def invcb(cls, v: NDArray_N_6_6, skip_check: bool = False) -> NDArray_N_6_6:
        v = cls._check_shapeb_and_valueb(v, skip_check=skip_check)
        v_inv = np.zeros(v.shape)
        v_inv[:, :3, :3] = v[:, :3, :3].transpose(0, 2, 1)
        v_inv[:, 3:, 3:] = v[:, :3, :3].transpose(0, 2, 1)
        v_inv[:, :3, 3:] = v[:, :3, 3:].transpose(0, 2, 1)
        return v_inv

    @classmethod
    def Rc(cls, v: NDArray_6_6, skip_check: bool = False) -> NDArray_3_3:
        v = cls._check_shape_and_value(v, skip_check=skip_check)
        return v[:3, :3]

    @classmethod
    def Rcb(cls, v: NDArray_N_6_6, skip_check: bool = False) -> NDArray_N_3_3:
        v = cls._check_shapeb_and_valueb(v, skip_check=skip_check)
        return v[:, :3, :3]

    @classmethod
    def t3c(cls, v: NDArray_6_6, skip_check: bool = False) -> NDArray_3_1:
        v = cls._check_shape_and_value(v, skip_check=skip_check)
        skew_t = v[:3, 3:] @ v[:3, :3].T
        return utils.veeskew3(skew_t)

    @classmethod
    def t3cb(cls, v: NDArray_N_6_6, skip_check: bool = False) -> NDArray_N_3_1:
        v = cls._check_shapeb_and_valueb(v, skip_check=skip_check)
        skew_tb = v[:, :3, 3:] @ v[:, :3, :3].transpose(0, 2, 1)
        return utils.veeskew3b(skew_tb)

    @classmethod
    def _rot_basec(cls, axis: Literal["x", "y", "z"], theta: RealScalar) -> NDArray_6_6:
        return utils.rot6c(axis, theta)

    @classmethod
    def _rot_basecb(
        cls, axis: Literal["x", "y", "z"], theta: NDArray_1D | RealScalar
    ) -> NDArray_N_6_6:
        return utils.rot6cb(axis, theta)

    @classmethod
    def _trans_basec(
        cls, axis: Literal["x", "y", "z"], theta: RealScalar
    ) -> NDArray_6_6:
        v = np.eye(6)
        t = np.zeros((3, 1))
        if axis == "x":
            t[0] = theta
        elif axis == "y":
            t[1] = theta
        else:
            t[2] = theta
        v[:3, 3:] = utils.skew3(t)
        return v

    @classmethod
    def _trans_basecb(
        cls, axis: Literal["x", "y", "z"], theta: NDArray_1D | RealScalar
    ) -> NDArray_N_6_6:
        if utils.is_realscaler(theta):
            num = 1
        else:
            theta = cast(NDArray_1D, theta)
            num = theta.shape[0]
        v = np.zeros((num, 6, 6))
        v[:] = np.eye(6)
        tb = np.zeros((num, 3, 1))
        if axis == "x":
            tb[:, 0, 0] = theta
        elif axis == "y":
            tb[:, 1, 0] = theta
        else:
            tb[:, 2, 0] = theta
        v[:, :3, 3:] = utils.skew3b(tb)
        return v
