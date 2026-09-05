from __future__ import annotations

from types import NotImplementedType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import lieink.atoms

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import (
    Any,
    Literal,
    Self,
    TypeAliasType,
    cast,
    overload,
)

import numpy as np
from beartype import beartype
from rich import print
from rich.pretty import pprint

from lieink import utils
from lieink.annotations import (
    NDArray,
    NDArray_1D,
    NDArray_2D,
    NDArray_3,
    NDArray_3_1,
    NDArray_3_3,
    NDArray_3_N,
    NDArray_3D,
    NDArray_4,
    NDArray_4_1,
    NDArray_4_4,
    NDArray_4_N,
    NDArray_6_1,
    NDArray_6_6,
    NDArray_N_3_1,
    NDArray_N_3_3,
    NDArray_N_4_1,
    NDArray_N_4_4,
    NDArray_N_6_1,
    NDArray_N_6_6,
    RealScalar,
)


@beartype
class BasicLie[
    allowed_input_type_single: NDArray,
    output_type_single: NDArray,
    allowed_input_type_batch: NDArray,
    output_type_batch: NDArray,
]:
    _r: int | None = None
    _c: int | None = None
    _v: output_type_single | None

    @classmethod
    def PRINT_ALLOWED_OPERATORS(cls):
        pprint(ALLOWED_OPERATORS)

    @classmethod
    def UPDATE_ALLOWED_OPERATORS(
        cls,
        object_type_1: type[Any] | TypeAliasType,
        operator: str,
        object_type_2: type[Any] | TypeAliasType | None,
        result_type: type[Any] | TypeAliasType | dict[str, type[Any] | TypeAliasType],
    ):

        operator_dict = ALLOWED_OPERATORS.get(operator)
        if operator_dict is None:
            ALLOWED_OPERATORS[operator] = {object_type_1: {object_type_2: result_type}}
        else:
            allowed_type = operator_dict.get(object_type_1)
            if allowed_type is None:
                operator_dict[object_type_1] = {object_type_2: result_type}
            else:
                result_type_ = allowed_type.get(object_type_2)
                if result_type_ is None:
                    allowed_type[object_type_2] = result_type
                else:
                    if type(result_type_) is dict and type(result_type) is dict:
                        result_type_.update(result_type)
                    elif type(result_type_) is dict and type(result_type) is not dict:
                        raise ValueError(
                            f"The result type for operator {operator} is already defined for type {utils.get_TypeName(object_type_1)} and {utils.get_TypeName(object_type_2)} as {result_type_}. If you want to change it, you need to use a form of dict, whose form is like match_str:{utils.get_TypeName(result_type)}."
                        )
                    elif type(result_type_) is not dict and type(result_type) is dict:
                        keys = list(result_type.values())
                        if result_type_ not in keys:
                            raise ValueError(
                                f"The result type for operator {operator} is already defined for type {utils.get_TypeName(object_type_1)} and {utils.get_TypeName(object_type_2)} as {utils.get_TypeName(result_type_)}, if you want to change it, you need to use a form of dict, whose value includes {utils.get_TypeName(result_type_)}."
                            )
                        allowed_type[object_type_2] = result_type
                    else:
                        raise ValueError(
                            f"The result type for operator {operator} is already defined for type {utils.get_TypeName(object_type_1)} and {utils.get_TypeName(object_type_2)} as {utils.get_TypeName(result_type_)}, if you want to change it, you need to use a form of dict, whose value includes {utils.get_TypeName(result_type_)}."
                        )

    @classmethod
    def _check_operator(
        cls, operator: str, other: object
    ) -> (
        type[Any]
        | TypeAliasType
        | dict[str, type[Any] | TypeAliasType]
        | NotImplementedType
    ):

        other_type = utils.get_Type(other)
        if cls._type() == Lie or other_type == Lie:
            return Lie

        operator_dict = ALLOWED_OPERATORS.get(operator)
        if operator_dict is None:
            raise TypeError(f"operator {operator} is not allowed")
        my_type = cls._type()
        allowed_types = operator_dict.get(my_type)
        if allowed_types is None:
            return NotImplemented

        result_type = allowed_types.get(other_type)

        if result_type is None:
            return NotImplemented

        return result_type

    @property
    def r(self) -> int | None:
        if self._v is not None:
            return self._v.shape[0]
        else:
            return None

    @property
    def c(self) -> int | None:
        if self._v is not None:
            return self._v.shape[1]
        else:
            return None

    def __add__(self, other: Any) -> BasicLie[Any, Any, Any, Any] | NotImplementedType:
        self_v = self._check_assigned()
        other_v = other._check_assigned()
        return_type = self._check_operator("+", other)
        if return_type is NotImplemented:
            return return_type
        return self._add_impl(self_v, other_v, return_type)  # type: ignore

    def _add_impl(
        self, self_v: NDArray, other_v: NDArray, return_type: type
    ) -> BasicLie[Any, Any, Any, Any]:
        new_v = self_v + other_v
        return return_type(new_v)  # type: ignore

    def __sub__(self, other: Any) -> BasicLie[Any, Any, Any, Any] | NotImplementedType:
        return self + (-1 * other)

    def __mul__(self, other: Any) -> BasicLie[Any, Any, Any, Any] | NotImplementedType:
        self_v = self._check_assigned()
        other_v = other._check_assigned()
        return_type = self._check_operator("*", other)
        if return_type is NotImplemented:
            return return_type
        return self._mul_impl(self_v, other_v, return_type)  # type: ignore

    def _mul_impl(
        self, self_v: NDArray, other_v: NDArray, return_type: type
    ) -> BasicLie[Any, Any, Any, Any]:
        new_v = self_v @ other_v
        return return_type(new_v)

    def __rmul__(
        self, other: RealScalar
    ) -> BasicLie[Any, Any, Any, Any] | NotImplementedType:
        self_v = self._check_assigned()
        return_type = self._check_operator("*", other)
        if return_type is NotImplemented:
            return return_type
        return self._rmul_impl(self_v, other, return_type)  # type: ignore

    def _rmul_impl(
        self, self_v: NDArray, other_v: RealScalar, return_type: type
    ) -> BasicLie[Any, Any, Any, Any] | NotImplementedType:
        new_v = other_v * self_v
        return return_type(new_v)

    def __matmul__(
        self, other: Any
    ) -> BasicLie[Any, Any, Any, Any] | NotImplementedType:
        self_v = self._check_assigned()
        other_v = other._check_assigned()
        return_type = self._check_operator("@", other)
        if return_type is NotImplemented:
            return return_type
        return self._matmul_impl(self_v, other_v, return_type)  # type: ignore

    def _matmul_impl(
        self, self_v: NDArray, other_v: NDArray, return_type: type
    ) -> BasicLie[Any, Any, Any, Any]:
        if utils.check_symmetry(other_v) is not False:
            new_v = self_v @ other_v @ self_v.T
        else:
            new_v = self_v @ other_v @ self.invc(self_v, skip_check=True)  # type: ignore
        return return_type(new_v)

    @property
    def shape(self) -> list[int | None]:
        return [self.r, self.c]

    @classmethod
    def _type(cls) -> type[Self]:
        return cls

    @property
    def mytype(self) -> type[Self]:
        return self._type()

    @property
    def v(self) -> output_type_single:
        v = self._check_assigned()
        return v  # type: ignore

    def copy(self) -> Self:
        new_one = self._type()(self._v.copy())  # type: ignore
        return new_one

    @classmethod
    def _check_shape(cls, v: allowed_input_type_single) -> allowed_input_type_single:

        if v.ndim == 1:
            v = v.reshape(-1, 1)  # type: ignore

        r, c = v.shape
        if cls._r is None or cls._c is None:
            return v
        elif r != cls._r or c != cls._c:
            raise ValueError(f"v must have shape ({cls._r},{cls._c}), now ({r},{c}).")
        return v

    @classmethod
    def _check_shapeb(cls, v: allowed_input_type_batch) -> allowed_input_type_batch:

        return cls._reshapeb(v)

    @classmethod
    def _check_value(cls, v: allowed_input_type_single) -> allowed_input_type_single:
        return v.astype(np.float64)  # type: ignore

    @classmethod
    def _check_valueb(cls, v: allowed_input_type_batch) -> allowed_input_type_batch:
        return v.astype(np.float64)  # type: ignore

    @classmethod
    def _check_shape_and_value(
        cls, v: allowed_input_type_single, skip_check: bool = False
    ) -> allowed_input_type_single:
        if skip_check is False:
            v = cls._check_shape(v)
            v = cls._check_value(v)
        return v.copy()

    @classmethod
    def _check_shapeb_and_valueb(
        cls, v: allowed_input_type_batch, skip_check: bool = False
    ) -> allowed_input_type_batch:
        if skip_check is False:
            v = cls._check_shapeb(v)
            v = cls._check_valueb(v)
        return v.copy()

    @classmethod
    def _reshapeb(cls, v: allowed_input_type_batch) -> allowed_input_type_batch:

        if cls._r is None or cls._c is None:
            return v

        if v.ndim == 1:
            if v.shape[0] != cls._r:
                raise ValueError(
                    f"v must have shape ({cls._r}, 1), now ({v.shape[0]}, 1)."
                )
            v_reshaped = v.reshape(1, cls._r, cls._c)
            return v_reshaped  # type: ignore
        elif v.ndim == 2:
            if cls._c == 1:
                if v.shape[0] != cls._r:
                    raise ValueError(
                        f"v must have shape ({cls._r}, 1), now ({v.shape[0]}, 1)."
                    )
                v_reshaped = v.reshape(cls._r, v.shape[1], 1).swapaxes(0, 1)
                return v_reshaped  # type: ignore
            else:
                raise ValueError(
                    f"2D input is strictly forbidden for objects with _c={cls._c}. "
                    f"Please provide a 3D batch array of shape (N, {cls._r}, {cls._c})."
                )
        else:
            if v.shape[1] != cls._r or v.shape[2] != cls._c:
                raise ValueError(
                    f"v must have shape (n, {cls._r}, {cls._c}), now (n, {v.shape[1]}, {v.shape[2]})."
                )
            return v

    @classmethod
    def reshapeb(cls, v: allowed_input_type_batch) -> output_type_batch:
        return cls._reshapeb(v)  # type: ignore

    def __init__(
        self, v: allowed_input_type_single | BasicLie[Any, Any, Any, Any] | None = None
    ) -> None:
        self._v = None
        if v is not None:
            if isinstance(v, BasicLie):
                self.assign(v.v)  # type: ignore
            else:
                self.assign(v)

    @overload
    def _check_assigned(
        self, dont_raise: Literal[False] = False
    ) -> allowed_input_type_single: ...
    @overload
    def _check_assigned(
        self, dont_raise: Literal[True]
    ) -> allowed_input_type_single | Literal[False]: ...

    def _check_assigned(
        self, dont_raise: bool = False
    ) -> allowed_input_type_single | Literal[False]:
        if self._v is None:
            if not dont_raise:
                raise ValueError(f"v of {self.mytype.__name__} is not assigned yet")
            else:
                return False
        return self._v  # type: ignore

    def assign(self, v: allowed_input_type_single) -> Self:
        self_v = self._check_assigned(dont_raise=True)
        if self_v is not False:
            raise ValueError(f"v of {self.mytype.__name__} has been assigned already")
        self._v = self._check_shape_and_value(v)  # type: ignore
        return self

    def _inplace_or_not(self, v: output_type_single, inplace: bool) -> Self:
        if inplace is False:
            return self.mytype(v)  # type: ignore
        else:
            self._v = v
            return self

    def __repr__(self) -> str:
        v = self._v
        name = self.mytype.__name__
        if v is None:
            return f"{name}:({v})\n"
        else:
            if self._c == 1:
                v = v.flatten()
            v = np.array2string(
                v, precision=4, separator=", ", suppress_small=True, prefix=f"{name}("
            )
            return f"{name}({v})"

    def print(self) -> None:
        print(self)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.mytype):
            raise TypeError(
                f"other must be an instance of {self.mytype.__name__}, instead {type(other).__name__}"
            )
        if self.mytype is not other.mytype:
            raise TypeError("self and other must be of the same type")
        self_v = self._check_assigned()
        other_v = other._check_assigned()

        return utils.check_equal(self_v, other_v) is not False

    @classmethod
    def _repeat(
        cls,
        v1: NDArray_3D | NDArray_2D | NDArray_1D,
        v2: NDArray_3D | NDArray_2D | NDArray_1D,
    ) -> tuple[NDArray_3D, NDArray_3D]:

        if v1.ndim == 2:
            v1 = v1.reshape(1, v1.shape[0], v1.shape[1])
        elif v1.ndim == 1:
            v1 = v1.reshape(1, v1.shape[0], 1)
        else:
            pass

        if v2.ndim == 2:
            v2 = v2.reshape(1, v2.shape[0], v2.shape[1])
        elif v2.ndim == 1:
            v2 = v2.reshape(1, v2.shape[0], 1)
        else:
            pass

        num1 = v1.shape[0]
        num2 = v2.shape[0]

        if num1 == num2:
            pass
        elif num1 == 1 and num2 != 1:
            v1 = v1.repeat(num2, axis=0)
        elif num1 != 1 and num2 == 1:
            v2 = v2.repeat(num1, axis=0)
        else:
            raise ValueError(
                f"The shape of v1 {v1.shape} and v2 {v2.shape} are not compatible."
            )

        return v1, v2

    @classmethod
    def addcb(
        cls,
        v1: NDArray_3D | NDArray_2D | NDArray_1D,
        v2: NDArray_3D | NDArray_2D | NDArray_1D,
    ) -> NDArray_3D:
        v1, v2 = cls._repeat(v1, v2)
        return v1 + v2

    @classmethod
    def subcb(
        cls,
        v1: NDArray_3D | NDArray_2D | NDArray_1D,
        v2: NDArray_3D | NDArray_2D | NDArray_1D,
    ) -> NDArray_3D:
        v1, v2 = cls._repeat(v1, v2)
        return v1 - v2

    @classmethod
    def mulcb(
        cls,
        v1: NDArray_3D | NDArray_1D | RealScalar,
        v2: NDArray_3D | NDArray_1D | RealScalar,
    ) -> NDArray_3D:

        scaler = False

        if isinstance(v1, (float, int, np.floating, np.integer)):
            v1 = np.array([[[v1]]])
            scaler = True
        else:
            if v1.ndim == 3:
                pass
            else:
                v1 = v1.reshape(-1, 1, 1)
                scaler = True

        if isinstance(v2, (float, int, np.floating, np.integer)):
            v2 = np.array([[[v2]]])
            scaler = True
        else:
            if v2.ndim == 3:
                pass
            else:
                v2 = v2.reshape(-1, 1, 1)
                scaler = True

        v1, v2 = cls._repeat(v1, v2)

        if scaler:
            return v1 * v2
        return v1 @ v2

    @classmethod
    def adjointcb(
        cls,
        v1: NDArray_3D | NDArray_2D,
        v2: NDArray_3D | NDArray_2D,
    ) -> NDArray_3D:

        if v1.shape[-1] != v1.shape[-2]:
            raise ValueError(
                "v1 must be a square matrix or the last two dimensions must be equal."
            )

        if v2.shape[-1] != v2.shape[-2]:
            raise ValueError(
                "v2 must be a square matrix or the last two dimensions must be equal."
            )

        v1, v2 = cls._repeat(v1, v2)
        if utils.check_symmetryb(v2) is not False:
            v1_inv = v1.transpose(0, 2, 1)
        else:
            v1_inv = np.linalg.inv(v1)

        return v1 @ v2 @ v1_inv

    @property
    def T(self) -> Lie:
        v = self._check_assigned()
        return Lie(v.T)


@beartype
class Lie(
    BasicLie[
        NDArray,
        NDArray,
        NDArray,
        NDArray,
    ]
):
    pass


@beartype
class BasicVector[
    allowed_input_type_single: NDArray,
    output_type_single: NDArray,
    allowed_input_type_batch: NDArray,
    output_type_batch: NDArray,
](
    BasicLie[
        allowed_input_type_single,
        output_type_single,
        allowed_input_type_batch,
        output_type_batch,
    ],
    ABC,
):
    def __init__(
        self,
        v: allowed_input_type_single | BasicLie[Any, Any, Any, Any] | None = None,
        normalize: bool = False,
    ) -> None:
        super().__init__(v)
        if normalize:
            self.normalize(True)

    @classmethod
    def lengthc(
        cls,
        v: allowed_input_type_single,
        skip_check: bool = False,
    ) -> RealScalar:

        if not skip_check:
            v = cls._check_shape_and_value(v)
        return np.linalg.norm(v)

    @classmethod
    def lengthcb(
        cls, v: allowed_input_type_batch, skip_check: bool = False
    ) -> NDArray_1D:

        if not skip_check:
            v = cls._check_shapeb_and_valueb(v)
        return np.linalg.norm(v, axis=1).flatten()

    @property
    def length(self) -> RealScalar:
        self_v = self._check_assigned()
        return self.lengthc(self_v, skip_check=True)

    @classmethod
    def normalizec(
        cls, v: allowed_input_type_single, skip_check: bool = False
    ) -> output_type_single:
        v = cls._check_shape_and_value(v, skip_check=skip_check)
        length = cls.lengthc(v, skip_check=True)
        if utils.check_equal(length, 0) is not False:
            return v  # type: ignore
        else:
            return v / length  # type: ignore

    @classmethod
    def normalizecb(
        cls, v: allowed_input_type_batch, skip_check: bool = False
    ) -> output_type_batch:
        v = cls._check_shapeb_and_valueb(v, skip_check=skip_check)
        lengthb = cls.lengthcb(v, skip_check=True)

        mask = utils.find_zeros(lengthb)
        mask_ = np.logical_not(mask)
        if mask_.any():
            v[mask_] = v[mask_] / lengthb[mask_].reshape(-1, 1, 1)
        return v  # type: ignore

    def normalize(self, inplace: bool = False) -> Self:
        self_v = self._check_assigned()
        self_v = self.normalizec(self_v, skip_check=True)
        return self._inplace_or_not(self_v, inplace)


@beartype
class BasicPoint[
    allowed_input_type_single: NDArray,
    output_type_single: NDArray,
    allowed_input_type_batch: NDArray,
    output_type_batch: NDArray,
](
    BasicLie[
        allowed_input_type_single,
        output_type_single,
        allowed_input_type_batch,
        output_type_batch,
    ],
    ABC,
):
    @classmethod
    def lengthc(
        cls,
        v: allowed_input_type_single,
        skip_check: bool = False,
    ) -> RealScalar:

        if not skip_check:
            v = cls._check_shape_and_value(v)
        return np.linalg.norm(v[:3]) / cls.scalec(v, skip_check=True)

    @classmethod
    def lengthcb(
        cls, v: allowed_input_type_batch, skip_check: bool = False
    ) -> NDArray_1D:

        if not skip_check:
            v = cls._check_shapeb_and_valueb(v)
        return np.linalg.norm(v[:, :3], axis=1).flatten() / cls.scalecb(
            v, skip_check=True
        )

    @property
    def length(self) -> RealScalar:
        self_v = self._check_assigned()
        return self.lengthc(self_v, skip_check=True)

    @classmethod
    @abstractmethod
    def scalec(
        cls,
        v: allowed_input_type_single,
        skip_check: bool = False,
    ) -> RealScalar:
        pass

    @classmethod
    @abstractmethod
    def scalecb(
        cls, v: allowed_input_type_batch, skip_check: bool = False
    ) -> NDArray_1D:
        pass

    @property
    def scale(self) -> RealScalar:
        self_v = self._check_assigned()
        return self.scalec(self_v, skip_check=True)

    @classmethod
    def normalizec(
        cls, v: allowed_input_type_single, skip_check: bool = False
    ) -> output_type_single:
        v = cls._check_shape_and_value(v, skip_check=skip_check)
        return v / cls.scalec(v, skip_check=True)  # type: ignore

    @classmethod
    def normalizecb(
        cls, v: allowed_input_type_batch, skip_check: bool = False
    ) -> output_type_batch:
        v = cls._check_shapeb_and_valueb(v, skip_check=skip_check)
        return v / cls.scalecb(v, skip_check=True).reshape(-1, 1, 1)  # type: ignore

    def normalize(self, inplace: bool = False) -> Self:
        self_v = self._check_assigned()
        self_v = self.normalizec(self_v, skip_check=True)
        return self._inplace_or_not(self_v, inplace)

    @classmethod
    def init_pointc(cls) -> output_type_single:
        if cls._r == 4:
            return np.array([0.0, 0.0, 0.0, 1.0]).reshape(4, 1)  # type: ignore
        else:
            return np.array([0.0, 0.0, 0.0]).reshape(3, 1)  # type: ignore

    @classmethod
    def init_pointcb(cls, num: int) -> output_type_batch:
        if cls._r == 4:
            v = np.zeros((num, 4, 1))
            v[:, -1] = 1.0
        else:
            v = np.zeros((num, 3, 1))
        return v  # type: ignore

    def init_point(self) -> Self:
        self._v = self.init_pointc()
        return self

    @classmethod
    def _transAxisc(
        cls,
        axis: Literal["x", "y", "z"],
        theta: RealScalar,
        v: allowed_input_type_single | None = None,
        skip_check: bool = False,
    ) -> allowed_input_type_single:

        if v is None:
            v = cls.init_pointc()  # type: ignore
        else:
            v = cls._check_shape_and_value(v, skip_check=skip_check)

        if axis == "x":
            v[0] += theta * cls.scalec(v, skip_check=True)  # type: ignore
        elif axis == "y":
            v[1] += theta * cls.scalec(v, skip_check=True)  # type: ignore
        elif axis == "z":
            v[2] += theta * cls.scalec(v, skip_check=True)  # type: ignore

        return v  # type: ignore

    @classmethod
    def _transAxiscb(
        cls,
        axis: Literal["x", "y", "z"],
        theta: NDArray_1D | RealScalar,
        v: allowed_input_type_batch | None = None,
        skip_check: bool = False,
    ) -> allowed_input_type_batch:

        if utils.is_realscaler(theta) is not False:
            num_theta = 1
        else:
            num_theta = cast(int, theta.shape[0])  # type: ignore

        if v is not None:
            v = cls._check_shapeb_and_valueb(v, skip_check=skip_check)
            if num_theta != v.shape[0]:
                raise ValueError("The 1-D vector theta must be of the same num as v")
        else:
            v = cls.init_pointcb(num_theta)  # type: ignore

        if axis == "x":
            v[:, 0, 0] = theta * cls.scalecb(v, skip_check=True)  # type: ignore
        elif axis == "y":
            v[:, 1, 0] = theta * cls.scalecb(v, skip_check=True)  # type: ignore
        elif axis == "z":
            v[:, 2, 0] = theta * cls.scalecb(v, skip_check=True)  # type: ignore

        return v  # type: ignore

    @classmethod
    def _transSeriesc(
        cls,
        axes: Iterable[Literal["x", "y", "z"]],
        thetas: NDArray_1D | RealScalar,
        v: allowed_input_type_single | None = None,
        skip_check: bool = False,
    ) -> allowed_input_type_single:

        num_axis = len(axes)  # type: ignore
        if utils.is_realscaler(thetas) is not False:
            thetas = np.array([thetas] * num_axis)
        else:
            thetas = cast(NDArray_1D, thetas)
            if thetas.shape[0] != num_axis:
                raise ValueError("thetas and axes must have the same length")
        if v is None:
            v = cls.init_liegroupc()  # type: ignore
        else:
            v = cls._check_shape_and_value(v, skip_check=skip_check)

        for i in range(num_axis):
            v = cls._transAxisc(axes[i], thetas[i], v, skip_check=True)  # type: ignore

        return v  # type: ignore

    @classmethod
    def _transSeriescb(
        cls,
        axes: Iterable[Literal["x", "y", "z"]],
        thetas: NDArray_1D | RealScalar,
        v: allowed_input_type_single | None = None,
        skip_check: bool = False,
    ) -> allowed_input_type_single:

        num_axis = len(axes)  # type: ignore
        if utils.is_realscaler(thetas) is not False:
            thetas = np.array([thetas] * num_axis)
        else:
            thetas = cast(NDArray_1D, thetas)
            if thetas.shape[0] != num_axis:
                raise ValueError("thetas and axes must have the same length")
        if v is None:
            v = cls.init_liegroupc()  # type: ignore
        else:
            v = cls._check_shapeb_and_valueb(v, skip_check=skip_check)  # type: ignore

        for i in range(num_axis):
            v = cls._transAxiscb(axes[i], thetas[i], v, skip_check=True)  # type: ignore

        return v  # type: ignore

    def transSeries(
        self,
        axes: Iterable[Literal["x", "y", "z"]],
        thetas: NDArray_1D | RealScalar,
        inplace: bool = False,
    ) -> Self:
        self_v = self._check_assigned(dont_raise=True)
        if self_v is False:
            self_v = self.init_pointc()  # type: ignore
        self_v = self._transSeriesc(axes, thetas, self_v, skip_check=True)  # type: ignore
        return self._inplace_or_not(self_v, inplace)  # type: ignore

    @classmethod
    def transxc(
        cls,
        x: RealScalar,
        v: allowed_input_type_single | None = None,
        skip_check: bool = False,
    ) -> output_type_single:

        return cls._transAxisc("x", x, v, skip_check=skip_check)  # type: ignore

    @classmethod
    def transxcb(
        cls,
        x: RealScalar | NDArray_1D,
        v: allowed_input_type_batch | None = None,
        skip_check: bool = False,
    ) -> output_type_batch:

        return cls._transAxiscb("x", x, v, skip_check=skip_check)  # type: ignore

    def transx(self, x: RealScalar, inplace: bool = False) -> Self:
        self_v = self._check_assigned(dont_raise=True)
        if self_v is False:
            self_v = self.init_pointc()
        self_v = self.transxc(x, self_v, skip_check=True)  # type: ignore
        return self._inplace_or_not(self_v, inplace)

    @classmethod
    def transyc(
        cls,
        y: RealScalar,
        v: allowed_input_type_single | None = None,
        skip_check: bool = False,
    ) -> output_type_single:

        return cls._transAxisc("y", y, v, skip_check=skip_check)  # type: ignore

    @classmethod
    def transycb(
        cls,
        y: RealScalar | NDArray_1D,
        v: allowed_input_type_batch | None = None,
        skip_check: bool = False,
    ) -> output_type_batch:

        return cls._transAxiscb("y", y, v, skip_check=skip_check)  # type: ignore

    def transy(self, y: RealScalar, inplace: bool = False) -> Self:
        self_v = self._check_assigned(dont_raise=True)
        if self_v is False:
            self_v = self.init_pointc()
        self_v = self.transyc(y, self_v, skip_check=True)  # type: ignore
        return self._inplace_or_not(self_v, inplace)

    @classmethod
    def transzc(
        cls,
        z: RealScalar,
        v: allowed_input_type_single | None = None,
        skip_check: bool = False,
    ) -> output_type_single:

        return cls._transAxisc("z", z, v, skip_check=skip_check)  # type: ignore

    @classmethod
    def transzcb(
        cls,
        z: RealScalar | NDArray_1D,
        v: allowed_input_type_batch | None = None,
        skip_check: bool = False,
    ) -> output_type_batch:

        return cls._transAxiscb("z", z, v, skip_check=skip_check)  # type: ignore

    def transz(self, z: RealScalar, inplace: bool = False) -> Self:
        self_v = self._check_assigned(dont_raise=True)
        if self_v is False:
            self_v = self.init_pointc()
        self_v = self.transzc(z, self_v, skip_check=True)  # type: ignore
        return self._inplace_or_not(self_v, inplace)


@beartype
class BasicLieAlgebra[
    allowed_input_type_single: NDArray,
    output_type_single: NDArray,
    allowed_input_type_batch: NDArray,
    output_type_batch: NDArray,
](
    BasicLie[
        allowed_input_type_single,
        output_type_single,
        allowed_input_type_batch,
        output_type_batch,
    ],
    ABC,
):
    def __init__(
        self,
        v: allowed_input_type_single | BasicLie[Any, Any, Any, Any] | None = None,
        normalize: bool = False,
    ) -> None:
        super().__init__(v)
        if normalize:
            self.normalize(True)

    @classmethod
    @abstractmethod
    def intensityc(
        cls, v: allowed_input_type_single, skip_check: bool = False
    ) -> RealScalar:
        pass

    @classmethod
    @abstractmethod
    def intensitycb(
        cls, v: allowed_input_type_batch, skip_check: bool = False
    ) -> NDArray_1D:
        pass

    @property
    def intensity(self) -> RealScalar:
        self_v = self._check_assigned()
        return self.intensityc(self_v, skip_check=True)

    @classmethod
    def normalizec(
        cls, v: allowed_input_type_single, skip_check: bool = False
    ) -> output_type_single:

        intensity = cls.intensityc(v, skip_check=skip_check)
        if utils.check_equal(intensity, 0) is not False:
            return v  # type: ignore
        else:
            return v / intensity  # type: ignore

    @classmethod
    def normalizecb(
        cls, v: allowed_input_type_batch, skip_check: bool = False
    ) -> output_type_batch:

        intensity = cls.intensitycb(v, skip_check=skip_check)

        mask = utils.find_zeros(intensity)
        mask_ = np.logical_not(mask)
        if mask_.any():
            v[mask_] = v[mask_] / intensity[mask_].reshape(-1, 1, 1)
        return v  # type: ignore

    def normalize(self, inplace: bool = False) -> Self:
        self_v = self._check_assigned()
        self_v = self.normalizec(self_v, skip_check=True)
        return self._inplace_or_not(self_v, inplace)

    @classmethod
    @abstractmethod
    def expc(
        cls,
        v: allowed_input_type_single,
        intensity: RealScalar,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        pass

    @classmethod
    @abstractmethod
    def expcb(
        cls,
        v: allowed_input_type_batch,
        intensity: RealScalar | NDArray_1D,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        pass

    @abstractmethod
    def exp(self, intensity: RealScalar, *args: Any, **kwargs: Any) -> Any:
        pass


@beartype
class LieAlgebra3[
    allowed_input_type_single: NDArray,
    output_type_single: NDArray,
    allowed_input_type_batch: NDArray,
    output_type_batch: NDArray,
](
    BasicLieAlgebra[
        allowed_input_type_single,
        output_type_single,
        allowed_input_type_batch,
        output_type_batch,
    ],
    ABC,
):
    @classmethod
    def intensityc(
        cls, v: allowed_input_type_single, skip_check: bool = False
    ) -> RealScalar:

        v = cls._check_shape_and_value(v, skip_check=skip_check)

        if v.shape[1] == 1:
            return np.linalg.norm(v)

        return (np.sum(v**2) / 2) ** 0.5

    @classmethod
    def intensitycb(
        cls, v: allowed_input_type_batch, skip_check: bool = False
    ) -> NDArray_1D:

        v = cls._check_shapeb_and_valueb(v, skip_check=skip_check)

        if v.shape[2] == 1:
            return np.linalg.norm(v, axis=1).flatten()

        v1 = v[:, 2, 1]
        v2 = v[:, 0, 2]
        v3 = v[:, 1, 0]
        return (v1**2 + v2**2 + v3**2) ** 0.5

    @classmethod
    def _exp_basec(
        cls,
        so3_v: NDArray_3_3,
        intensity: RealScalar,
        normalize: bool = False,
        skip_check: bool = False,
    ) -> NDArray_3_3:

        if not skip_check and utils.check_skew3_matrix(so3_v) is False:
            raise ValueError("so3_v is not skew3 matrix")

        intensity_ = (np.sum(so3_v**2) / 2) ** 0.5
        if normalize is False:
            intensity = intensity_ * intensity
        if utils.check_equal(intensity, 0) is False:
            so3_v_norm = so3_v / intensity_
        else:
            so3_v_norm = so3_v

        return (
            np.eye(3)
            + np.sin(intensity) * so3_v_norm
            + ((1 - np.cos(intensity)) * so3_v_norm @ so3_v_norm)
        )

    @classmethod
    def _exp_basecb(
        cls,
        so3_vb: NDArray_N_3_3,
        intensity: RealScalar | NDArray_1D,
        normalize: bool = False,
        skip_check: bool = False,
    ) -> NDArray_N_3_3:

        if not skip_check and utils.check_skew3_matrixb(so3_vb) is False:
            raise ValueError("so3_vb is not skew3 matrixb")

        intensity_ = utils.is_realscaler(intensity)
        if intensity_ is False:
            intensity = np.asarray(intensity, dtype=np.float64)
        else:
            intensity = np.array([intensity_])

        v1 = so3_vb[:, 2, 1]
        v2 = so3_vb[:, 0, 2]
        v3 = so3_vb[:, 1, 0]
        intensity_ = (v1**2 + v2**2 + v3**2) ** 0.5

        if normalize is False:
            intensity = intensity_ * intensity

        mask = utils.find_zeros(intensity_)
        mask_ = np.logical_not(mask)
        so3_vb_norm = so3_vb.copy()
        if mask_.any():
            so3_vb_norm[mask_] = so3_vb[mask_] / intensity_[mask_].reshape(-1, 1, 1)

        sin_t = np.sin(intensity).reshape(-1, 1, 1)
        cos_t = np.cos(intensity).reshape(-1, 1, 1)

        SO3_v = np.zeros((so3_vb_norm.shape[0], 3, 3))
        SO3_v[:, :3, :3] = np.eye(3)

        return SO3_v + sin_t * so3_vb_norm + ((1 - cos_t) * so3_vb_norm @ so3_vb_norm)  # type: ignore

    @classmethod
    def expc(
        cls,
        v: allowed_input_type_single,
        intensity: RealScalar,
        normalize: bool = False,
        skip_check: bool = False,
    ) -> NDArray_3_3:

        v = cls._check_shape_and_value(v, skip_check=skip_check)

        if v.shape[1] == 1:
            return cls._exp_basec(
                utils.skew3(v), intensity, normalize=normalize, skip_check=True
            )

        return cls._exp_basec(v, intensity, normalize=normalize, skip_check=True)

    @classmethod
    def expcb(
        cls,
        v: allowed_input_type_batch,
        intensity: RealScalar | NDArray_1D,
        normalize: bool = False,
        skip_check: bool = False,
    ) -> NDArray_N_3_3:

        v = cls._check_shapeb_and_valueb(v, skip_check=skip_check)

        if v.shape[2] == 1:
            return cls._exp_basecb(
                utils.skew3b(v), intensity, normalize=normalize, skip_check=True
            )

        return cls._exp_basecb(v, intensity, normalize=normalize, skip_check=True)

    def exp(self, intensity: RealScalar, normalize: bool = False) -> lieink.atoms.SO3:

        self_v = self._check_assigned()
        v = self.expc(self_v, intensity, normalize=normalize, skip_check=True)
        result_type = self._check_operator("exp", intensity)

        return result_type(v)  # type: ignore

    @classmethod
    def toso3c(
        cls, v: allowed_input_type_single, skip_check: bool = False
    ) -> NDArray_3_3:
        return utils.skew3(v)

    @classmethod
    def toso3cb(
        cls, v: allowed_input_type_batch, skip_check: bool = False
    ) -> NDArray_N_3_3:
        return utils.skew3b(v)

    def toso3(self) -> lieink.atoms.so3:
        v = self._check_assigned()
        from lieink.atoms import so3

        return so3(self.toso3c(v, skip_check=True))

    @classmethod
    def toTwist3c(
        cls, v: allowed_input_type_single, skip_check: bool = False
    ) -> NDArray_3_1:
        v = cls._check_shape_and_value(v, skip_check=skip_check)
        return utils.veeskew3(v, skip_check=True)

    @classmethod
    def toTwist3cb(
        cls, v: allowed_input_type_batch, skip_check: bool = False
    ) -> NDArray_N_3_1:
        v = cls._check_shapeb_and_valueb(v, skip_check=skip_check)
        return utils.veeskew3b(v, skip_check=True)

    def toTwist3(self) -> lieink.atoms.Twist3:
        v = self._check_assigned()
        from lieink.atoms import Twist3

        return Twist3(self.toTwist3c(v, skip_check=True))


class LieAlgebra[
    allowed_input_type_single: NDArray,
    output_type_single: NDArray,
    allowed_input_type_batch: NDArray,
    output_type_batch: NDArray,
](
    BasicLieAlgebra[
        allowed_input_type_single,
        output_type_single,
        allowed_input_type_batch,
        output_type_batch,
    ],
    ABC,
):
    @classmethod
    @abstractmethod
    def angular_part_skewc(
        cls, v: allowed_input_type_single, skip_check: bool = False
    ) -> NDArray_3_3:
        pass

    @classmethod
    @abstractmethod
    def angular_part_skewcb(
        cls, v: allowed_input_type_batch, skip_check: bool = False
    ) -> NDArray_N_3_3:
        pass

    @property
    def angular_part_skew(self) -> lieink.atoms.so3:
        angular_part_skew_v = self.angular_part_skewc(
            self._check_assigned(), skip_check=True
        )
        from lieink.atoms import so3

        return so3(angular_part_skew_v)

    @classmethod
    @abstractmethod
    def angular_part_veec(
        cls, v: allowed_input_type_single, skip_check: bool = False
    ) -> NDArray_3_1:
        pass

    @classmethod
    @abstractmethod
    def angular_part_veecb(
        cls, v: allowed_input_type_batch, skip_check: bool = False
    ) -> NDArray_N_3_1:
        pass

    @property
    def angular_part_vee(self) -> lieink.atoms.Twist3:
        angular_part_vee_v = self.angular_part_veec(
            self._check_assigned(), skip_check=True
        )
        from lieink.atoms import Twist3

        return Twist3(angular_part_vee_v)

    @classmethod
    @abstractmethod
    def linear_part3c(
        cls, v: allowed_input_type_single, skip_check: bool = False
    ) -> NDArray_3_1:
        pass

    @classmethod
    @abstractmethod
    def linear_part3cb(
        cls, v: allowed_input_type_batch, skip_check: bool = False
    ) -> NDArray_N_3_1:
        pass

    @property
    def linear_part3(self) -> lieink.atoms.Point3:
        linear_part3_v = self.linear_part3c(self._check_assigned(), skip_check=True)
        from lieink.atoms import Point3

        return Point3(linear_part3_v)

    @classmethod
    def linear_part4c(
        cls, v: allowed_input_type_single, skip_check: bool = False
    ) -> NDArray_4_1:
        v = cls._check_shape_and_value(v, skip_check=skip_check)
        linear_part3 = cls.linear_part3c(v, skip_check=True)
        v_ = np.ones((4, 1))
        v_[:3] = linear_part3
        return v_

    @classmethod
    def linear_part4cb(
        cls, v: allowed_input_type_batch, skip_check: bool = False
    ) -> NDArray_N_4_1:
        v = cls._check_shapeb_and_valueb(v, skip_check=skip_check)
        linear_part3 = cls.linear_part3cb(v, skip_check=True)
        v_ = np.ones((v.shape[0], 4, 1))
        v_[:, :3] = linear_part3
        return v_

    @property
    def linear_part4(self) -> lieink.atoms.Point4:
        linear_part4_v = self.linear_part4c(self._check_assigned(), skip_check=True)
        from lieink.atoms import Point4

        return Point4(linear_part4_v)

    @classmethod
    def intensityc(
        cls, v: allowed_input_type_single, skip_check: bool = False
    ) -> RealScalar:
        v = cls._check_shape_and_value(v, skip_check=skip_check)
        angular_part = cls.angular_part_veec(v, skip_check=True)
        linear_part = cls.linear_part3c(v, skip_check=True)

        if utils.get_TypeName(cls._type()) == "Wrench":
            angular_part, linear_part = linear_part, angular_part

        if utils.check_equal(angular_part, 0) is not False:
            return np.linalg.norm(linear_part)
        else:
            return np.linalg.norm(angular_part)

    @classmethod
    def intensitycb(
        cls, v: allowed_input_type_batch, skip_check: bool = False
    ) -> NDArray_1D:

        v = cls._check_shapeb_and_valueb(v, skip_check=skip_check)
        angular_part = cls.angular_part_veecb(v, skip_check=True)
        linear_part = cls.linear_part3cb(v, skip_check=True)

        if utils.get_TypeName(cls._type()) == "Wrench":
            angular_part, linear_part = linear_part, angular_part

        intensity = np.linalg.norm(angular_part, axis=1).flatten()
        mask = utils.find_zeros(intensity, 0)
        intensity[mask] = np.linalg.norm(linear_part[mask], axis=1).flatten()

        return intensity

    @classmethod
    def pitchc(
        cls, v: allowed_input_type_single, skip_check: bool = False
    ) -> RealScalar:
        v = cls._check_shape_and_value(v, skip_check=skip_check)
        angular_part = cls.angular_part_veec(v, skip_check=True)
        linear_part = cls.linear_part3c(v, skip_check=True)

        if utils.get_TypeName(cls._type()) == "Wrench":
            angular_part, linear_part = linear_part, angular_part

        if utils.check_equal(angular_part, 0) is not False:
            return np.inf
        return (angular_part.T @ linear_part / np.linalg.norm(angular_part) ** 2)[0, 0]

    @classmethod
    def pitchcb(
        cls, v: allowed_input_type_batch, skip_check: bool = False
    ) -> NDArray_1D:
        v = cls._check_shapeb_and_valueb(v, skip_check=skip_check)
        angular_part = cls.angular_part_veecb(v, skip_check=True)
        linear_part = cls.linear_part3cb(v, skip_check=True)

        if utils.get_TypeName(cls._type()) == "Wrench":
            angular_part, linear_part = linear_part, angular_part

        angular_part_norm = np.linalg.norm(angular_part, axis=1).flatten()
        mask = np.logical_not(utils.find_zeros(angular_part_norm, 0))
        pitch = np.full(angular_part_norm.shape[0], np.inf)

        pitch[mask] = (
            angular_part[mask].transpose(0, 2, 1) @ linear_part[mask]
        ).flatten() / angular_part_norm[mask] ** 2

        return pitch

    def pitch(self) -> RealScalar:
        return self.pitchc(self._check_assigned())

    @classmethod
    def radius_vectorc(
        cls, v: allowed_input_type_single, skip_check: bool = False
    ) -> NDArray_3_1:
        v = cls._check_shape_and_value(v, skip_check=skip_check)

        if utils.get_TypeName(cls._type()) == "Wrench":
            angular_part = cls.linear_part3c(v, skip_check=True)
            angular_part = utils.skew3(angular_part)
            linear_part = cls.angular_part_veec(v, skip_check=True)
        else:
            angular_part = cls.angular_part_skewc(v, skip_check=True)
            linear_part = cls.linear_part3c(v, skip_check=True)

        if utils.check_equal(angular_part, 0) is not False:
            return np.array([[np.inf], [np.inf], [np.inf]]).astype(np.float64)

        return angular_part @ linear_part / (np.sum(angular_part**2) / 2)

    @classmethod
    def radius_vectorcb(
        cls, v: allowed_input_type_batch, skip_check: bool = False
    ) -> NDArray_N_3_1:
        v = cls._check_shapeb_and_valueb(v, skip_check=skip_check)
        angular_part = cls.angular_part_skewcb(v, skip_check=True)
        linear_part = cls.linear_part3cb(v, skip_check=True)

        if utils.get_TypeName(cls._type()) == "Wrench":
            angular_part = cls.linear_part3cb(v, skip_check=True)
            angular_part = utils.skew3b(angular_part)
            linear_part = cls.angular_part_veecb(v, skip_check=True)
        else:
            angular_part = cls.angular_part_skewcb(v, skip_check=True)
            linear_part = cls.linear_part3cb(v, skip_check=True)

        angular_part_norm = (
            angular_part[:, 0, 1] ** 2
            + angular_part[:, 0, 2] ** 2
            + angular_part[:, 1, 2] ** 2
        ) ** 0.5
        mask = np.logical_not(utils.find_zeros(angular_part_norm, 0))
        radius_vector = np.full((angular_part_norm.shape[0], 3, 1), np.inf)

        radius_vector[mask] = (angular_part[mask] @ linear_part[mask]) / (
            (angular_part_norm[mask]).reshape(-1, 1, 1)
        ) ** 2

        return radius_vector

    def radius_vector(self) -> lieink.atoms.Point3:
        from lieink.atoms import Point3

        return Point3(self.radius_vectorc(self._check_assigned()))

    @overload
    @classmethod
    def expc(
        cls,
        v: allowed_input_type_single,
        intensity: RealScalar,
        expto: Literal["SE3"] = "SE3",
        normalize: bool = False,
        skip_check: bool = False,
    ) -> NDArray_4_4: ...

    @overload
    @classmethod
    def expc(
        cls,
        v: allowed_input_type_single,
        intensity: RealScalar,
        expto: Literal["Ad", "coAd"],
        normalize: bool = False,
        skip_check: bool = False,
    ) -> NDArray_6_6: ...

    @classmethod
    def expc(
        cls,
        v: allowed_input_type_single,
        intensity: RealScalar,
        expto: Literal["SE3", "Ad", "coAd"] = "SE3",
        normalize: bool = False,
        skip_check: bool = False,
    ) -> NDArray_4_4 | NDArray_6_6:

        v = cls._check_shape_and_value(v, skip_check=skip_check)
        skew3_w = cls.angular_part_skewc(v, skip_check=True)
        vee3_w = cls.angular_part_veec(v, skip_check=True)
        vee3_v = cls.linear_part3c(v, skip_check=True)

        v = cls._exp_basec(  # type: ignore
            skew3_w,
            vee3_w,
            vee3_v,
            intensity,
            normalize,
            expto,
            skip_check=True,
        )

        return v

    @overload
    @classmethod
    def expcb(
        cls,
        v: allowed_input_type_batch,
        intensity: NDArray_1D | RealScalar,
        expto: Literal["SE3"] = "SE3",
        normalize: bool = False,
        skip_check: bool = False,
    ) -> NDArray_N_4_4: ...

    @overload
    @classmethod
    def expcb(
        cls,
        v: allowed_input_type_batch,
        intensity: NDArray_1D | RealScalar,
        expto: Literal["Ad", "coAd"],
        normalize: bool = False,
        skip_check: bool = False,
    ) -> NDArray_N_6_6: ...

    @classmethod
    def expcb(
        cls,
        v: allowed_input_type_batch,
        intensity: NDArray_1D | RealScalar,
        expto: Literal["SE3", "Ad", "coAd"] = "SE3",
        normalize: bool = False,
        skip_check: bool = False,
    ) -> NDArray_N_4_4 | NDArray_N_6_6:

        v = cls._check_shapeb_and_valueb(v, skip_check=skip_check)
        skew3_w = cls.angular_part_skewcb(v, skip_check=True)
        vee3_w = cls.angular_part_veecb(v, skip_check=True)
        vee3_v = cls.linear_part3cb(v, skip_check=True)

        v = cls._exp_basecb(  # type: ignore
            skew3_w,
            vee3_w,
            vee3_v,
            intensity,
            normalize,
            expto,
            skip_check=True,
        )

        return v

    @overload
    def exp(
        self,
        intensity: RealScalar,
        expto: Literal["SE3"] = "SE3",
        normalize: bool = False,
    ) -> lieink.atoms.SE3: ...

    @overload
    def exp(
        self,
        intensity: RealScalar,
        expto: Literal["Ad"],
        normalize: bool = False,
    ) -> lieink.atoms.Ad: ...

    @overload
    def exp(
        self,
        intensity: RealScalar,
        expto: Literal["coAd"],
        normalize: bool = False,
    ) -> lieink.atoms.coAd: ...

    def exp(
        self,
        intensity: RealScalar,
        expto: Literal["SE3", "Ad", "coAd"] = "SE3",
        normalize: bool = False,
    ) -> lieink.atoms.SE3 | lieink.atoms.Ad | lieink.atoms.coAd:

        self_v = self._check_assigned()
        v = self.expc(
            self_v,
            intensity,
            normalize=normalize,
            skip_check=True,
            expto=expto,
        )
        result_type = self._check_operator("exp", intensity)[expto]  # type: ignore

        return result_type(v)  # type: ignore

    @classmethod
    def _exp_basec(
        cls,
        skew3_w: NDArray_3_3,
        vee3_w: NDArray_3_1,
        vee3_v: NDArray_3_1,
        intensity: RealScalar,
        normalize: bool = False,
        expto: Literal["SE3", "Ad", "coAd"] = "SE3",
        skip_check: bool = False,
    ) -> NDArray_4_4 | NDArray_6_6:

        if not skip_check:
            if utils.check_skew3_matrix(skew3_w) is False:
                raise ValueError("skew3_w is not skew3 matrix")
            if utils.check_equal(utils.skew3(vee3_w), skew3_w) is False:
                raise ValueError("vee3_w is not vee3 of skew3_w")

        intensity_w = np.linalg.norm(vee3_w)
        if utils.check_equal(intensity_w, 0) is not False:
            if normalize:
                intensity_v = np.linalg.norm(vee3_v)
                if utils.check_equal(intensity_v, 0) is False:
                    vee3_v = vee3_v / intensity_v

            rotation_matrix = np.eye(3)
            t = intensity * vee3_v

        else:
            if not normalize:
                intensity = intensity * intensity_w
            skew3_w_norm = skew3_w / intensity_w
            vee3_w_norm = vee3_w / intensity_w
            vee3_v_norm = vee3_v / intensity_w

            rotation_matrix = LieAlgebra3._exp_basec(  # type: ignore
                skew3_w_norm, intensity, skip_check=True
            )

            t = (
                (np.eye(3) - rotation_matrix) @ skew3_w_norm
                + intensity * vee3_w_norm @ vee3_w_norm.T
            ) @ vee3_v_norm

        return LieGroup._combine_basec(  # type: ignore
            rotation_matrix,
            t,
            combineto=expto,
            skip_check=skip_check,
        )

    @classmethod
    def _exp_basecb(
        cls,
        skew3_wb: NDArray_N_3_3,
        vee3_wb: NDArray_N_3_1,
        vee3_vb: NDArray_N_3_1,
        intensityb: NDArray_1D | RealScalar,
        normalizeb: bool = False,
        expto: Literal["SE3", "Ad", "coAd"] = "SE3",
        skip_check: bool = False,
    ) -> NDArray_N_4_4 | NDArray_N_6_6:

        if not skip_check:
            if utils.check_skew3_matrixb(skew3_wb) is False:
                raise ValueError("skew3_wb is not skew3 matrix")
            if utils.check_equal(utils.skew3b(vee3_wb), skew3_wb) is False:
                raise ValueError("vee3_wb is not vee3 of skew3_w")
            if (
                skew3_wb.shape[0] != vee3_wb.shape[0]
                or vee3_wb.shape[0] != vee3_vb.shape[0]
            ):
                raise ValueError("skew3_wb, vee3_wb, vee3_vb must have the same num")

        num = skew3_wb.shape[0]

        if utils.is_realscaler(intensityb) is not False:
            intensityb = np.array([intensityb] * num).reshape(-1, 1, 1)
        else:
            if intensityb.shape[0] != num:  # type: ignore
                raise ValueError("intensityb must have the same num")
            intensityb = intensityb.reshape(-1, 1, 1)  # type: ignore

        norm_wb = np.linalg.norm(vee3_wb, axis=1).reshape(-1, 1, 1)
        norm = norm_wb.copy()

        mask_norm_wb_zeros = utils.find_zeros(norm_wb).flatten()
        mask_norm_wb_not_zeros = np.logical_not(mask_norm_wb_zeros)
        norm[mask_norm_wb_zeros] = np.linalg.norm(
            vee3_vb[mask_norm_wb_zeros], axis=1
        ).reshape(-1, 1, 1)
        norm[utils.find_zeros(norm)] = 1

        if not normalizeb:
            intensityb = intensityb * norm  # type: ignore

        skew_wb_norm = skew3_wb / norm
        vee_wb_norm = vee3_wb / norm
        vee_vb_norm = vee3_vb / norm

        rotation_matrixb = np.zeros((num, 3, 3))
        rotation_matrixb[:, :, :] = np.eye(3)
        tb = cast(NDArray_N_3_1, vee_vb_norm * intensityb)

        rotation_matrixb[mask_norm_wb_not_zeros] = LieAlgebra3._exp_basecb(  # type: ignore
            skew_wb_norm[mask_norm_wb_not_zeros],
            intensityb[mask_norm_wb_not_zeros].flatten(),  # type: ignore
            skip_check=True,
        )

        As_term1 = (
            (np.eye(3) - rotation_matrixb[mask_norm_wb_not_zeros])
            @ skew_wb_norm[mask_norm_wb_not_zeros],
        )

        As_term2 = intensityb[mask_norm_wb_not_zeros] * np.einsum(  # type: ignore
            "nij,nkj->nik",
            vee_wb_norm[mask_norm_wb_not_zeros],
            vee_wb_norm[mask_norm_wb_not_zeros],
        )  # type: ignore

        As = cast(NDArray_N_3_3, As_term1 + As_term2)
        tb[mask_norm_wb_not_zeros] = As @ vee_vb_norm[mask_norm_wb_not_zeros]

        return LieGroup._combine_basecb(  # type: ignore
            rotation_matrixb, tb, combineto=expto, skip_check=True
        )

    @classmethod
    def _combine_basec(
        cls,
        angular_part: NDArray_3_3 | NDArray_3_1 | NDArray_3,
        linear_part: NDArray_3_3 | NDArray_3_1 | NDArray_3 | NDArray_4_1 | NDArray_4,
        se3_or_Twist_or_Wrench_or_ad_or_coad: Literal[
            "se3", "Twist", "Wrench", "ad", "coad"
        ] = "se3",
        skip_check: bool = False,
    ) -> NDArray_4_4 | NDArray_6_1 | NDArray_6_6:

        if angular_part.ndim == 1:
            vee3_angular_part = angular_part.reshape(3, 1)
            skew3_angular_part = utils.skew3(vee3_angular_part)
        elif angular_part.shape[-1] == 3:
            skew3_angular_part = angular_part
            if not skip_check and utils.check_skew3_matrix(skew3_angular_part) is False:
                raise ValueError("angular_part is not skew3 matrix")
            vee3_angular_part = utils.veeskew3(skew3_angular_part)
        else:
            vee3_angular_part = angular_part
            skew3_angular_part = utils.skew3(vee3_angular_part)

        if linear_part.ndim == 1:
            linear_part = linear_part.reshape(-1, 1)

        if linear_part.shape[-1] == 3:
            skew3_linear_part = linear_part
            if not skip_check and utils.check_skew3_matrix(skew3_linear_part) is False:
                raise ValueError("linear_part is not skew3 matrix")
            vee3_linear_part = utils.veeskew3(skew3_linear_part)
        elif linear_part.shape[0] == 4:
            if utils.check_equal(linear_part[-1], 0) is not False:
                raise ValueError("linear_part[-1] is 0")
            vee3_linear_part = linear_part[:3] / linear_part[-1]
            skew3_linear_part = utils.skew3(vee3_linear_part)
        else:
            vee3_linear_part = linear_part
            skew3_linear_part = utils.skew3(vee3_linear_part)

        if se3_or_Twist_or_Wrench_or_ad_or_coad == "se3":
            v = np.zeros((4, 4))
            v[:3, :3] = skew3_angular_part
            v[:3, 3:] = vee3_linear_part
        elif (
            se3_or_Twist_or_Wrench_or_ad_or_coad == "Twist"
            or se3_or_Twist_or_Wrench_or_ad_or_coad == "Wrench"
        ):
            v = np.zeros((6, 1))
            v[:3] = vee3_angular_part
            v[3:] = vee3_linear_part
        elif se3_or_Twist_or_Wrench_or_ad_or_coad == "ad":
            v = np.zeros((6, 6))
            v[:3, :3] = skew3_angular_part
            v[3:, 3:] = skew3_angular_part
            v[3:, :3] = skew3_linear_part
        else:
            v = np.zeros((6, 6))
            v[:3, :3] = skew3_angular_part
            v[3:, 3:] = skew3_angular_part
            v[:3, 3:] = skew3_linear_part

        return v

    @classmethod
    def _combine_basecb(
        cls,
        angular_partb: NDArray_N_3_3 | NDArray_N_3_1 | NDArray_3_N,
        linear_partb: NDArray_N_3_3
        | NDArray_N_3_1
        | NDArray_3_N
        | NDArray_N_4_1
        | NDArray_4_N,
        se3_or_Twist_or_Wrench_or_ad_or_coad: Literal[
            "se3", "Twist", "Wrench", "ad", "coad"
        ] = "se3",
        skip_check: bool = False,
    ) -> NDArray_N_4_4 | NDArray_N_6_1 | NDArray_N_6_6:

        if angular_partb.ndim == 2:
            vee3_angular_partb = angular_partb.reshape(3, -1, 1).swapaxes(0, 1)
            skew3_angular_partb = utils.skew3b(vee3_angular_partb)
        elif angular_partb.shape[-1] == 3:
            skew3_angular_partb = angular_partb
            if (
                not skip_check
                and utils.check_skew3_matrixb(skew3_angular_partb) is False
            ):
                raise ValueError("angular_partb is not skew3 matrixb")
            vee3_angular_partb = utils.veeskew3b(skew3_angular_partb)
        else:
            vee3_angular_partb = angular_partb
            skew3_angular_partb = utils.skew3b(vee3_angular_partb)

        if linear_partb.ndim == 2:
            vee3_linear_partb = linear_partb.reshape(
                linear_partb.shape[0], -1, 1
            ).swapaxes(0, 1)
            if vee3_linear_partb.shape[1] == 4:
                if utils.find_zeros(vee3_linear_partb[:, 3]).any():
                    raise ValueError(
                        "linear_partb has zero in last row, which means it contains at least a Vector4."
                    )
                vee3_linear_partb = vee3_linear_partb[:, :3] / vee3_linear_partb[:, 3:4]
            skew3_linear_partb = utils.skew3b(vee3_linear_partb)
        elif linear_partb.shape[-1] == 3:
            skew3_linear_partb = linear_partb
            if (
                not skip_check
                and utils.check_skew3_matrixb(skew3_linear_partb) is False
            ):
                raise ValueError("linear_partb is not skew3 matrixb")
            vee3_linear_partb = utils.veeskew3b(skew3_linear_partb)
        else:
            vee3_linear_partb = linear_partb
            skew3_linear_partb = utils.skew3b(vee3_linear_partb)

        num = skew3_angular_partb.shape[0]
        if num != skew3_linear_partb.shape[0]:
            raise ValueError(
                "skew3_angular_partb and skew3_linear_partb must have same num."
            )

        if se3_or_Twist_or_Wrench_or_ad_or_coad == "se3":
            v = np.zeros((num, 4, 4))
            v[:, :3, :3] = skew3_angular_partb
            v[:, :3, 3:] = vee3_linear_partb
        elif (
            se3_or_Twist_or_Wrench_or_ad_or_coad == "Twist"
            or se3_or_Twist_or_Wrench_or_ad_or_coad == "Wrench"
        ):
            v = np.zeros((num, 6, 1))
            v[:, :3] = vee3_angular_partb
            v[:, 3:] = vee3_linear_partb
        elif se3_or_Twist_or_Wrench_or_ad_or_coad == "ad":
            v = np.zeros((num, 6, 6))
            v[:, :3, :3] = skew3_angular_partb
            v[:, 3:, 3:] = skew3_angular_partb
            v[:, 3:, :3] = skew3_linear_partb
        else:
            v = np.zeros((num, 6, 6))
            v[:, :3, :3] = skew3_angular_partb
            v[:, 3:, 3:] = skew3_angular_partb
            v[:, :3, 3:] = skew3_linear_partb

        return v

    @classmethod
    def combinec(
        cls,
        angular_part: NDArray_3_3 | NDArray_3_1 | NDArray_3,
        linear_part: NDArray_3_3 | NDArray_3_1 | NDArray_3 | NDArray_4_1 | NDArray_4,
        skip_check: bool = False,
    ) -> output_type_single:
        mytype_name = utils.get_TypeName(cls._type())  # type: ignore
        return cls._combine_basec(
            angular_part,
            linear_part,
            mytype_name,  # type: ignore
            skip_check=skip_check,
        )  # type: ignore

    @classmethod
    def combinecb(
        cls,
        angular_partb: NDArray_N_3_3 | NDArray_N_3_1 | NDArray_3_N,
        linear_partb: NDArray_N_3_3 | NDArray_N_3_1 | NDArray_3_N,
        skip_check: bool = False,
    ) -> output_type_batch:
        mytype_name = utils.get_TypeName(cls._type())  # type: ignore
        return cls._combine_basecb(
            angular_partb,
            linear_partb,
            mytype_name,  # type: ignore
            skip_check=skip_check,
        )  # type: ignore

    def combine(
        self,
        angular_part: lieink.atoms.so3
        | lieink.atoms.Twist3
        | NDArray_3_3
        | NDArray_3_1
        | NDArray_3,
        linear_part: lieink.atoms.so3
        | lieink.atoms.Twist3
        | lieink.atoms.Point3
        | lieink.atoms.Point4
        | NDArray_3_3
        | NDArray_3_1
        | NDArray_3,
        skip_check: bool = False,
    ):
        if self._check_assigned(dont_raise=True) is not False:
            raise ValueError("self has been assigned")

        if isinstance(angular_part, Lie):
            angular_part = angular_part._check_assigned()  # type: ignore
        if isinstance(linear_part, Lie):
            linear_part = linear_part._check_assigned()  # type: ignore
        self._v = self.combinec(angular_part, linear_part, skip_check=skip_check)  # type: ignore
        return self

    @classmethod
    def tose3c(
        cls, v: allowed_input_type_single, skip_check: bool = False
    ) -> NDArray_4_4:
        angular_part = cls.angular_part_skewc(v, skip_check=skip_check)
        linear_part = cls.linear_part3c(v, skip_check=True)
        return cls._combine_basec(angular_part, linear_part, "se3", skip_check=True)

    @classmethod
    def tose3cb(
        cls, v: allowed_input_type_batch, skip_check: bool = False
    ) -> NDArray_N_4_4:
        angular_part = cls.angular_part_skewcb(v, skip_check=skip_check)
        linear_part = cls.linear_part3cb(v, skip_check=True)
        return cls._combine_basecb(angular_part, linear_part, "se3", skip_check=True)

    def tose3(self) -> lieink.atoms.se3:
        self_v = self._check_assigned()
        from lieink.atoms import se3

        return se3(self.tose3c(self_v, skip_check=True))

    @classmethod
    def toTwistc(
        cls, v: allowed_input_type_single, skip_check: bool = False
    ) -> NDArray_6_1:
        angular_part = cls.angular_part_skewc(v, skip_check=skip_check)
        linear_part = cls.linear_part3c(v, skip_check=True)
        return cls._combine_basec(angular_part, linear_part, "Twist", skip_check=True)

    @classmethod
    def toTwistcb(
        cls, v: allowed_input_type_batch, skip_check: bool = False
    ) -> NDArray_N_6_1:
        angular_part = cls.angular_part_skewcb(v, skip_check=skip_check)
        linear_part = cls.linear_part3cb(v, skip_check=True)
        return cls._combine_basecb(angular_part, linear_part, "Twist", skip_check=True)

    def toTwist(self) -> lieink.atoms.Twist:
        self_v = self._check_assigned()
        from lieink.atoms import Twist

        return Twist(self.toTwistc(self_v, skip_check=True))

    @classmethod
    def toadc(
        cls, v: allowed_input_type_single, skip_check: bool = False
    ) -> NDArray_6_6:
        angular_part = cls.angular_part_skewc(v, skip_check=skip_check)
        linear_part = cls.linear_part3c(v, skip_check=True)
        return cls._combine_basec(angular_part, linear_part, "ad", skip_check=True)

    @classmethod
    def toadcb(
        cls, v: allowed_input_type_batch, skip_check: bool = False
    ) -> NDArray_N_6_6:
        angular_part = cls.angular_part_skewcb(v, skip_check=skip_check)
        linear_part = cls.linear_part3cb(v, skip_check=True)
        return cls._combine_basecb(angular_part, linear_part, "ad", skip_check=True)

    def toad(self) -> lieink.atoms.ad:
        self_v = self._check_assigned()
        from lieink.atoms import ad

        return ad(self.toadc(self_v, skip_check=True))

    @classmethod
    def tocoadc(
        cls, v: allowed_input_type_single, skip_check: bool = False
    ) -> NDArray_6_6:
        angular_part = cls.angular_part_skewc(v, skip_check=skip_check)
        linear_part = cls.linear_part3c(v, skip_check=True)
        return cls._combine_basec(angular_part, linear_part, "coad", skip_check=True)

    @classmethod
    def tocoadcb(
        cls, v: allowed_input_type_batch, skip_check: bool = False
    ) -> NDArray_N_6_6:
        angular_part = cls.angular_part_skewcb(v, skip_check=skip_check)
        linear_part = cls.linear_part3cb(v, skip_check=True)
        return cls._combine_basecb(angular_part, linear_part, "coad", skip_check=True)

    def tocoad(self) -> lieink.atoms.coad:
        self_v = self._check_assigned()
        from lieink.atoms import coad

        return coad(self.tocoadc(self_v, skip_check=True))


class BasicLieGroup[
    allowed_input_type_single: NDArray,
    output_type_single: NDArray,
    allowed_input_type_batch: NDArray,
    output_type_batch: NDArray,
](
    BasicLie[
        allowed_input_type_single,
        output_type_single,
        allowed_input_type_batch,
        output_type_batch,
    ],
    ABC,
):
    @classmethod
    @abstractmethod
    def invc(
        cls, v: allowed_input_type_single, skip_check: bool = False
    ) -> output_type_single:
        pass

    @classmethod
    @abstractmethod
    def invcb(
        cls, v: allowed_input_type_batch, skip_check: bool = False
    ) -> output_type_batch:
        pass

    @property
    def inv(self) -> Self:
        self_v = self._check_assigned()
        return self.mytype(self.invc(self_v, skip_check=True))  # type: ignore

    @classmethod
    @abstractmethod
    def logc(cls, v: allowed_input_type_single) -> Any:
        pass

    @classmethod
    @abstractmethod
    def logcb(cls, v: allowed_input_type_batch) -> Any:
        pass

    @abstractmethod
    def log(self) -> Any:
        pass

    @classmethod
    def init_liegroupc(cls) -> output_type_single:
        if cls._r == 3:
            v = np.eye(3)
        elif cls._r == 4:
            v = np.eye(4)
        else:
            v = np.eye(6)
        return cast(output_type_single, v)

    @classmethod
    def init_liegroupcb(cls, num: int) -> output_type_batch:

        if cls._r == 3:
            v = np.zeros((num, 3, 3))
            v[:, :3, :3] = np.eye(3)
        elif cls._r == 4:
            v = np.zeros((num, 4, 4))
            v[:, :4, :4] = np.eye(4)
        else:
            v = np.zeros((num, 6, 6))
            v[:, :6, :6] = np.eye(6)

        return cast(output_type_batch, v)

    def init_liegroup(self) -> Self:
        self._v = self.init_liegroupc()
        return self

    @classmethod
    @abstractmethod
    def _rot_basec(
        cls, axis: Literal["x", "y", "z"], theta: RealScalar
    ) -> allowed_input_type_single:
        pass

    @classmethod
    @abstractmethod
    def _rot_basecb(
        cls, axis: Literal["x", "y", "z"], theta: NDArray_1D | RealScalar
    ) -> allowed_input_type_batch:
        pass

    @classmethod
    def _rotAxisc(
        cls,
        axis: Literal["x", "y", "z"],
        theta: RealScalar,
        v: allowed_input_type_single | None = None,
        left_or_right: Literal["left", "right"] = "right",
        skip_check: bool = False,
    ) -> allowed_input_type_single:

        v_ = cls._rot_basec(axis, theta)

        if v is None:
            return v_
        else:
            v = cls._check_shape_and_value(v, skip_check=skip_check)
            if left_or_right == "right":
                return v @ v_  # type: ignore
            else:
                return v_ @ v  # type: ignore

    @classmethod
    def _rotAxiscb(
        cls,
        axis: Literal["x", "y", "z"],
        thetas: NDArray_1D | RealScalar,
        v: allowed_input_type_batch | None = None,
        left_or_right: Literal["left", "right"] = "right",
        skip_check: bool = False,
    ) -> allowed_input_type_batch:

        v_ = cls._rot_basecb(axis, thetas)

        if v is None:
            return v_
        else:
            v = cls._check_shapeb_and_valueb(v, skip_check=skip_check)
            if v.shape[0] != v_.shape[0]:
                raise ValueError(f"v and {axis} must have the same number of elements.")
            if left_or_right == "right":
                return Lie.mulcb(v, v_)  # type: ignore
            else:
                return Lie.mulcb(v_, v)  # type: ignore

    @classmethod
    def rotxc(
        cls,
        x: RealScalar,
        v: allowed_input_type_single | None = None,
        left_or_right: Literal["left", "right"] = "right",
        skip_check: bool = False,
    ) -> output_type_single:

        return cls._rotAxisc("x", x, v, left_or_right, skip_check)  # type: ignore

    @classmethod
    def rotxcb(
        cls,
        x: NDArray_1D | RealScalar,
        v: allowed_input_type_batch | None = None,
        left_or_right: Literal["left", "right"] = "right",
        skip_check: bool = False,
    ) -> output_type_batch:

        return cls._rotAxiscb("x", x, v, left_or_right, skip_check)  # type: ignore

    def rotx(
        self,
        x: RealScalar,
        left_or_right: Literal["left", "right"] = "right",
        inplace: bool = False,
    ) -> Self:

        self_v = self.rotxc(x, self._v, left_or_right, skip_check=True)  # type: ignore
        return self._inplace_or_not(self_v, inplace)

    @classmethod
    def rotyc(
        cls,
        y: RealScalar,
        v: allowed_input_type_single | None = None,
        left_or_right: Literal["left", "right"] = "right",
        skip_check: bool = False,
    ) -> output_type_single:

        return cls._rotAxisc("y", y, v, left_or_right, skip_check)  # type: ignore

    @classmethod
    def rotycb(
        cls,
        y: NDArray_1D | RealScalar,
        v: allowed_input_type_batch | None = None,
        left_or_right: Literal["left", "right"] = "right",
        skip_check: bool = False,
    ) -> output_type_batch:

        return cls._rotAxiscb("y", y, v, left_or_right, skip_check)  # type: ignore

    def roty(
        self,
        y: RealScalar,
        left_or_right: Literal["left", "right"] = "right",
        inplace: bool = False,
    ) -> Self:

        self_v = self.rotyc(y, self._v, left_or_right, skip_check=True)  # type: ignore
        return self._inplace_or_not(self_v, inplace)

    @classmethod
    def rotzc(
        cls,
        z: RealScalar,
        v: allowed_input_type_single | None = None,
        left_or_right: Literal["left", "right"] = "right",
        skip_check: bool = False,
    ) -> output_type_single:

        return cls._rotAxisc("z", z, v, left_or_right, skip_check)  # type: ignore

    @classmethod
    def rotzcb(
        cls,
        z: NDArray_1D | RealScalar,
        v: allowed_input_type_batch | None = None,
        left_or_right: Literal["left", "right"] = "right",
        skip_check: bool = False,
    ) -> output_type_batch:

        return cls._rotAxiscb("z", z, v, left_or_right, skip_check)  # type: ignore

    def rotz(
        self,
        z: RealScalar,
        left_or_right: Literal["left", "right"] = "right",
        inplace: bool = False,
    ) -> Self:

        self_v = self.rotzc(z, self._v, left_or_right, skip_check=True)  # type: ignore
        return self._inplace_or_not(self_v, inplace)

    @classmethod
    def _rotSeriesc(
        cls,
        axes: Iterable[Literal["x", "y", "z"]],
        thetas: NDArray_1D | RealScalar,
        v: allowed_input_type_single | None = None,
        left_or_right: Literal["left", "right"] = "right",
        skip_check: bool = False,
    ) -> allowed_input_type_single:

        num_axis = len(axes)  # type: ignore
        if utils.is_realscaler(thetas) is not False:
            thetas = np.array([thetas] * num_axis)
        else:
            thetas = cast(NDArray_1D, thetas)
            if thetas.shape[0] != num_axis:
                raise ValueError("thetas and axes must have the same length")
        if v is None:
            v = cls.init_liegroupc()  # type: ignore
        else:
            v = cls._check_shape_and_value(v, skip_check=skip_check)

        for i in range(num_axis):
            v = cls._rotAxisc(axes[i], thetas[i], v, left_or_right, skip_check)  # type: ignore

        return v  # type: ignore

    @classmethod
    def _rotSeriescb(
        cls,
        axes: Iterable[Literal["x", "y", "z"]],
        thetas: NDArray_1D | RealScalar,
        v: allowed_input_type_batch | None = None,
        left_or_right: Literal["left", "right"] = "right",
        skip_check: bool = False,
    ) -> allowed_input_type_batch:

        num_axis = len(axes)  # type: ignore
        if utils.is_realscaler(thetas) is not False:
            thetas = np.array([thetas] * num_axis)
        else:
            thetas = cast(NDArray_1D, thetas)
            if thetas.shape[0] != num_axis:
                raise ValueError("thetas and axes must have the same length")
        if v is None:
            v = cls.init_liegroupcb(1)  # type: ignore
        else:
            v = cls._check_shapeb_and_valueb(v, skip_check=skip_check)

        for i in range(num_axis):
            v = cls._rotAxiscb(axes[i], thetas[i], v, left_or_right, skip_check)  # type: ignore

        return v  # type: ignore

    def rotSeries(
        self,
        axes: Iterable[Literal["x", "y", "z"]],
        thetas: NDArray_1D | RealScalar,
        left_or_right: Literal["left", "right"] = "right",
        inplace: bool = False,
    ) -> Self:
        self_v = self._rotSeriesc(axes, thetas, self._v, left_or_right, skip_check=True)  # type: ignore
        return self._inplace_or_not(self_v, inplace)  # type: ignore


@beartype
class LieGroup3[
    allowed_input_type_single: NDArray,
    output_type_single: NDArray,
    allowed_input_type_batch: NDArray,
    output_type_batch: NDArray,
](
    BasicLieGroup[
        allowed_input_type_single,
        output_type_single,
        allowed_input_type_batch,
        output_type_batch,
    ],
    ABC,
):
    @overload
    @classmethod
    def logc(
        cls,
        v: allowed_input_type_single,
        so3_or_Twist3: Literal["so3"] = "so3",
        return_intensity: Literal[False] = False,
        normalize: bool = False,
        skip_check: bool = False,
    ) -> NDArray_3_3: ...

    @overload
    @classmethod
    def logc(
        cls,
        v: allowed_input_type_single,
        so3_or_Twist3: Literal["so3"] = "so3",
        *,
        return_intensity: Literal[True],
        normalize: bool = False,
        skip_check: bool = False,
    ) -> tuple[NDArray_3_3, RealScalar]: ...

    @overload
    @classmethod
    def logc(
        cls,
        v: allowed_input_type_single,
        so3_or_Twist3: Literal["Twist3"],
        return_intensity: Literal[False] = False,
        normalize: bool = False,
        skip_check: bool = False,
    ) -> NDArray_3_1: ...

    @overload
    @classmethod
    def logc(
        cls,
        v: allowed_input_type_single,
        so3_or_Twist3: Literal["Twist3"],
        return_intensity: Literal[True],
        normalize: bool = False,
        skip_check: bool = False,
    ) -> tuple[NDArray_3_1, RealScalar]: ...

    @classmethod
    def logc(
        cls,
        v: allowed_input_type_single,
        so3_or_Twist3: Literal["so3", "Twist3"] = "so3",
        return_intensity: bool = False,
        normalize: bool = False,
        skip_check: bool = False,
    ) -> (
        NDArray_3_3
        | tuple[NDArray_3_3, RealScalar]
        | NDArray_3_1
        | tuple[NDArray_3_1, RealScalar]
    ):
        v = cls._check_shape_and_value(v, skip_check=skip_check)  # type: ignore
        cos_intensity = (np.trace(v) - 1) / 2
        cos_intensity = np.clip(cos_intensity, -1, 1)
        intensity = np.arccos(cos_intensity)
        if utils.check_equal(intensity, 0) is not False:
            v = np.zeros((3, 3))  # type: ignore
        else:
            v = (v - v.T) / (2 * np.sin(intensity))
        if normalize is False:
            v = v * intensity
        if so3_or_Twist3 == "so3":
            if return_intensity:
                return v, intensity
            else:
                return v
        else:
            if return_intensity:
                return utils.veeskew3(v), intensity
            else:
                return utils.veeskew3(v)

    @overload
    @classmethod
    def logcb(
        cls,
        v: allowed_input_type_batch,
        so3_or_Twist3: Literal["so3"] = "so3",
        return_intensity: Literal[False] = False,
        normalize: bool = False,
        skip_check: bool = False,
    ) -> NDArray_N_3_3: ...

    @overload
    @classmethod
    def logcb(
        cls,
        v: allowed_input_type_batch,
        so3_or_Twist3: Literal["so3"] = "so3",
        *,
        return_intensity: Literal[True],
        normalize: bool = False,
        skip_check: bool = False,
    ) -> tuple[NDArray_N_3_3, NDArray_1D]: ...

    @overload
    @classmethod
    def logcb(
        cls,
        v: allowed_input_type_batch,
        so3_or_Twist3: Literal["Twist3"],
        return_intensity: Literal[False] = False,
        normalize: bool = False,
        skip_check: bool = False,
    ) -> NDArray_N_3_1: ...

    @overload
    @classmethod
    def logcb(
        cls,
        v: allowed_input_type_batch,
        so3_or_Twist3: Literal["Twist3"],
        return_intensity: Literal[True],
        normalize: bool = False,
        skip_check: bool = False,
    ) -> tuple[NDArray_N_3_1, NDArray_1D]: ...

    @classmethod
    def logcb(
        cls,
        v: allowed_input_type_batch,
        so3_or_Twist3: Literal["so3", "Twist3"] = "so3",
        return_intensity: bool = False,
        normalize: bool = False,
        skip_check: bool = False,
    ) -> (
        NDArray_N_3_3
        | tuple[NDArray_N_3_3, NDArray_1D]
        | NDArray_N_3_1
        | tuple[NDArray_N_3_1, NDArray_1D]
    ):
        v = cls._check_shapeb_and_valueb(v, skip_check=skip_check)  # type: ignore

        cos_intensity = (v[:, 0, 0] + v[:, 1, 1] + v[:, 2, 2] - 1) / 2
        cos_intensity = np.clip(cos_intensity, -1, 1)

        intensity = np.arccos(cos_intensity)

        mask = utils.find_zeros(intensity)
        mask_ = np.logical_not(mask)
        ws = np.zeros((v.shape[0], 3, 3))

        if mask_.any():
            ws[mask_] = (v[mask_] - v[mask_].transpose(0, 2, 1)) / (
                2 * np.sin(intensity[mask_].reshape(-1, 1, 1))
            )
            if normalize is False:
                ws = ws * intensity.reshape(-1, 1, 1)

        if so3_or_Twist3 != "so3":
            ws = utils.veeskew3b(ws)

        if return_intensity:
            return ws, intensity
        else:
            return ws

    @overload
    def log(
        self,
        so3_or_Twist3: Literal["so3"] = "so3",
        return_intensity: Literal[False] = False,
        normalize: bool = False,
    ) -> lieink.atoms.so3: ...

    @overload
    def log(
        self,
        so3_or_Twist3: Literal["so3"] = "so3",
        *,
        return_intensity: Literal[True],
        normalize: bool = False,
    ) -> tuple[lieink.atoms.so3, RealScalar]: ...

    @overload
    def log(
        self,
        so3_or_Twist3: Literal["Twist3"],
        return_intensity: Literal[False] = False,
        normalize: bool = False,
    ) -> lieink.atoms.Twist3: ...

    @overload
    def log(
        self,
        so3_or_Twist3: Literal["Twist3"],
        return_intensity: Literal[True],
        normalize: bool = False,
    ) -> tuple[lieink.atoms.Twist3, RealScalar]: ...

    def log(
        self,
        so3_or_Twist3: Literal["so3", "Twist3"] = "so3",
        return_intensity: bool = False,
        normalize: bool = False,
    ) -> (
        lieink.atoms.so3
        | tuple[lieink.atoms.so3, RealScalar]
        | lieink.atoms.Twist3
        | tuple[lieink.atoms.Twist3, RealScalar]
    ):

        logc_func: Any = self.logc
        check_operator_func: Any = self._check_operator

        self_v = self._check_assigned()
        v = logc_func(
            self_v, so3_or_Twist3, return_intensity, normalize, skip_check=True
        )
        result_type = check_operator_func("log", None)[so3_or_Twist3]
        if return_intensity:
            return result_type(v[0]), v[1]
        return result_type(v)


@beartype
class LieGroup[
    allowed_input_type_single: NDArray,
    output_type_single: NDArray,
    allowed_input_type_batch: NDArray,
    output_type_batch: NDArray,
](
    BasicLieGroup[
        allowed_input_type_single,
        output_type_single,
        allowed_input_type_batch,
        output_type_batch,
    ],
    ABC,
):
    @classmethod
    @abstractmethod
    def Rc(cls, v: allowed_input_type_single, skip_check: bool = False) -> NDArray_3_3:
        pass

    @classmethod
    @abstractmethod
    def Rcb(
        cls, v: allowed_input_type_batch, skip_check: bool = False
    ) -> NDArray_N_3_3:
        pass

    @property
    def R(self) -> lieink.atoms.SO3:
        from lieink.atoms import SO3

        v = self._check_assigned()
        return SO3(self.Rc(v, skip_check=True))

    @classmethod
    @abstractmethod
    def t3c(cls, v: allowed_input_type_single, skip_check: bool = False) -> NDArray_3_1:
        pass

    @classmethod
    @abstractmethod
    def t3cb(
        cls, v: allowed_input_type_batch, skip_check: bool = False
    ) -> NDArray_N_3_1:
        pass

    @property
    def t3(self) -> lieink.atoms.Point3:
        v = self._check_assigned()
        from lieink.atoms import Point3

        return Point3(self.t3c(v, skip_check=True))

    @classmethod
    def t4c(cls, v: allowed_input_type_single, skip_check: bool = False) -> NDArray_4_1:
        t3 = cls.t3c(v, skip_check=skip_check)
        t4 = np.ones((4, 1))
        t4[:3] = t3
        return t4

    @classmethod
    def t4cb(
        cls, v: allowed_input_type_batch, skip_check: bool = False
    ) -> NDArray_N_4_1:
        t3b = cls.t3cb(v, skip_check=skip_check)
        t4b = np.ones((v.shape[0], 4, 1))
        t4b[:, :3] = t3b
        return t4b

    @property
    def t4(self) -> lieink.atoms.Point4:
        v = self._check_assigned()
        from lieink.atoms import Point4

        return Point4(self.t4c(v, skip_check=True))

    @classmethod
    def _combine_basec(
        cls,
        rotation_matrix: NDArray_3_3,
        t: NDArray_3_1 | NDArray_3 | NDArray_4_1 | NDArray_4,
        combineto: Literal["SE3", "Ad", "coAd"] = "SE3",
        skip_check: bool = False,
    ) -> NDArray_4_4 | NDArray_6_6:

        if not skip_check:
            if utils.check_rotation_matrix(rotation_matrix) is False:
                raise ValueError("rotation_matrix is not a rotation matrix")
            if t.ndim == 1:
                t = t.reshape(-1, 1)

        if t.shape[0] == 3:
            pass
        else:
            if utils.check_equal(t[-1], 0) is not False:
                raise ValueError(
                    "t[3] is zero, which means it is a Vector4 instead of Point4."
                )
            t = t[:3] / t[-1]

        if combineto == "SE3":
            v = np.eye(4)
            v[:3, :3] = rotation_matrix
            v[:3, 3:] = t
        elif combineto == "Ad":
            v = np.zeros((6, 6))
            v[:3, :3] = rotation_matrix
            v[3:, 3:] = rotation_matrix
            v[3:, :3] = utils.skew3(t) @ rotation_matrix
        else:
            v = np.zeros((6, 6))
            v[:3, :3] = rotation_matrix
            v[3:, 3:] = rotation_matrix
            v[:3, 3:] = utils.skew3(t) @ rotation_matrix

        return v

    @classmethod
    def _combine_basecb(
        cls,
        rotation_matrixb: NDArray_N_3_3,
        tb: NDArray_N_3_1 | NDArray_3_N | NDArray_N_4_1 | NDArray_4_N,
        combineto: Literal["SE3", "Ad", "coAd"] = "SE3",
        skip_check: bool = False,
    ) -> NDArray_N_4_4 | NDArray_N_6_6:

        if not skip_check:
            if utils.check_rotation_matrixb(rotation_matrixb) is False:
                raise ValueError("rotation_matrixb is not all rotation matrices")
            if tb.ndim == 2:
                tb = tb.reshape(tb.shape[0], -1, 1).swapaxes(0, 1)
            if tb.shape[0] != rotation_matrixb.shape[0]:
                raise ValueError("rotation_matrixb and tb must have the same num.")
        if tb.shape[1] == 4:
            if utils.find_zeros(tb[:, -1]).any():
                raise ValueError(
                    "tb[3] is zero, which means it is a Vector4 instead of Point4."
                )
            tb = tb[:, :3] / tb[:, -1:]

        num = rotation_matrixb.shape[0]

        if combineto == "SE3":
            v = np.zeros((num, 4, 4))
            v[:, -1, -1] = 1
            v[:, :3, :3] = rotation_matrixb
            v[:, :3, 3:] = tb
        elif combineto == "Ad":
            v = np.zeros((num, 6, 6))
            v[:, :3, :3] = rotation_matrixb
            v[:, 3:, 3:] = rotation_matrixb
            v[:, 3:, :3] = utils.skew3b(tb) @ rotation_matrixb
        else:
            v = np.zeros((6, 6))
            v[:, :3, :3] = rotation_matrixb
            v[:, 3:, 3:] = rotation_matrixb
            v[:, :3, 3:] = utils.skew3b(tb) @ rotation_matrixb

        return v

    @classmethod
    def combinec(
        cls,
        rotation_matrix: NDArray_3_3,
        t: NDArray_3_1 | NDArray_3 | NDArray_4_1 | NDArray_4,
        skip_check: bool = False,
    ) -> output_type_single:

        mytype_name = utils.get_TypeName(cls._type())
        return cls._combine_basec(rotation_matrix, t, mytype_name, skip_check)  # type: ignore

    @classmethod
    def combinecb(
        cls,
        rotation_matrixb: NDArray_N_3_3,
        tb: NDArray_N_3_1 | NDArray_3_N | NDArray_N_4_1 | NDArray_4_N,
        skip_check: bool = False,
    ) -> output_type_batch:

        mytype_name = utils.get_TypeName(cls._type())
        return cls._combine_basecb(rotation_matrixb, tb, mytype_name, skip_check)  # type: ignore

    def combine(
        self,
        SO3_instance: lieink.atoms.SO3 | NDArray_3_3,
        Point_instance: lieink.atoms.Point3
        | lieink.atoms.Point4
        | NDArray_3_1
        | NDArray_3
        | NDArray_4_1
        | NDArray_4,
        skip_check: bool = False,
    ) -> Self:

        from lieink.atoms import SO3

        if isinstance(SO3_instance, SO3):
            rotation_matrix = SO3_instance._check_assigned()  # type: ignore
        else:
            rotation_matrix = SO3_instance

        from lieink.atoms import Point3, Point4

        if isinstance(Point_instance, (Point3, Point4)):
            t = Point_instance._check_assigned()  # type: ignore
        else:
            t = Point_instance

        return self.mytype(self.combinec(rotation_matrix, t, skip_check=skip_check))  # type: ignore

    @classmethod
    @abstractmethod
    def _trans_basec(
        cls, axis: Literal["x", "y", "z"], theta: RealScalar
    ) -> allowed_input_type_single:
        pass

    @classmethod
    @abstractmethod
    def _trans_basecb(
        cls, axis: Literal["x", "y", "z"], theta: NDArray_1D | RealScalar
    ) -> allowed_input_type_batch:
        pass

    @classmethod
    def _transAxisc(
        cls,
        axis: Literal["x", "y", "z"],
        theta: RealScalar,
        v: allowed_input_type_single | None = None,
        left_or_right: Literal["left", "right"] = "right",
        skip_check: bool = False,
    ) -> allowed_input_type_single:

        v_ = cls._trans_basec(axis, theta)

        if v is None:
            return v_
        else:
            v = cls._check_shape_and_value(v, skip_check=skip_check)
            if left_or_right == "right":
                return v @ v_  # type: ignore
            else:
                return v_ @ v  # type: ignore

    @classmethod
    def _transAxiscb(
        cls,
        axis: Literal["x", "y", "z"],
        thetas: NDArray_1D | RealScalar,
        v: allowed_input_type_batch | None = None,
        left_or_right: Literal["left", "right"] = "right",
        skip_check: bool = False,
    ) -> allowed_input_type_batch:

        v_ = cls._trans_basecb(axis, thetas)

        if v is None:
            return v_
        else:
            v = cls._check_shapeb_and_valueb(v, skip_check=skip_check)
            if v.shape[0] != v_.shape[0]:
                raise ValueError(f"v and {axis} must have the same number of elements.")
            if left_or_right == "right":
                return Lie.mulcb(v, v_)  # type: ignore
            else:
                return Lie.mulcb(v_, v)  # type: ignore

    @classmethod
    def transxc(
        cls,
        x: RealScalar,
        v: allowed_input_type_single | None = None,
        left_or_right: Literal["left", "right"] = "right",
        skip_check: bool = False,
    ) -> output_type_single:

        return cls._transAxisc("x", x, v, left_or_right, skip_check)  # type: ignore

    @classmethod
    def transxcb(
        cls,
        x: NDArray_1D | RealScalar,
        v: allowed_input_type_batch | None = None,
        left_or_right: Literal["left", "right"] = "right",
        skip_check: bool = False,
    ) -> output_type_batch:

        return cls._transAxiscb("x", x, v, left_or_right, skip_check)  # type: ignore

    def transx(
        self,
        x: RealScalar,
        left_or_right: Literal["left", "right"] = "right",
        inplace: bool = False,
    ) -> Self:

        self_v = self.transxc(x, self._v, left_or_right, skip_check=True)  # type: ignore
        return self._inplace_or_not(self_v, inplace)

    @classmethod
    def transyc(
        cls,
        y: RealScalar,
        v: allowed_input_type_single | None = None,
        left_or_right: Literal["left", "right"] = "right",
        skip_check: bool = False,
    ) -> output_type_single:

        return cls._transAxisc("y", y, v, left_or_right, skip_check)  # type: ignore

    @classmethod
    def transycb(
        cls,
        y: NDArray_1D | RealScalar,
        v: allowed_input_type_batch | None = None,
        left_or_right: Literal["left", "right"] = "right",
        skip_check: bool = False,
    ) -> output_type_batch:

        return cls._transAxiscb("y", y, v, left_or_right, skip_check)  # type: ignore

    def transy(
        self,
        y: RealScalar,
        left_or_right: Literal["left", "right"] = "right",
        inplace: bool = False,
    ) -> Self:

        self_v = self.transyc(y, self._v, left_or_right, skip_check=True)  # type: ignore
        return self._inplace_or_not(self_v, inplace)

    @classmethod
    def transzc(
        cls,
        z: RealScalar,
        v: allowed_input_type_single | None = None,
        left_or_right: Literal["left", "right"] = "right",
        skip_check: bool = False,
    ) -> output_type_single:

        return cls._transAxisc("z", z, v, left_or_right, skip_check)  # type: ignore

    @classmethod
    def transzcb(
        cls,
        z: NDArray_1D | RealScalar,
        v: allowed_input_type_batch | None = None,
        left_or_right: Literal["left", "right"] = "right",
        skip_check: bool = False,
    ) -> output_type_batch:

        return cls._transAxiscb("z", z, v, left_or_right, skip_check)  # type: ignore

    def transz(
        self,
        z: RealScalar,
        left_or_right: Literal["left", "right"] = "right",
        inplace: bool = False,
    ) -> Self:

        self_v = self.transzc(z, self._v, left_or_right, skip_check=True)  # type: ignore
        return self._inplace_or_not(self_v, inplace)

    @classmethod
    def _transSeriesc(
        cls,
        axes: Iterable[Literal["x", "y", "z"]],
        thetas: NDArray_1D | RealScalar,
        v: allowed_input_type_single | None = None,
        left_or_right: Literal["left", "right"] = "right",
        skip_check: bool = False,
    ) -> allowed_input_type_single:

        num_axis = len(axes)  # type: ignore
        if utils.is_realscaler(thetas) is not False:
            thetas = np.array([thetas] * num_axis)
        else:
            thetas = cast(NDArray_1D, thetas)
            if thetas.shape[0] != num_axis:
                raise ValueError("thetas and axes must have the same length")
        if v is None:
            v = cls.init_liegroupc()  # type: ignore
        else:
            v = cls._check_shape_and_value(v, skip_check=skip_check)

        for i in range(num_axis):
            v = cls._transAxisc(axes[i], thetas[i], v, left_or_right, skip_check)  # type: ignore

        return v  # type: ignore

    @classmethod
    def _transSeriescb(
        cls,
        axes: Iterable[Literal["x", "y", "z"]],
        thetas: NDArray_1D | RealScalar,
        v: allowed_input_type_batch | None = None,
        left_or_right: Literal["left", "right"] = "right",
        skip_check: bool = False,
    ) -> allowed_input_type_batch:

        num_axis = len(axes)  # type: ignore
        if utils.is_realscaler(thetas) is not False:
            thetas = np.array([thetas] * num_axis)
        else:
            thetas = cast(NDArray_1D, thetas)
            if thetas.shape[0] != num_axis:
                raise ValueError("thetas and axes must have the same length")
        if v is None:
            v = cls.init_liegroupcb(1)  # type: ignore
        else:
            v = cls._check_shapeb_and_valueb(v, skip_check=skip_check)

        for i in range(num_axis):
            v = cls._transAxiscb(axes[i], thetas[i], v, left_or_right, skip_check)  # type: ignore

        return v  # type: ignore

    def transSeries(
        self,
        axes: Iterable[Literal["x", "y", "z"]],
        thetas: NDArray_1D | RealScalar,
        left_or_right: Literal["left", "right"] = "right",
        inplace: bool = False,
    ) -> Self:
        self_v = self._transSeriesc(
            axes,
            thetas,
            self._v,  # type: ignore
            left_or_right,
            skip_check=True,
        )
        return self._inplace_or_not(self_v, inplace)  # type: ignore

    @overload
    @classmethod
    def logc(
        cls,
        v: allowed_input_type_single,
        se3_or_Twist_or_ad_or_coad: Literal["se3"] = "se3",
        return_intensity: Literal[False] = False,
        normalize: bool = False,
        skip_check: bool = False,
    ) -> NDArray_4_4: ...

    @overload
    @classmethod
    def logc(
        cls,
        v: allowed_input_type_single,
        se3_or_Twist_or_ad_or_coad: Literal["se3"] = "se3",
        *,
        return_intensity: Literal[True],
        normalize: bool = False,
        skip_check: bool = False,
    ) -> tuple[NDArray_4_4, RealScalar]: ...

    @overload
    @classmethod
    def logc(
        cls,
        v: allowed_input_type_single,
        se3_or_Twist_or_ad_or_coad: Literal["Twist"],
        return_intensity: Literal[False] = False,
        normalize: bool = False,
        skip_check: bool = False,
    ) -> NDArray_6_1: ...

    @overload
    @classmethod
    def logc(
        cls,
        v: allowed_input_type_single,
        se3_or_Twist_or_ad_or_coad: Literal["Twist"],
        return_intensity: Literal[True],
        normalize: bool = False,
        skip_check: bool = False,
    ) -> tuple[NDArray_6_1, RealScalar]: ...

    @overload
    @classmethod
    def logc(
        cls,
        v: allowed_input_type_single,
        se3_or_Twist_or_ad_or_coad: Literal["ad", "coad"],
        return_intensity: Literal[False] = False,
        normalize: bool = False,
        skip_check: bool = False,
    ) -> NDArray_6_6: ...

    @overload
    @classmethod
    def logc(
        cls,
        v: allowed_input_type_single,
        se3_or_Twist_or_ad_or_coad: Literal["ad", "coad"],
        return_intensity: Literal[True],
        normalize: bool = False,
        skip_check: bool = False,
    ) -> tuple[NDArray_6_6, RealScalar]: ...

    @classmethod
    def logc(
        cls,
        v: allowed_input_type_single,
        se3_or_Twist_or_ad_or_coad: Literal["se3", "Twist", "ad", "coad"] = "se3",
        return_intensity: bool = False,
        normalize: bool = False,
        skip_check: bool = False,
    ) -> (
        NDArray_4_4
        | tuple[NDArray_4_4, RealScalar]
        | NDArray_6_1
        | tuple[NDArray_6_1, RealScalar]
        | NDArray_6_6
        | tuple[NDArray_6_6, RealScalar]
    ):

        v = cls._check_shape_and_value(v, skip_check=skip_check)
        rotation_matrix_v = cls.Rc(v, skip_check=True)
        t_v = cls.t3c(v, skip_check=True)

        skew_w, intensity = LieGroup3.logc(  # type: ignore
            rotation_matrix_v,
            "so3",
            return_intensity=True,
            normalize=True,
            skip_check=True,
        )
        vee_w = utils.veeskew3(skew_w)

        if utils.check_equal(intensity, 0) is not False:
            vee_v = t_v
            intensity = np.linalg.norm(vee_v)
            if utils.check_equal(intensity, 0) is False:
                vee_v = vee_v / intensity
        else:
            A = (np.eye(3) - rotation_matrix_v) @ skew_w + intensity * vee_w @ vee_w.T
            vee_v = np.linalg.solve(A, t_v)

        if normalize is False:
            skew_w = skew_w * intensity
            vee_v = vee_v * intensity

        v = LieAlgebra._combine_basec(  # type: ignore
            skew_w, vee_v, se3_or_Twist_or_ad_or_coad, skip_check=True
        )

        if return_intensity:
            return v, intensity
        else:
            return v

    @overload
    @classmethod
    def logcb(
        cls,
        v: allowed_input_type_batch,
        se3_or_Twist_or_ad_or_coad: Literal["se3"] = "se3",
        return_intensity: Literal[False] = False,
        normalize: bool = False,
        skip_check: bool = False,
    ) -> NDArray_N_4_4: ...

    @overload
    @classmethod
    def logcb(
        cls,
        v: allowed_input_type_batch,
        se3_or_Twist_or_ad_or_coad: Literal["se3"] = "se3",
        *,
        return_intensity: Literal[True],
        normalize: bool = False,
        skip_check: bool = False,
    ) -> tuple[NDArray_N_4_4, NDArray_1D]: ...

    @overload
    @classmethod
    def logcb(
        cls,
        v: allowed_input_type_batch,
        se3_or_Twist_or_ad_or_coad: Literal["Twist"],
        return_intensity: Literal[False] = False,
        normalize: bool = False,
        skip_check: bool = False,
    ) -> NDArray_N_6_1: ...

    @overload
    @classmethod
    def logcb(
        cls,
        v: allowed_input_type_batch,
        se3_or_Twist_or_ad_or_coad: Literal["Twist"],
        *,
        return_intensity: Literal[True],
        normalize: bool = False,
        skip_check: bool = False,
    ) -> tuple[NDArray_N_6_1, NDArray_1D]: ...

    @overload
    @classmethod
    def logcb(
        cls,
        v: allowed_input_type_batch,
        se3_or_Twist_or_ad_or_coad: Literal["ad", "coad"],
        return_intensity: Literal[False] = False,
        normalize: bool = False,
        skip_check: bool = False,
    ) -> NDArray_N_6_6: ...

    @overload
    @classmethod
    def logcb(
        cls,
        v: allowed_input_type_batch,
        se3_or_Twist_or_ad_or_coad: Literal["ad", "coad"],
        *,
        return_intensity: Literal[True],
        normalize: bool = False,
        skip_check: bool = False,
    ) -> tuple[NDArray_N_6_6, NDArray_1D]: ...

    @classmethod
    def logcb(
        cls,
        v: allowed_input_type_batch,
        se3_or_Twist_or_ad_or_coad: Literal["se3", "Twist", "ad", "coad"] = "se3",
        return_intensity: bool = False,
        normalize: bool = False,
        skip_check: bool = False,
    ) -> (
        NDArray_N_4_4
        | tuple[NDArray_N_4_4, NDArray_1D]
        | NDArray_N_6_1
        | tuple[NDArray_N_6_1, NDArray_1D]
        | NDArray_N_6_6
        | tuple[NDArray_N_6_6, NDArray_1D]
    ):

        v = cls._check_shapeb_and_valueb(v, skip_check=skip_check)
        rotation_matrix_vb = cls.Rcb(v, skip_check=True)
        t_vb = cls.t3cb(v, skip_check=True)

        skew_wb, intensity_wb = LieGroup3.logcb(  # type: ignore
            rotation_matrix_vb,
            "so3",
            return_intensity=True,
            normalize=True,
            skip_check=True,
        )
        vee_wb = utils.veeskew3b(skew_wb)
        mask = utils.find_zeros(intensity_wb)
        mask_ = np.logical_not(mask)
        vee_vb = np.zeros(vee_wb.shape)

        intensityb = np.zeros(intensity_wb.shape)
        intensityb[mask_] = intensity_wb[mask_]

        if mask.any():
            intensityb[mask] = np.linalg.norm(t_vb[mask], axis=1).flatten()
            mask_intensityb = np.logical_not(utils.find_zeros(intensityb))
            vee_vb[mask_intensityb] = t_vb[mask_intensityb] / intensityb[
                mask_intensityb
            ].reshape(-1, 1, 1)

        if mask_.any():
            vee_wb_2 = np.einsum("nij,nkj->nik", vee_wb[mask_], vee_wb[mask_])  # type: ignore
            Ab = (np.eye(3) - rotation_matrix_vb[mask_]) @ skew_wb[mask_] + intensityb[
                mask_
            ].reshape(-1, 1, 1) * vee_wb_2
            vee_vb[mask_] = np.linalg.solve(Ab, t_vb[mask_])

        if normalize is False:
            intensityb = intensityb.reshape(-1, 1, 1)
            skew_wb = skew_wb * intensityb
            vee_wb = vee_wb * intensityb
            vee_vb = vee_vb * intensityb
            intensityb = intensityb.flatten()

        v = LieAlgebra._combine_basecb(  # type: ignore
            skew_wb, vee_vb, se3_or_Twist_or_ad_or_coad, skip_check=True
        )

        if return_intensity:
            return v, intensityb
        else:
            return v

    @overload
    def log(
        self,
        se3_or_Twist_or_ad_or_coad: Literal["se3"] = "se3",
        return_intensity: Literal[False] = False,
        normalize: bool = False,
    ) -> lieink.atoms.se3: ...

    @overload
    def log(
        self,
        se3_or_Twist_or_ad_or_coad: Literal["se3"] = "se3",
        *,
        return_intensity: Literal[True],
        normalize: bool = False,
    ) -> tuple[lieink.atoms.se3, RealScalar]: ...

    @overload
    def log(
        self,
        se3_or_Twist_or_ad_or_coad: Literal["Twist"],
        return_intensity: Literal[False] = False,
        normalize: bool = False,
    ) -> lieink.atoms.Twist: ...

    @overload
    def log(
        self,
        se3_or_Twist_or_ad_or_coad: Literal["Twist"],
        *,
        return_intensity: Literal[True],
        normalize: bool = False,
    ) -> tuple[lieink.atoms.Twist, RealScalar]: ...

    @overload
    def log(
        self,
        se3_or_Twist_or_ad_or_coad: Literal["ad"],
        return_intensity: Literal[False] = False,
        normalize: bool = False,
    ) -> lieink.atoms.ad: ...

    @overload
    def log(
        self,
        se3_or_Twist_or_ad_or_coad: Literal["ad"],
        *,
        return_intensity: Literal[True],
        normalize: bool = False,
    ) -> tuple[lieink.atoms.ad, RealScalar]: ...

    @overload
    def log(
        self,
        se3_or_Twist_or_ad_or_coad: Literal["coad"],
        return_intensity: Literal[False] = False,
        normalize: bool = False,
    ) -> lieink.atoms.coad: ...

    @overload
    def log(
        self,
        se3_or_Twist_or_ad_or_coad: Literal["coad"],
        *,
        return_intensity: Literal[True],
        normalize: bool = False,
    ) -> tuple[lieink.atoms.coad, RealScalar]: ...

    def log(
        self,
        se3_or_Twist_or_ad_or_coad: Literal["se3", "Twist", "ad", "coad"] = "se3",
        return_intensity: bool = False,
        normalize: bool = False,
    ) -> (
        lieink.atoms.se3
        | tuple[lieink.atoms.se3, RealScalar]
        | lieink.atoms.Twist
        | tuple[lieink.atoms.Twist, RealScalar]
        | lieink.atoms.ad
        | tuple[lieink.atoms.ad, RealScalar]
        | lieink.atoms.coad
        | tuple[lieink.atoms.coad, RealScalar]
    ):
        logc_func: Any = self.logc
        check_operator_func: Any = self._check_operator

        self_v = self._check_assigned()
        v = logc_func(
            self_v,
            se3_or_Twist_or_ad_or_coad,
            return_intensity,
            normalize,
            skip_check=True,
        )
        result_type = check_operator_func("log", None)[se3_or_Twist_or_ad_or_coad]
        if return_intensity:
            return result_type(v[0]), v[1]
        return result_type(v)

    @classmethod
    def toSE3c(
        cls, v: allowed_input_type_single, skip_check: bool = False
    ) -> NDArray_4_4:
        R = cls.Rc(v, skip_check=skip_check)
        t3 = cls.t3c(v, skip_check=True)
        return cls._combine_basec(R, t3, "SE3", skip_check=True)

    @classmethod
    def toSE3cb(
        cls, v: allowed_input_type_batch, skip_check: bool = False
    ) -> NDArray_N_4_4:
        R = cls.Rcb(v, skip_check=skip_check)
        t3 = cls.t3cb(v, skip_check=True)
        return cls._combine_basecb(R, t3, "SE3", skip_check=True)

    def toSE3(self) -> lieink.atoms.SE3:
        self_v = self._check_assigned()
        from lieink.atoms import SE3

        return SE3(self.toSE3c(self_v, skip_check=True))

    @classmethod
    def toAdc(
        cls, v: allowed_input_type_single, skip_check: bool = False
    ) -> NDArray_6_6:
        R = cls.Rc(v, skip_check=skip_check)
        t3 = cls.t3c(v, skip_check=True)
        return cls._combine_basec(R, t3, "Ad", skip_check=True)

    @classmethod
    def toAdcb(
        cls, v: allowed_input_type_batch, skip_check: bool = False
    ) -> NDArray_N_6_6:
        R = cls.Rcb(v, skip_check=skip_check)
        t3 = cls.t3cb(v, skip_check=True)
        return cls._combine_basecb(R, t3, "Ad", skip_check=True)

    def toAd(self) -> lieink.atoms.Ad:
        self_v = self._check_assigned()
        from lieink.atoms import Ad

        return Ad(self.toAdc(self_v, skip_check=True))

    @classmethod
    def tocoAdc(
        cls, v: allowed_input_type_single, skip_check: bool = False
    ) -> NDArray_6_6:
        R = cls.Rc(v, skip_check=skip_check)
        t3 = cls.t3c(v, skip_check=True)
        return cls._combine_basec(R, t3, "coAd", skip_check=True)

    @classmethod
    def tocoAdcb(
        cls, v: allowed_input_type_batch, skip_check: bool = False
    ) -> NDArray_N_6_6:
        R = cls.Rcb(v, skip_check=skip_check)
        t3 = cls.t3cb(v, skip_check=True)
        return cls._combine_basecb(R, t3, "coAd", skip_check=True)

    def tocoAd(self) -> lieink.atoms.coAd:
        self_v = self._check_assigned()
        from lieink.atoms import coAd

        return coAd(self.tocoAdc(self_v, skip_check=True))


ALLOWED_OPERATORS: dict[
    str,
    dict[
        type[Any] | TypeAliasType,
        dict[
            type[Any] | TypeAliasType | None,
            type[Any] | TypeAliasType | dict[str, type[Any] | TypeAliasType],
        ],
    ],
] = {}
