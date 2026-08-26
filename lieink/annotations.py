import re
from typing import Any, Literal

import numpy as np
from beartype import beartype
from jaxtyping import Bool, Num

# 实数标量类型注释
type RealScalar = float | int | np.floating[Any] | np.integer[Any]

# 矩阵类型注释
NDArray = np.ndarray[Any, np.dtype[Any]]


NDArrayBool = Bool[NDArray, "*"]
NDArrayBool_1D = Bool[NDArray, "N"]
NDArrayBool_2D = Bool[NDArray, "N M"]
NDArrayBool_3D = Bool[NDArray, "N M K"]
NDArrayBool_4D = Bool[NDArray, "N M K L"]

NDArray_1D = Num[NDArray, "N"]
NDArray_2D = Num[NDArray, "N M"]
NDArray_3D = Num[NDArray, "N M K"]
NDArray_4D = Num[NDArray, "N M K L"]

NDArray_3_1 = Num[NDArray, "3 1"]
NDArray_3 = Num[NDArray, "3"]
NDArray_4_1 = Num[NDArray, "4 1"]
NDArray_4 = Num[NDArray, "4"]
NDArray_6_1 = Num[NDArray, "6 1"]
NDArray_6 = Num[NDArray, "6"]

NDArray_3_3 = Num[NDArray, "3 3"]
NDArray_4_4 = Num[NDArray, "4 4"]
NDArray_6_6 = Num[NDArray, "6 6"]

NDArray_N_3_1 = Num[NDArray, "N 3 1"]

NDArray_N_3 = Num[NDArray, "N 3"]
NDArray_N_4_1 = Num[NDArray, "N 4 1"]

NDArray_N_4 = Num[NDArray, "N 4"]
NDArray_N_6_1 = Num[NDArray, "N 6 1"]
NDArray_N_6 = Num[NDArray, "N 6"]

NDArray_N_3_3 = Num[NDArray, "N 3 3"]

NDArray_N_4_4 = Num[NDArray, "N 4 4"]

NDArray_N_6_6 = Num[NDArray, "N 6 6"]


NDArray_3_N = Num[NDArray, "3 N"]
NDArray_4_N = Num[NDArray, "4 N"]
NDArray_6_N = Num[NDArray, "6 N"]


@beartype
def get_NDArrayAnnotation(array: NDArray) -> Any:

    shape = array.shape
    shape = [str(i) for i in shape]
    shape_str = " ".join(shape)

    if array.dtype == np.bool_:
        return Bool[NDArray, shape_str]

    return Num[NDArray, shape_str]


def is_NDArrayAnnotation(type_hint: Any) -> Any | Literal[False]:

    if "jaxtyping" not in str(type_hint):
        return False
    else:
        return type_hint


def get_NDArrayAnnotationShape(type_hint: Any) -> list[str | int]:

    if not is_NDArrayAnnotation(type_hint):
        raise TypeError("type_hint must be NDArrayAnnotations")

    target_hint = type_hint

    hint_str = str(target_hint)
    match = re.search(r"\[.*?,\s*['\"](.*?)['\"]\]", hint_str)

    shape = match
    if shape is None:
        raise TypeError("type_hint must be NDArrayAnnotations")
    shape = shape.group(1).split(" ")
    shape = [int(i) if i.isdigit() else i for i in shape]

    return shape
