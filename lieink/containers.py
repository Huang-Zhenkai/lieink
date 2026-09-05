from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal

import numpy as np
from beartype import beartype
from scipy.linalg import block_diag  # type: ignore

from lieink import utils
from lieink.annotations import NDArray, NDArray_2D, NDArray_3D
from lieink.basic_atoms import BasicLie as _BasicLie
from lieink.basic_atoms import Lie
from lieink.basic_containers import BasicContainer as _BasicContainer


@beartype
class Container(_BasicContainer[Any]):
    def __init__(
        self,
        content: Iterable[Any] | Any | None = None,
    ):
        super().__init__(content)  # type: ignore


@beartype
class LieContainer(_BasicContainer[_BasicLie[Any, Any, Any, Any]]):
    def __init__(
        self,
        content: Iterable[_BasicLie[Any, Any, Any, Any]]
        | _BasicLie[Any, Any, Any, Any]
        | type[_BasicLie[Any, Any, Any, Any]]
        | None = None,
    ):
        if isinstance(content, type):
            self.content_type = content
        super().__init__(content)  # type: ignore

    def _check_type(self, item: Any) -> Any:
        if self.content_type is None and not isinstance(item, _BasicLie):  # type: ignore
            raise TypeError(
                f"The input item is an instance of {utils.get_Type(item)} instead of _BasicLie"
            )
        return super()._check_type(item)

    def extend(self, other_list: Iterable[Any] | NDArray):  # type:ignore
        if isinstance(other_list, np.ndarray):
            other_list = self.content_type.reshapeb(other_list)  # type: ignore
        return super().extend(other_list)  # type: ignore

    @property
    def shape(self) -> tuple[int, int, int]:
        return len(self), self.content_type._r, self.content_type._c  # type: ignore

    def _to_array_list(
        self,
        items: _BasicLie[Any, Any, Any, Any]
        | Iterable[_BasicLie[Any, Any, Any, Any]]
        | NDArray
        | Iterable[NDArray]
        | None,
    ) -> list[NDArray]:

        if items is None:
            return []
        if isinstance(items, np.ndarray):
            items = self.content_type.reshapeb(items)
            items = list(items)  # type: ignore
            return items  # type: ignore
        elif isinstance(items, _BasicLie):
            if self.content_type._r == items._r and self.content_type._c == items._c:  # type: ignore
                items = [items.v]
                return items
            else:
                raise ValueError("Shape mismatch")
        else:
            result = []
            for it in items:  # type: ignore
                if isinstance(it, np.ndarray):
                    result.append(it)  # type: ignore
                else:
                    if (
                        self.content_type._r == items._r  # type: ignore
                        and self.content_type._c == items._c  # type: ignore
                    ):
                        result.append(it.v)  # type: ignore
                    else:
                        raise ValueError("Shape mismatch")
            return result  # type: ignore

    def toNDArray_3D(
        self,
        index: None | Iterable[int] = None,
        head_add: None
        | _BasicLie[Any, Any, Any, Any]
        | Iterable[_BasicLie[Any, Any, Any, Any]]
        | NDArray
        | Iterable[NDArray] = None,
        tail_add: None
        | _BasicLie[Any, Any, Any, Any]
        | Iterable[_BasicLie[Any, Any, Any, Any]]
        | NDArray
        | Iterable[NDArray] = None,
    ) -> NDArray_3D:

        length = self._check_not_empty()

        if index is None:
            index = list(range(length))

        self_arrays = [self[i].v for i in index]

        head_arrays = self._to_array_list(head_add)
        tail_arrays = self._to_array_list(tail_add)
        all_arrays = head_arrays + self_arrays + tail_arrays
        if not all_arrays:
            return np.empty((0, 0, 0))

        return np.stack(all_arrays, axis=0)

    def toNDArray_2D(
        self,
        format: Literal["r", "c", "d"],
        index: None | Iterable[int] = None,
        head_add: None
        | _BasicLie[Any, Any, Any, Any]
        | Iterable[_BasicLie[Any, Any, Any, Any]]
        | NDArray
        | Iterable[NDArray] = None,
        tail_add: None
        | _BasicLie[Any, Any, Any, Any]
        | Iterable[_BasicLie[Any, Any, Any, Any]]
        | NDArray
        | Iterable[NDArray] = None,
    ) -> NDArray_2D:

        all_arrays = self.toNDArray_3D(index, head_add, tail_add)

        if all_arrays.shape[0] == 0:
            return np.empty((0, 0))

        if format == "r":
            return np.vstack(all_arrays)  # type: ignore
        elif format == "c":
            return np.hstack(all_arrays)  # type: ignore
        else:
            return block_diag(*all_arrays)  # type: ignore

    def toLie(
        self,
        format: Literal["r", "c", "d"],
        index: None | Iterable[int] = None,
        head_add: None
        | _BasicLie[Any, Any, Any, Any]
        | Iterable[_BasicLie[Any, Any, Any, Any]]
        | NDArray
        | Iterable[NDArray] = None,
        tail_add: None
        | _BasicLie[Any, Any, Any, Any]
        | Iterable[_BasicLie[Any, Any, Any, Any]]
        | NDArray
        | Iterable[NDArray] = None,
    ) -> Lie:
        return Lie(self.toNDArray_2D(format, index, head_add, tail_add))
