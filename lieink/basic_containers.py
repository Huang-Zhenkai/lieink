from __future__ import annotations

from collections import UserList
from collections.abc import Callable, Iterable
from types import NotImplementedType
from typing import (
    Any,
    Self,
    TypeAliasType,
)

import numpy as np
from beartype import beartype
from rich import print
from rich.pretty import pprint
from rich.text import Text

from lieink import utils


@beartype
class BasicContainer[T](UserList[T]):
    @classmethod
    def create_a_container(cls, instance: Any) -> Any:
        content_type = utils.get_Type(instance)
        container_type = CONTAINER_TYPES.get(content_type)
        if container_type is None:
            allowed_types = list(CONTAINER_TYPES.keys())
            allowed_types.reverse()
            for allowed_type in allowed_types[:-1]:
                if issubclass(content_type, allowed_type):  # type: ignore
                    return CONTAINER_TYPES[allowed_type]([instance])  # type: ignore
            return CONTAINER_TYPES[None]([instance])
        return container_type([instance])

    @classmethod
    def PRINT_CONTAINER_TYPES(cls):
        pprint(CONTAINER_TYPES)

    @classmethod
    def UPDATE_CONTAINER_TYPES(
        cls,
        content_type: type | TypeAliasType | None,
        container_type: type[BasicContainer[Any]],
    ):
        if CONTAINER_TYPES.get(content_type) is None:
            CONTAINER_TYPES[content_type] = container_type
        else:
            raise ValueError(
                f"Container type {container_type} for content type {content_type} already exists."
            )

    def __init__(
        self, content: Iterable[T] | type[T] | T | TypeAliasType | None = None
    ):
        if "content_type" not in self.__dict__:
            self.content_type = None
        super().__init__(None)
        if isinstance(content, Iterable) and not isinstance(content, (str, np.ndarray)):
            content = list(content)  # type: ignore
            self.extend(content)
        elif isinstance(content, (type, TypeAliasType)):
            self.content_type = content
        elif content is None:
            pass
        else:
            self.append(content)  # type: ignore

    def _check_type(self, item: Any) -> T:
        if self.content_type is None:
            self.content_type = utils.get_Type(item)
        else:
            item_type = utils.get_Type(item)
            if utils.get_TypeName(item_type) != utils.get_TypeName(self.content_type):
                if isinstance(self.content_type, type):
                    item = self.content_type(item)  # type: ignore
                else:
                    raise TypeError(
                        f"The input item is an instance of {utils.get_Type(item)} instead of {self.content_type}"
                    )
        return item

    def _check_not_empty(self):
        length = len(self)
        if length == 0:
            raise ValueError(f"This instance of {self.mytype.__name__} is empty.")
        return length

    def append(self, item: Any) -> Self:  # type: ignore
        item = self._check_type(item)
        super().append(item)
        return self

    def extend(self, other_list: Iterable[Any]) -> Self:  # type: ignore
        other_list = [self._check_type(item) for item in other_list]
        super().extend(other_list)
        return self

    def insert(self, index: int, item: Any) -> Self:  # type: ignore
        item = self._check_type(item)
        super().insert(index, item)
        return self

    def __getitem__(self, index: int | slice) -> T | Self:  # type: ignore
        if isinstance(index, slice):
            return self.mytype().extend(super().__getitem__(index))  # type: ignore
        return super().__getitem__(index)

    def __setitem__(self, index: int | slice, value: Iterable[Any] | Any):  # type: ignore
        if isinstance(value, Iterable):
            value = [self._check_type(item) for item in value]  # type: ignore
        else:
            value = self._check_type(value)

        super().__setitem__(index, value)  # type: ignore

    @classmethod
    def _type(cls) -> type[Self]:
        return cls

    @property
    def mytype(self) -> type[Self]:
        return self._type()

    def print(self) -> None:
        content_name = utils.get_TypeName(self.content_type)
        string = Text()
        string.append(self.mytype.__name__, style="bold bright_blue").append(
            "[", style="bold"
        ).append(content_name, style="bold bright_magenta").append("]", style="bold")
        print(string)
        print("[")
        num = len(self.data)
        for i in range(num):
            data_repr_i = self.data[i].__repr__().split("\n")
            data_repr_i = "\n".join("│ " + line for line in data_repr_i)
            if i != num - 1:
                data_repr_i += ","
            print(data_repr_i)
        print("]")

    def copy(self) -> Self:
        new_one = self._type()(self.data.copy())  # type: ignore
        return new_one

    def _inplace_or_not(self, inplace: bool) -> Self:
        return self if inplace else self.copy()

    def reverse(self, inplace: bool = False) -> Self:  # type: ignore
        len_self = self._check_not_empty()
        result = self._inplace_or_not(inplace)
        for i in range(len_self // 2):
            result[i], result[len_self - i - 1] = result[len_self - i - 1], result[i]
        return result

    def map(
        self,
        METHOD: Callable[..., Any],
        *method_args: Any,
        **method_kwargs: Any,
    ) -> Any:

        num = self._check_not_empty()

        pos_args = []
        for arg in method_args:
            if isinstance(arg, Iterable) and not isinstance(arg, (str)):
                len_arg = len(arg)  # type: ignore
                if len_arg != num:
                    raise ValueError(
                        f"Positional argument length mismatch: expected {num}, got {len_arg}"
                    )
                pos_args.append(arg)  # type: ignore
            else:
                pos_args.append([arg] * num)  # type: ignore

        # 广播关键字参数
        kw_args = {}
        for key, val in method_kwargs.items():
            if isinstance(val, Iterable) and not isinstance(val, (str)):  # type: ignore
                len_val = len(val)  # type: ignore
                if len_val != num:
                    raise ValueError(
                        f"Keyword argument '{key}' length mismatch: expected {num}, got {len_val}"
                    )
                kw_args[key] = val
            else:
                kw_args[key] = [val] * num

        if pos_args:
            pos_cols = list(zip(*pos_args))  # type: ignore
        else:
            pos_cols = [()] * num

        if kw_args:
            keys = list(kw_args.keys())  # type: ignore
            vals = [kw_args[k] for k in keys]  # type: ignore
            kw_cols = [dict(zip(keys, col)) for col in zip(*vals)]  # type: ignore
        else:
            kw_cols = [{}] * num  # type: ignore

        first_result = METHOD(self[0], *pos_cols[0], **kw_cols[0])

        if isinstance(first_result, tuple):
            # 多返回值
            results = [self.create_a_container(v) for v in first_result]  # type: ignore
            for i in range(1, num):
                res = METHOD(self[i], *pos_cols[i], **kw_cols[i])
                for j, val in enumerate(res):
                    results[j].append(val)

            return tuple(results)  # type: ignore

        else:
            # 单返回值
            data = self.create_a_container(first_result)
            for i in range(1, num):
                data.append(METHOD(self[i], *pos_cols[i], **kw_cols[i]))

            return data

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(
                f"Attribute {name} can not be accessed by __getattr__."
            )

        def _get_descriptor(cls: Any, attr_name: str):
            for base in cls.__mro__:
                if attr_name in base.__dict__:
                    return base.__dict__[attr_name]
            return None

        if hasattr(self.content_type, name):
            method0 = getattr(self.content_type, name)

            if callable(method0):
                descriptor = _get_descriptor(self.content_type, name)

                # 根据描述器类型构造包装函数
                if isinstance(descriptor, staticmethod):
                    # 静态方法：忽略第一个参数（self[i]）
                    def wrapper(*args: Any, **kwargs: Any):  # type: ignore
                        return descriptor.__func__(*args[1:], **kwargs)  # type: ignore
                elif isinstance(descriptor, classmethod):
                    # 类方法：第一个参数应为类本身
                    def wrapper(*args: Any, **kwargs: Any) -> Any:
                        return descriptor.__func__(  # type: ignore
                            self.content_type,  # type: ignore
                            *args[1:],
                            **kwargs,
                        )
                else:
                    # 普通实例方法
                    wrapper = method0

                def method(*args: Any, **kwargs: Any) -> Any:  # type: ignore
                    return self.map(wrapper, *args, **kwargs)

                return method

            else:
                # 非可调用属性（如类变量），需聚合各元素的值
                self._check_not_empty()

                def method() -> Any:  # type: ignore
                    results = []
                    for idx, item in enumerate(self):
                        result = getattr(item, name)
                        if not isinstance(result, tuple):
                            result = (result,)
                        if idx == 0:
                            results = [self.create_a_container(r) for r in result]  # type: ignore
                        else:
                            for i, r in enumerate(result):  # type: ignore
                                results[i].append(r)
                    return results[0] if len(results) == 1 else tuple(results)

                return method()

        # 2. 属性属于容器内第一个元素（实例属性）
        if len(self) > 0 and hasattr(self[0], name):

            def method():
                results = []
                for idx, item in enumerate(self):
                    result = getattr(item, name)
                    if not isinstance(result, tuple):
                        result = (result,)
                    if idx == 0:
                        results = [self.create_a_container(r) for r in result]  # type: ignore
                    else:
                        for i, r in enumerate(result):  # type: ignore
                            results[i].append(r)
                return results[0] if len(results) == 1 else tuple(results)

            return method()

        raise AttributeError(
            f"'{self.mytype().__name__}' object has no attribute '{name}'"
        )

    def _operator(
        self, operator: str, other: Any
    ) -> BasicContainer[Any] | NotImplementedType:

        len_self = self._check_not_empty()

        if isinstance(other, Iterable) and not isinstance(other, (str, np.ndarray)):
            len_other = len(other)  # type: ignore
        else:
            len_other = len_self
            other = [other] * len_self  # type: ignore

        if len_self != len_other:
            raise ValueError(
                f"The container and other must have the same length, but are {len_self} and {len_other}."
            )
        else:
            if operator == "+":
                return self.map(lambda x, y: x + y, other)  # type: ignore
            elif operator == "r+":
                return self.map(lambda x, y: y + x, other)  # type: ignore
            elif operator == "-":
                return self.map(lambda x, y: x - y, other)  # type: ignore
            elif operator == "r-":
                return self.map(lambda x, y: y - x, other)  # type: ignore
            elif operator == "*":
                return self.map(lambda x, y: x * y, other)  # type: ignore
            elif operator == "r*":
                return self.map(lambda x, y: y * x, other)  # type: ignore
            elif operator == "/":
                return self.map(lambda x, y: x / y, other)  # type: ignore
            elif operator == "r/":
                return self.map(lambda x, y: y / x, other)  # type: ignore
            elif operator == "@":
                return self.map(lambda x, y: x @ y, other)  # type: ignore
            elif operator == "r@":
                return self.map(lambda x, y: y @ x, other)  # type: ignore
            else:
                return NotImplemented

    def __add__(self, other: Any) -> BasicContainer[Any] | NotImplementedType:  # type: ignore
        return self._operator("+", other)

    def __radd__(self, other: Any) -> BasicContainer[Any] | NotImplementedType:  # type: ignore
        return self._operator("r+", other)

    def __sub__(self, other: Any) -> BasicContainer[Any] | NotImplementedType:  # type: ignore
        return self._operator("-", other)

    def __rsub__(self, other: Any) -> BasicContainer[Any] | NotImplementedType:  # type: ignore
        return self._operator("r-", other)

    def __mul__(self, other: Any) -> BasicContainer[Any] | NotImplementedType:  # type: ignore
        return self._operator("*", other)

    def __rmul__(self, other: Any) -> BasicContainer[Any] | NotImplementedType:  # type: ignore
        return self._operator("r*", other)

    def __truediv__(self, other: Any) -> BasicContainer[Any] | NotImplementedType:  # type: ignore
        return self._operator("/", other)

    def __rtruediv__(self, other: Any) -> BasicContainer[Any] | NotImplementedType:  # type: ignore
        return self._operator("r/", other)

    def __matmul__(self, other: Any) -> BasicContainer[Any] | NotImplementedType:  # type: ignore
        return self._operator("@", other)

    def __rmatmul__(self, other: Any) -> BasicContainer[Any] | NotImplementedType:  # type: ignore
        return self._operator("r@", other)

    def sum(self) -> T | NotImplementedType:
        len_self = self._check_not_empty()
        result = self[0]
        for i in range(1, len_self):
            result = result + self[i]  # type: ignore
        return result  # type: ignore

    def cumsum(self, inplace: bool = False):
        len_self = self._check_not_empty()
        result = self._inplace_or_not(inplace)
        if len_self == 1:
            return result
        else:
            for i in range(1, len_self):
                result[i] = result[i - 1] + self[i]  # type: ignore
            return result

    def prod(self) -> T:
        len_self = self._check_not_empty()
        result = self[0]
        for i in range(1, len_self):
            result = result * self[i]  # type: ignore
        return result  # type: ignore

    def cumprod(self, inplace: bool = False):
        len_self = self._check_not_empty()
        result = self._inplace_or_not(inplace)
        if len_self == 1:
            return result
        else:
            for i in range(1, len_self):
                result[i] = result[i - 1] * self[i]  # type: ignore
            return result

    def __array_ufunc__(self, ufunc: Any, method: Any, *inputs: Any, **kwargs: Any):

        if method != "__call__" or len(inputs) != 2:
            return NotImplemented

        a, b = inputs
        if a is self:
            other, reverse = b, False
        elif b is self:
            other, reverse = a, True
        else:
            return NotImplemented

        if not isinstance(other, np.ndarray):
            return NotImplemented

        # 完整的 ufunc -> (正向方法, 反向方法)
        op_map = {  # type: ignore
            np.add: ("__add__", "__radd__"),
            np.subtract: ("__sub__", "__rsub__"),
            np.multiply: ("__mul__", "__rmul__"),
            np.matmul: ("__matmul__", "__rmatmul__"),
        }

        pair = op_map.get(ufunc)  # type: ignore
        if pair is None:
            return NotImplemented

        forward_name, reverse_name = pair
        method_name = reverse_name if reverse else forward_name

        # numpy 数组 → list（0维转标量）
        if other.ndim == 0:
            converted = other.item()  # type: ignore
        else:
            converted = other  # type: ignore

        return getattr(self, method_name)(converted)


CONTAINER_TYPES: dict[type | TypeAliasType | None, type[BasicContainer[Any]]] = {}
