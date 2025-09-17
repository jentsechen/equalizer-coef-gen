import json
from typing import List, Dict, Tuple, Union, Optional, Any, Callable, TypeVar, Generic, Type

from enum import Enum, auto
import numpy as np

from attrs import asdict, define, field, validators

from sys import path
path += ["/home/jtc/equalizer_analysis/trx_interposer_output_analysis/util"]

from ToJsonBase import ToJsonBase
from util.json_io import read_json


def to_enum(v: Union[str, int, list], enum_type: Type[Enum]) -> Union[Enum, list]:
    if isinstance(v, str):
        return enum_type[v]
    if isinstance(v, int):
        return enum_type(v)
    if isinstance(v, list):
        return [to_enum(x, enum_type) for x in v]
    return v


@define
class ToJson(ToJsonBase):

    def __attrs_post_init__(self):
        self.__post_init__()

    def to_dict(self) -> dict:
        return {k: self.formatting(v) for (k, v) in asdict(self).items() if v is not None}

    def from_dict(self, data: dict):
        self.__init__(**self.convert_x(data))  # type: ignore
        return self

    def from_json(self, data: str):
        return self.from_dict(read_json(data))


class Convertor:

    @staticmethod
    def allow_none_wrap(func: Callable, allow_none: bool) -> Callable:

        def wrapper(x):
            if x is None:
                return None
            return func(x)

        return func if not allow_none else wrapper

    def __init__(self, func: Union[Callable, Type[ToJsonBase], Type[Enum], Type[np.ndarray], Type[complex]], allow_none: bool = False) -> None:

        if isinstance(func, type) and issubclass(func, ToJsonBase):
            self.func = Convertor.asToJson(func, allow_none)
        elif isinstance(func, type) and issubclass(func, Enum):
            self.func = Convertor.asEnum(func, allow_none)
        elif isinstance(func, type) and issubclass(func, np.ndarray):
            self.func = Convertor.asNDarray(allow_none)
        elif isinstance(func, type) and issubclass(func, complex):
            self.func = Convertor.asCpx(allow_none)
        else:

            def funcwrap(variable: Any):
                if isinstance(variable, list):
                    return [funcwrap(y) for y in variable]
                return func(variable)

            self.func = self.allow_none_wrap(funcwrap, allow_none)

    def __call__(self, variable):
        return self.func(variable)

    @staticmethod
    def asToJson(mytype: Type[ToJsonBase], allow_none: bool = True) -> Callable:

        def func(variable: Any) -> Any:
            if isinstance(variable, mytype):
                return variable
            if isinstance(variable, dict):
                return mytype().from_dict(variable)
            if isinstance(variable, list):
                return [func(x) for x in variable]
            raise ValueError("ToJsonConvertor: variable type {} is not a dict or type of {}".format(type(variable).__name__, mytype.__name__))

        return Convertor.allow_none_wrap(func, allow_none)

    @staticmethod
    def asEnum(enum_type: Type[Enum], allow_none: bool = False) -> Callable:

        def func(value: Union[str, int, list]) -> Union[Enum, list]:
            if isinstance(value, enum_type):
                return value
            if isinstance(value, str):
                return enum_type[value]
            if isinstance(value, int):
                return enum_type(value)
            if isinstance(value, list):
                return [func(x) for x in value]
            raise ValueError("ToJsonConvertor: variable type {} is not a [str or int or {}]".format(type(value).__name__, enum_type.__name__))

        return Convertor.allow_none_wrap(func, allow_none)

    @staticmethod
    def asNDarray(allow_none: bool = False) -> Callable:
        return Convertor.allow_none_wrap(np.array, allow_none)

    @staticmethod
    def asCpx(allow_none: bool = False) -> Callable:

        def func(value: complex or str or list or dict):
            if isinstance(value, str):
                if value.count("j") == 1:
                    return complex(value.replace(" ", ""))
                if value.startswith("(") and value.endswith(")"):
                    s1 = value.replace('(', '').replace(')', '').split(',')
                    return complex(float(s1[0]), float(s1[1]))
            if isinstance(value, list):
                return [func(x) for x in value]
            if isinstance(value, np.ndarray):
                return np.array([func(x) for x in value])
            if isinstance(value, complex):
                return value
            RuntimeError("Invalid data type")

        return Convertor.allow_none_wrap(func, allow_none)


class Validator:

    @staticmethod
    def allow_none_wrap(func: Callable, allow_none: bool) -> Callable:

        def wrapper(instance, attribute, value):
            if value is not None:
                func(instance, attribute, value)

        return func if not allow_none else wrapper

    def __init__(self, func: Union[Callable, List[Callable], Type[Any]], allow_none: bool = False):

        if isinstance(func, type):
            self.func = Validator.check_type(func, allow_none)
        else:

            def func_wrap(instance, attribute, variable: Union[List[Any], Any]) -> None:

                def func_list_run(instance, attribute, variable: Union[List[Any], Any]) -> None:
                    if isinstance(func, list):
                        for subfunc in func:
                            subfunc(instance, attribute, variable)
                        return
                    func(instance, attribute, variable)

                if isinstance(variable, list):
                    for y in variable:
                        func_wrap(instance, attribute, y)
                else:
                    func_list_run(instance, attribute, variable)

            self.func = Validator.allow_none_wrap(func_wrap, allow_none)

    def __call__(self, instance, attribute, variable) -> None:
        return self.func(instance, attribute, variable)

    @staticmethod
    def check_type(mytype, allow_none: bool = True) -> Callable:

        def func(instance, attribute, variable: Union[List[Any], Any]) -> None:
            if isinstance(variable, list):
                for y in variable:
                    func(instance, attribute, y)
                return
            if not isinstance(variable, mytype):
                raise ValueError("value must be instance of {}".format(mytype.__name__))

        return Validator.allow_none_wrap(func, allow_none)


if __name__ == '__main__':

    class myEnum(Enum):
        X, Y = 0, auto()

    def number_is_larger_than_3(instance, attribute, value):
        if value < 3:
            raise ValueError("Value must be larger than 3")

    @define
    class Mydata1(ToJson):
        a: Optional[int] = field(default=None, converter=Convertor(int, True), validator=Validator(int, True))
        b: Optional[np.int8] = field(default=None, converter=Convertor(np.int8, True), validator=Validator([Validator(np.int8), number_is_larger_than_3], True))
        p: Optional[myEnum] = field(default=None, converter=Convertor(myEnum, True), validator=Validator(myEnum, True))
        y: Optional[complex] = field(default=None, converter=Convertor(complex, True), validator=Validator(complex, True))

    @define
    class MyData(ToJson):
        b: Mydata1 = field(default=None, converter=Convertor(Mydata1, True), validator=Validator(Mydata1, True))
        w: Optional[np.ndarray] = field(default=None, converter=Convertor(np.ndarray, True), validator=Validator(np.ndarray, True))
        xx: Optional[List[Mydata1]] = field(default=None, converter=Convertor(Mydata1, True), validator=Validator(Mydata1, True))

    @define
    class MyData2(ToJson):
        a: Optional[List[MyData]] = field(default=None, converter=Convertor(MyData, True), validator=Validator(MyData, True))

    a = Mydata1(a=5, p="X", b=np.int8(4), y=2 + 3j)
    print(a)
    print(a.to_json())
    a.save_json("test.json")
    b = Mydata1().from_json(a.to_json())
    print(b)
    c = MyData(w=[1 + 1j, 2, 3, 4, 5], xx=[b] * 4)  # type: ignore
    print(c)
    c.save_json("test2.json")
    d = MyData.load_json("test2.json")

    e = MyData2(a=[c] * 3)
    print(e)
