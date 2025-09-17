# from pydantic import BaseModel, ValidationError, validator, Field
import json
from typing import List, Dict, Tuple, Union, Optional, Any, Callable, TypeVar, Generic, Type
from enum import Enum, auto
import numpy as np

import pydantic
# from to_json import ToJson as ToJson0

validator = pydantic.validator
# BaseModel = pydantic.BaseModel
Field = pydantic.Field
ValidationError = pydantic.ValidationError

from ToJsonBase import ToJsonBase


class ToJson(ToJsonBase, pydantic.BaseModel):

    class Config:
        arbitrary_types_allowed = True
        validate_call = True

    def __post_init__(self):
        pass

    def model_post_init(self, __context: Any):
        self.__post_init__()

    def to_dict(self) -> dict:
        return self.formatting(self.model_dump())

    def convert_to_dict_for_old_to_json(self) -> dict:
        return self.to_dict()


    def from_dict(self, data: dict):
        self.__init__(**self.convert_x(data))
        return self

    @classmethod
    def to_ndarray(cls, v) -> np.ndarray:
        return np.array(v)

    @classmethod
    def convert_enum(cls, v, enum_type: Type[Enum]) -> Enum:
        if isinstance(v, str):
            return enum_type[v]
        if isinstance(v, int):
            return enum_type(v)
        if isinstance(v, list):
            return [cls.convert_enum(x, enum_type) for x in v]
        return v

    @classmethod
    def to_enum(cls, v, enum_type: Type[Enum]) -> Enum:
        return cls.convert_enum(v, enum_type)


BaseModel = ToJson


class myEnum(Enum):
    X, Y = auto(), auto()


class Mydata1(BaseModel):
    a: int = None


class MyData(BaseModel):
    a: int = None
    p: myEnum = None
    y: complex = None
    z: np.ndarray = Field(default_factory=lambda: np.zeros(5))
    w: np.ndarray = None
    aaa: Mydata1 = None

    @validator('z', 'w', pre=True)
    @classmethod
    def _convert_ndarray(cls, v) -> np.ndarray:
        return cls.to_ndarray(v)

    @validator('p', pre=True)
    @classmethod
    def _convert_p(cls, v) -> myEnum:
        return cls.convert_enum(v, myEnum)

    def do_somthing(self):
        print('do somthing: a={}'.format(self.a))

    def __post_init__(self):
        self.w = self.z


if __name__ == '__main__':

    x = MyData(a=5, p=myEnum.X, y=2 + 3j, z=np.array([1 + 1j, 2, 3, 4, 5]), w=np.array([1 + 1j, 2, 3, 4, 5]), aaa=Mydata1(a=3))
    ddd = x.to_json()
    print(ddd)
    y = MyData().from_json(ddd)
    print(y)
    y.save_json('test.json')
    z = MyData().load_json('test.json')
    z.do_somthing()
    print(z)