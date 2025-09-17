def append_search_path(l: list = []):
    from sys import path
    projpath = (__file__[0:(__file__.find('ws'))] if 'ws' in __file__ else __file__[0:(__file__.find('source'))]) + 'source'
    path += [projpath] + [projpath + '/' + x for x in l]


append_search_path(['pylib', 'pylib/util'])

from typing import List, Protocol, Tuple, Union
from util.ToJsonIO import ToJson, Validator, Convertor
from attrs import define, field

from enum import Enum, auto
import numpy as np

from bitstring import BitArray, BitStream
from common import sdim, qt, eSign, eMSB, eLSB

# class CRWrite(Protocol):

#     def write


class ToBits(Protocol):
    bw: int
    sign: eSign
    dim: sdim
    name: str

    def to_int(self, data: Union[int, float, complex]) -> int:
        raise NotImplementedError

    def to_int_cpx(self, data: Union[complex, np.complex128]) -> Tuple[int, int]:
        raise NotImplementedError

    def __data_validation__(self, data):

        def validate_type(types):
            if not isinstance(data, types):
                raise TypeError(f"{self.name} type error: data(={type(data)}) must be in [{types}]")

        if self.dim.dim == 0:
            validate_type((int, float, complex, np.complex128))
            return
        if self.dim.dim == 1:
            validate_type((np.ndarray, list))
            if not (len(data) == self.dim.len):
                raise ValueError(f"{self.name} value error: data length({len(data)}) not equal to dim.len({self.dim.len})")
            return
        elif self.dim.dim == 2:
            validate_type((np.ndarray, list))
            if isinstance(data, np.ndarray):
                if not ((len(data.shape) == 2) and (data.shape[0] == self.dim.row) and (data.shape[1] == self.dim.col)):
                    raise ValueError(f"{self.name} value error: data shape({data.shape}) must be equal to dim.row({self.dim.row}) and dim.col({self.dim.row})")
            else:
                if not ((len(data) == self.dim.row)):
                    raise ValueError(f"{self.name} value error: data row(={len(data)}) must be equal to dim.row(={self.dim.row})")
                for (i, x) in enumerate(data):
                    if not (len(x) == self.dim.col):
                        raise ValueError(f"{self.name} value error: data[{i}].col(={len(x)}) must be equal to dim.row(={self.dim.row})")
            return
        raise ValueError("dim.dim must be 0 or 1 or 2")

    def __convert__(self, data: Union[int, float, complex, np.complex128]) -> BitArray:
        if isinstance(data, (complex, np.complex128)):
            [r, i] = self.to_int_cpx(data)
            return BitArray(int=r, length=self.bw) + BitArray(int=i, length=self.bw)
        if isinstance(data, (int, float)):
            if (self.sign == eSign.Unsigned):
                return BitArray(uint=self.to_int(data), length=self.bw)
            return BitArray(int=self.to_int(data), length=self.bw)
        raise TypeError(f"{self.name} type error: data must be int or float or complex")

    def to_bits(self, data: Union[np.ndarray, list, int, float, complex]) -> BitArray:
        convert = self.__convert__
        self.__data_validation__(data)
        if self.dim.dim == 0:
            return convert(data)
        if self.dim.dim == 1:
            out = BitArray()
            for x in data:
                out += convert(x)
            return out
        if self.dim.dim == 2:
            out = BitArray()
            for x in data:
                for y in x:
                    out += convert(y)
            return out


@define
class CrSchInt(ToJson, ToBits):
    name: str = field(default="", converter=Convertor(str), validator=Validator(str))
    bw: int = field(default=0, converter=Convertor(int), validator=Validator(int))
    dim: sdim = field(default=sdim(), converter=Convertor(sdim), validator=Validator(sdim))
    sign: eSign = field(default=eSign.Unsigned, converter=Convertor(eSign), validator=Validator(eSign))

    def to_int(self, data: int) -> int:
        if self.sign == eSign.Unsigned and data < 0:
            raise ValueError("{self.name} value error: data must be positive")
        return data


@define
class CrSchFixBase(ToJson, ToBits):
    name: str = field(default="", converter=Convertor(str), validator=Validator(str))
    qz: qt = field(default=qt(), converter=Convertor(qt), validator=Validator(qt))
    dim: sdim = field(default=sdim(), converter=Convertor(sdim), validator=Validator(sdim))

    def __post_init__(self):
        self.bw = self.qz.bw
        self.sign = self.qz.sign


@define
class CrSchRFix(CrSchFixBase):

    def to_int(self, data: Union[float, int]) -> int:
        if not isinstance(data, (int, float)):
            raise TypeError(f"{self.name} type error: data must be int or float")
        return self.qz.to_int(data)


@define
class CrSchCFix(CrSchFixBase):

    def to_int_cpx(self, data: Union[complex, np.complex128]) -> Tuple[int, int]:
        if not isinstance(data, (complex, np.complex128)):
            raise TypeError(f"{self.name} type error: data must be complex")
        return self.qz.to_int_cpx(data)


@define
class CrSchBool(ToJson, ToBits):
    name: str = field(default="", converter=Convertor(str), validator=Validator(str))
    dim: sdim = field(default=sdim(), converter=Convertor(sdim), validator=Validator(sdim))
    bw = 1
    sign = eSign.Unsigned

    def to_int(self, data: bool) -> int:
        return int(data)


@define
class CrSchEnum(ToJson, ToBits):
    name: str = field(default="", converter=Convertor(str), validator=Validator(str))
    bw: int = field(default=0, converter=Convertor(int), validator=Validator(int))
    dim: sdim = field(default=sdim(), converter=Convertor(sdim), validator=Validator(sdim))
    sign = eSign.Unsigned
    options: List[str] = field(default=[""], converter=Convertor(str), validator=Validator(str))

    def __post_init__(self):
        self.bw = int(np.ceil(np.log2(len(self.options))))

    def to_int(self, data: str) -> int:
        if not isinstance(data, str):
            raise TypeError(f"{self.name} type error: data must be str")
        if not (data in self.options):
            raise ValueError(f"{self.name} value error: data(={data}) must be in {self.options}")
        return self.options.index(data)


if __name__ == '__main__':
    x = CrSchInt(bw=4, dim=sdim(len=4), sign=eSign.Signed)
    y = CrSchRFix(name="aaa", qz=qt(sign=eSign.Signed, int_bit=2, frac_bit=2, msb=eMSB.Sat, lsb=eLSB.Rnd), dim=sdim())
    z = CrSchRFix(name="aaa", qz=qt(sign=eSign.Signed, int_bit=2, frac_bit=2, msb=eMSB.Sat, lsb=eLSB.Rnd), dim=sdim(len=5))
    w = CrSchRFix(name="aaa", qz=qt(sign=eSign.Signed, int_bit=2, frac_bit=2, msb=eMSB.Sat, lsb=eLSB.Rnd), dim=sdim(row=2, col=2))
    print(f'{x.to_bits([1, 2, 3, 4]).bin}')
    print(f'{y.to_bits(2.3).bin}')
    print(f'{z.to_bits([1.2,2.3,3.4,4.5,5.6]).bin}')
    print(f'{w.to_bits(np.array([[1.2,2.3],[1.2,2.3]])).bin}')
    print(f'{w.to_bits([[1.2,2.3],[2.3,-3.2]]).bin}')