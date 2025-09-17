def append_search_path(l: list = []):
    from sys import path
    projpath = (__file__[0:(__file__.find('ws'))] if 'ws' in __file__ else __file__[0:(__file__.find('source'))]) + 'source'
    path += [projpath] + [projpath + '/' + x for x in l]


append_search_path(['pylib', 'pylib/util'])

from typing import List, Protocol, Tuple, Union
from util.ToJsonIO import ToJson, Validator, Convertor
from attrs import define, field
import math

from enum import Enum, auto
import numpy as np


class eSign(Enum):
    Unsigned, Signed, Signed1s = auto(), auto(), auto()


class eMSB(Enum):
    Sat, Wrap, SymS = auto(), auto(), auto()


class eLSB(Enum):
    Rnd, Trun = auto(), auto()


@define
class sdim(ToJson):
    len: int = field(default=0, converter=Convertor(int), validator=Validator(int))
    row: int = field(default=0, converter=Convertor(int), validator=Validator(int))
    col: int = field(default=0, converter=Convertor(int), validator=Validator(int))
    dim: int = field(default=0, converter=Convertor(int), validator=Validator(int))

    def __post_init__(self):
        if self.len == 0 and self.row == 0 and self.col == 0:
            self.dim = 0
        elif self.col == 0:
            self.dim = 1
        else:
            self.dim = 2

    def __str__(self):
        if self.dim == 0:
            return f"sdim()"
        if self.dim == 1:
            return f"sdim(len={self.len})"
        if self.dim == 2:
            return f"sdim(row={self.row},col={self.col})"


@define
class qt(ToJson):
    en: bool = field(default=True, converter=Convertor(bool), validator=Validator(bool))
    sign: eSign = field(default=eSign.Unsigned, converter=Convertor(eSign), validator=Validator(eSign))
    int_bit: int = field(default=0, converter=Convertor(int), validator=Validator(int))
    frac_bit: int = field(default=0, converter=Convertor(int), validator=Validator(int))
    msb: eMSB = field(default=eMSB.Sat, converter=Convertor(eMSB), validator=Validator(eMSB))
    lsb: eLSB = field(default=eLSB.Rnd, converter=Convertor(eLSB), validator=Validator(eLSB))

    def __str__(self):
        return f"qt(sign=eSign.{self.sign.name},int_bit={self.int_bit},frac_bit={self.frac_bit},msb=eMSB.{self.msb.name},lsb=eLSB.{self.lsb.name})"

    def __post_init__(self):
        self.bw = self.frac_bit + self.int_bit + (1 if self.sign == eSign.Signed else 0)
        self.ub = 2**(self.int_bit) - 2**(-self.frac_bit)
        if self.sign == eSign.Signed:
            if self.msb == eMSB.SymS:
                self.lb = -self.ub
            else:
                self.lb = -(2**(self.int_bit))
        else:
            self.lb = 0
        self.wrap_bound = (2**(self.int_bit + 1)) if self.sign == eSign.Signed else (2**self.int_bit)

    def apply(self, data: float):
        def lsb_proc(data: float):
            if (self.lsb == eLSB.Rnd):
                data = np.floor(data * (2**self.frac_bit) + 0.5) / (2**self.frac_bit)
            else:
                data = np.floor(data * (2**self.frac_bit)) / (2**self.frac_bit)
            return data
        def msb_proc(data: float):
            if (self.msb == eMSB.Wrap):
                data = math.fmod(data, self.wrap_bound)
                if data > self.ub:
                    data -= self.wrap_bound
                if data < self.lb:
                    data += self.wrap_bound
            else:
                data = max(min(data, self.ub), self.lb)
            return data
        return msb_proc(lsb_proc(data))

    def to_int(self, data: float):
        return self.apply(data) * (2**self.frac_bit)

    def to_int_cpx(self, data: complex) -> Tuple[int, int]:
        return (self.apply(data.real) * (2**self.frac_bit), self.apply(data.imag) * (2**self.frac_bit))
