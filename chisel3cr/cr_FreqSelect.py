def append_search_path(l: list = []):
    from sys import path
    projpath = (__file__[0:(__file__.find('ws'))] if 'ws' in __file__ else __file__[0:(__file__.find('source'))]) + 'source'
    path += [projpath] + [projpath + '/' + x for x in l]


append_search_path(['pylib', 'pylib/util'])

from typing import Dict, List, Optional, Protocol, Tuple, Union
from util.ToJsonIO import ToJson, Validator, Convertor
from attrs import define, field

from enum import Enum, auto
import numpy as np

from common import sdim, qt, eSign, eMSB, eLSB
from bitstring import BitArray, BitStream
from schema import CrSchBool, CrSchCFix, CrSchInt, CrSchRFix, ToBits
from AxiCrWrite import AxiCrWrite, DummyAxiCrWrite


@define
class cr_set_bfw(ToJson):
    name: str = field(init=False, default="set_bfw")
    addr: int = field(init=False, default=0)
    bfw: CrSchCFix = field(init=False, default=CrSchCFix(name="bfw", qz=qt(sign=eSign.Signed, int_bit=3, frac_bit=8, msb=eMSB.Sat, lsb=eLSB.Rnd), dim=sdim(len=8)))
    bfw_idx: CrSchInt = field(init=False, default=CrSchInt(name="bfw_idx", bw=4, dim=sdim(), sign=eSign.Unsigned))
    set_bfw_en: CrSchBool = field(init=False, default=CrSchBool(name="set_bfw_en", dim=sdim()))

    def to_bits(self, bfw: List[complex], bfw_idx: int, set_bfw_en: bool) -> BitArray:
        out = BitArray()
        out += self.bfw.to_bits(bfw)
        out += self.bfw_idx.to_bits(bfw_idx)
        out += self.set_bfw_en.to_bits(set_bfw_en)
        return out


@define
class cr_set_par(ToJson):
    name: str = field(init=False, default="set_par")
    addr: int = field(init=False, default=1)
    freq_select_thr: CrSchRFix = field(init=False, default=CrSchRFix(name="freq_select_thr", qz=qt(sign=eSign.Unsigned, int_bit=6, frac_bit=2, msb=eMSB.Sat, lsb=eLSB.Rnd), dim=sdim()))
    noise_est_thr: CrSchRFix = field(init=False, default=CrSchRFix(name="noise_est_thr", qz=qt(sign=eSign.Unsigned, int_bit=3, frac_bit=5, msb=eMSB.Sat, lsb=eLSB.Rnd), dim=sdim()))
    n_symb: CrSchInt = field(init=False, default=CrSchInt(name="n_symb", bw=4, dim=sdim(), sign=eSign.Unsigned))
    wind_th: CrSchInt = field(init=False, default=CrSchInt(name="wind_th", bw=5, dim=sdim(), sign=eSign.Unsigned))
    wind_width: CrSchInt = field(init=False, default=CrSchInt(name="wind_width", bw=5, dim=sdim(), sign=eSign.Unsigned))
    set_par_en: CrSchBool = field(init=False, default=CrSchBool(name="set_par_en", dim=sdim()))

    def to_bits(self, freq_select_thr: float, noise_est_thr: float, n_symb: int, wind_th: int, wind_width: int, set_par_en: bool) -> BitArray:
        out = BitArray()
        out += self.freq_select_thr.to_bits(freq_select_thr)
        out += self.noise_est_thr.to_bits(noise_est_thr)
        out += self.n_symb.to_bits(n_symb)
        out += self.wind_th.to_bits(wind_th)
        out += self.wind_width.to_bits(wind_width)
        out += self.set_par_en.to_bits(set_par_en)
        return out


@define
class FreqSelect(ToJson):
    axi: AxiCrWrite = field(default=DummyAxiCrWrite())
    name: str = field(init=False, default="FreqSelect")
    set_bfw: cr_set_bfw = field(init=False, default=cr_set_bfw())
    set_par: cr_set_par = field(init=False, default=cr_set_par())

    def func_set_bfw(self, bfw: List[complex], bfw_idx: int, set_bfw_en: bool) -> None:
        return self.axi.write(addr=self.set_bfw.addr, data=self.set_bfw.to_bits(bfw, bfw_idx, set_bfw_en))

    def func_set_par(self, freq_select_thr: float, noise_est_thr: float, n_symb: int, wind_th: int, wind_width: int, set_par_en: bool) -> None:
        return self.axi.write(addr=self.set_par.addr, data=self.set_par.to_bits(freq_select_thr, noise_est_thr, n_symb, wind_th, wind_width, set_par_en))


if __name__ == '__main__':

    class AXICRW(AxiCrWrite):

        def __write__(self, addr: int, data: BitArray) -> None:
            print(f'my write: addr={addr}, data={data.hex}')

    x = FreqSelect(axi=AXICRW())
    print(x.to_dict())
    x.func_set_bfw([1 + 2j, 3 + 4j, 5 + 6j, 7 + 8j, 1 + 2j, 3 + 4j, 5 + 6j, 7 + 8j], 1, True)
    x.func_set_par(1.2, 3.4, 5, 6, 7, True)