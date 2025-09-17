from cmath import e
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from typing import Dict, List, Optional, TypeVar, Generic, Union
from matplotlib.pyplot import xlabel
import plotly.graph_objects as go
from enum import Enum
import numpy as np
import plotly.graph_objects as go


class ToJson:

    def from_json(self, data: str):
        return self

    def from_dict(self, data: dict):
        return self

    def to_json(self):
        return ""

    def to_dict(self):
        return {}

    def save_json(self, filename: str):
        with open(filename, 'w') as f:
            f.write(self.to_json())

    def load_json(self, filename: str):
        with open(filename, 'r') as f:
            return self.from_json(f.read())


@dataclass_json
@dataclass
class SDim(ToJson):
    col: int = 0
    dim: int = 0
    len: int = 0
    row: int = 0


class eSign(Enum):
    Signed, Unsigned = 'Signed', 'Unsigned'


class eMsb(Enum):
    Sat, Wrap, SymS = 'Sat', 'Wrap', 'SymS'


class eLsb(Enum):
    Rnd, Trun = 'Rnd', 'Trun'


class eSType(Enum):
    double, int, cpx, bool = 'double', 'int', 'cpx', 'bool'


@dataclass_json
@dataclass
class Qformat(ToJson):
    frac_bit: int = 0
    int_bit: int = 0
    lsb: eLsb = eLsb.Trun
    msb: eMsb = eMsb.Sat
    sign: eSign = eSign.Signed


T = TypeVar('T')


@dataclass_json
@dataclass
class CFixWave(ToJson):
    re: List[float] = field(default_factory=list)
    im: List[float] = field(default_factory=list)

    def __post_init__(self):
        self.data = np.array([r + 1j * i for (r, i) in zip(self.re, self.im)])
        self.re = np.array(self.re)
        self.im = np.array(self.im)


@dataclass_json
@dataclass
class Signal(ToJson):
    timestamp: List[int] = field(default_factory=list)
    type: eSType = eSType.double
    clock_fs: float = 0.0
    enums: Optional[T] = None
    name: str = ""
    buf: T = None
    dim: SDim = SDim()
    fx: Qformat = Qformat()

    def __post_init__(self):
        self.timex = np.array([x / self.clock_fs for x in self.timestamp])

        def cpxbuf2cpxarr(buf: dict):
            out = []
            for a in zip(buf['re'], buf['im']):
                try:
                    out.append(a[0] + 1j * a[1])
                except:
                    print('len(out)={},len(buf)={}'.format(len(out), len(buf['re'])))
                    break

            return np.array(out)

        if self.dim.dim == 0:
            if (self.type == eSType.cpx):
                self.wave = cpxbuf2cpxarr(self.buf)
            else:
                self.wave = np.array(self.buf)
        if self.dim.dim == 1:
            if (self.type == eSType.cpx):
                self.wave = np.transpose(np.array([cpxbuf2cpxarr(buf) for buf in self.buf]), (1, 0))
            else:
                self.wave = np.array(self.buf)

    def get_tracer(self, func=None, name_surfix: str = ""):
        if func is None:
            if (self.type == eSType.cpx):
                return self.get_tracer(np.abs, "_abs")
            return go.Scatter(x=self.timex, y=self.wave, name=self.name)
        else:
            return go.Scatter(x=self.timex, y=func(self.wave), name=self.name + name_surfix)

    def show(self):
        fig = go.Figure()
        fig.add_trace(self.get_tracer())
        if (self.type == eSType.cpx):
            fig.add_trace(self.get_tracer(np.real, '_real'))
            fig.add_trace(self.get_tracer(np.imag, '_imag'))
        fig.update_xaxes(title_text='Time (us)')
        fig.show()

    def save_image(self, path, fn=None):
        fig = go.Figure()
        fig.add_trace(self.get_tracer())
        if (self.type == eSType.cpx):
            fig.add_trace(self.get_tracer(np.real, '_real'))
            fig.add_trace(self.get_tracer(np.imag, '_imag'))
        fig.update_xaxes(title_text='Time (us)')
        fig.write_html('{}/{}.html'.format(path, fn or self.name.replace('<', '').replace('>', '')))

    def slice(self, timex_l, timex_r=None):
        out = Signal().from_dict(self.to_dict())
        timex_r = timex_r or self.timex.max()
        out.wave = self.wave[(self.timex > timex_l) & (self.timex < timex_r)]
        out.timex = self.timex[(self.timex > timex_l) & (self.timex < timex_r)]

        return out

    def __getitem__(self, i: int):

        # out = Signal().from_dict(self.to_dict())
        # out.wave = self.wave[i]
        return self.wave[i]


def Sig(fn: str):
    return Signal().load_json(fn)


if __name__ == '__main__':

    x = Sig('/workspaces/Chisel3DigDev/bbsysdev/source/pylib/test/sat_com/baseband/rx/int/waveform/rx_test.ce[0].golay_mf.gunit<0>.out.a.json')
    x.show()