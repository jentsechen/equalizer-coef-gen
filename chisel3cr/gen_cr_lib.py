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


@define
class CRSchDescr(ToJson):
    name: str = field(default="", converter=Convertor(str), validator=Validator(str))
    type: str = field(default="int", converter=Convertor(str), validator=Validator(str))
    dim: sdim = field(default=sdim(), converter=Convertor(sdim), validator=Validator(sdim))
    qz: qt = field(default=qt(), converter=Convertor(qt), validator=Validator(qt))
    sign: eSign = field(default=eSign.Unsigned, converter=Convertor(eSign), validator=Validator(eSign))
    options: List[str] = field(default=[], converter=Convertor(List[str]), validator=Validator(List[str]))
    bw: int = field(default=0, converter=Convertor(int), validator=Validator(int))

    def get_cr_sch_instance(self):
        if self.type == "int":
            return f'{self.name}: CrSchInt = field(init=False,default=CrSchInt(name="{self.name}",bw={self.bw},dim={self.dim},sign=eSign.{self.sign.name})) '
        if self.type == "rfix":
            return f'{self.name}: CrSchRFix = field(init=False,default=CrSchRFix(name="{self.name}",qz={self.qz},dim={self.dim}))'
        if self.type == "cfix":
            return f'{self.name}: CrSchCFix = field(init=False,default=CrSchCFix(name="{self.name}",qz={self.qz},dim={self.dim}))'
        if self.type == "enum":
            return f'{self.name}: CrSchEnum = field(init=False,default=CrSchEnum(name="{self.name}",dim={self.dim},options=[{",".join(self.options)}]))'
        if self.type == 'bool':
            return f'{self.name}: CrSchBool = field(init=False,default=CrSchBool(name="{self.name}",dim={self.dim}))'
        return ''


@define
class CRDefSubCh(ToJson):
    name: str = field(default="", converter=Convertor(str), validator=Validator(str))
    addr: int = field(default=0, converter=Convertor(int), validator=Validator(int))
    val: List[CRSchDescr] = field(default=[], converter=Convertor(CRSchDescr), validator=Validator(CRSchDescr))

    def gen_args(self):

        def valtype2str(valtype):
            if valtype == 'int':
                return 'int'
            if valtype == 'rfix':
                return 'float'
            if valtype == 'cfix':
                return 'complex'
            if valtype == 'enum':
                return 'str'
            if valtype == 'bool':
                return 'bool'
            return ''

        args = []
        for x in self.val:
            if x.dim.dim == 0:
                args.append(f'{x.name}: {valtype2str(x.type)}')
            elif x.dim.dim == 1:
                args.append(f'{x.name}: List[{valtype2str(x.type)}]')
            elif x.dim.dim == 2:
                args.append(f'{x.name}: np.ndarray')
        return ', '.join(args)

    def get_func(self):
        out = []
        out.append(f'\tdef to_bits(self,{self.gen_args()}) -> BitArray:')
        out.append(f'\t\tout=BitArray()')
        for x in self.val:
            out.append(f'\t\tout += self.{x.name}.to_bits({x.name})')
        out.append(f'\t\treturn out')
        return '\n'.join(out)

    def get_cr_sch_class(self):

        def add_class(out: List[str]):
            out.append(f'@define')
            out.append(f'class cr_{self.name}(ToJson):')
            out.append(f'\tname: str = field(init=False,default="{self.name}")')
            out.append(f'\taddr: int = field(init=False,default={self.addr})')
            for x in self.val:
                out.append(f'\t{x.get_cr_sch_instance()}')

        def add_function(out: List[str]):
            out.append("")
            out.append(self.get_func())

        out = []
        add_class(out)
        add_function(out)
        return '\n'.join(out)

    def get_top_func(self):
        out = []
        out.append(f'\tdef func_{self.name}(self, {self.gen_args()}) -> None:')
        out.append(f'\t\treturn self.axi.write(addr=self.{self.name}.addr,data=self.{self.name}.to_bits({",".join([x.name for x in self.val])}))')
        return '\n'.join(out)


@define
class CRDefine(ToJson):
    name: str = field(default="", converter=Convertor(str), validator=Validator(str))
    setting: List[CRDefSubCh] = field(default=[], converter=Convertor(CRDefSubCh), validator=Validator(CRDefSubCh))

    def gen_import(self) -> str:
        return '''
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
    '''

    def gen_class(self, filename: Optional[str] = None):

        def add_import(out: List[str]):
            out.append(self.gen_import())

        def add_sub_cr_class(out: List[str]):
            for x in self.setting:
                out.append(x.get_cr_sch_class())

        def add_top_class(out: List[str]):
            out.append(f'@define')
            out.append(f'class {self.name}(ToJson):')
            out.append(f'\taxi: AxiCrWrite = field(default=DummyAxiCrWrite())')
            out.append(f'\tname: str = field(init=False, default="{self.name}")')
            for x in self.setting:
                out.append(f'\t{x.name}: cr_{x.name} = field(init=False, default=cr_{x.name}())')

        def add_functions(out: List[str]):
            out.append("")
            for x in self.setting:
                out.append("")
                out.append(x.get_top_func())

        def gen_file(out: List[str], filename: Optional[str] = None):
            filename = filename if filename else f'cr_{self.name}.py'
            with open(filename, 'w') as f:
                f.write('\n'.join(out))

        out = []
        add_import(out)
        add_sub_cr_class(out)
        add_top_class(out)
        add_functions(out)
        gen_file(out, filename)


if __name__ == '__main__':
    crd = CRDefine().from_json('/workspaces/Chisel3DigDev/out/TSensor/freq_select/FreqSelect.cr.json')
    crd.gen_class()

# if __name__ == '__main__':
#     x = FreqSelect(axi=DummyAxiCrWrite())
#     x.func_set_bfw([1 + 2j, 3 + 4j, 5 + 6j, 7 + 8j, 1 + 2j, 3 + 4j, 5 + 6j, 7 + 8j], 1, True)
#     x.func_set_par(1.2, 3.4, 5, 6, 7, True)