import sys

projroot = __file__[0:(__file__.find('source'))]
sys.path.append(projroot + "source/pylib")

import asyncio
from asyncio import subprocess
from subprocess import PIPE
from typing import List, Dict, Tuple, Union, Optional, Any, Callable, TypeVar, Generic, Type
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from pylib.util.to_json import ToJson
from enum import Enum, auto
import numpy as np


def taskrun(cmd: str, simlog: str = "sim.log", testing_run=False) -> None:
    print('log file:  {}'.format(simlog))

    if testing_run:
        print('cmd={}, simlog={}'.format(cmd, simlog))
        return None

    async def runexec(cmd):
        file = open(simlog, 'w')
        proc = await subprocess.create_subprocess_exec(*cmd.split(' '), stdin=PIPE, stdout=PIPE)
        while True:
            line = await proc.stdout.readline()
            if proc.returncode is not None:
                break
            if line == b'':
                break
            line = line.decode('utf-8')
            print(line.strip('\n'))
            file.write(line)
            file.flush()

            if proc.stderr:
                print(await proc.stderr.readline().decode('utf-8').strip('\n'))
        file.close()

    asyncio.run(runexec(cmd))


class SimRun(ToJson):
    exec = 'test_unit_all'
    # sim_setting: ToJson = ToJson()
    simset_fn = 'sim_setting.json'
    log_path = 'sim.log'
    post_action = None
    pre_action = None

    execfile = ''

    def __post_init__(self):
        self.execfile = projroot + "build/release/bin/" + self.exec
        print(self.execfile)

    def post_process_hook(self, func):
        self.post_action = func
        return self

    def pre_process_hook(self, func):
        self.pre_action = func
        return self

    def run(self, log_path: str = None, simset_fn: str = None, testing_run: bool = False):
        if self.pre_action is not None:
            self.pre_action()
        simset_fn = simset_fn or self.simset_fn
        self.save_json(simset_fn)
        taskrun(self.execfile + ' ' + simset_fn, log_path or self.log_path, testing_run)
        out = None
        if self.post_action is not None:
            out = self.post_action()

        return out


class EnT(Enum):
    off, on, defv = auto(), auto(), auto()


class eSign(Enum):
    Signed, Unsigned = auto(), auto()


class eMSB(Enum):
    Sat, SymS, Wrap = auto(), auto(), auto()


class eLSB(Enum):
    Trun, Rnd = auto(), auto()


@dataclass_json
@dataclass
class Qz(ToJson):
    sign: eSign = eSign.Signed
    int_bit: int = None
    frac_bit: int = None
    msb: eMSB = eMSB.Sat
    lsb: eLSB = eLSB.Rnd

    def __post_init__(self):
        self.tot_bit = self.frac_bit + self.int_bit + (1 if self.sign == eSign.Signed else 0)
        self.ub = 2**(self.int_bit) - 2**(-self.frac_bit)
        self.lb = -(2**(self.int_bit)) if self.sign == eSign.Signed else 0

    def apply(self, x):
        if (self.lsb == eLSB.Rnd):
            x = np.round(x * 2**self.frac_bit + 0.5) / 2**self.frac_bit
        else:
            x = np.floor(x * 2**self.frac_bit) / 2**self.frac_bit

        if (self.msb == eMSB.Sat):
            x = x % 2**self.int_bit
        else:
            x = max(min(x, self.ub), self.lb)

        return x

    def to_int(self, x):
        return self.apply(x) * 2**self.frac_bit


@dataclass_json
@dataclass
class QtTable(ToJson):
    table: Dict[str, Qz] = None

    def to_dict_formatting(self):
        return self.formating(self.to_dict()['table'])


@dataclass_json
@dataclass
class Init(ToJson):
    name: str = None
    fx: EnT = None
    dump_en: bool = None
    sub_dump: EnT = None
    qtable: QtTable = None
