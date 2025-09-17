import sys
import enum
import typing
# import subprocess
# from subprocess import Popen, PIPE

import asyncio
from asyncio import subprocess
from subprocess import PIPE

from util.json_io import save_json, read_json
from collections import namedtuple
import typeguard
import pandas as pd
from matplotlib import pyplot as plt

projroot = __file__[0:(__file__.find('source'))]
sys.path.append(projroot + "source/pylib")

# from enum import Enum, auto
# from typeguard import Optional as Some, typechecked
# from typing import List
# from util.json_io import Sig
from util.signal_parser import Sig
from util.signal_analy import MyPlot

Enum = enum.Enum
auto = enum.auto
typechecked = typeguard.typechecked
Some = typeguard.Optional
List = typing.List

_sim_set_file_name = 'simset.json'


def taskrun(cmd: str, simlog: str = "sim.log") -> None:
    print('log file:  {}'.format(simlog))

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


class SimSetPar:
    data = {}

    def init(self, d):
        self.data = {k: v for (k, v) in d.items() if (v is not None) and (k != 'self')}

        # self.par = namedtuple('par', self.data.keys())(**self.data)
        self.par = namedtuple('par', d.keys())(**d)

    def __to_dict__(self):

        def extendvalue(v):
            if isinstance(v, complex):
                return '({},{})'.format(v.real, v.imag)
            if isinstance(v, SimSetPar):
                return v.__to_dict__()
            if isinstance(v, list):
                return [extendvalue(x) for x in v]
            if isinstance(v, dict):
                return {k: extendvalue(v) for (k, v) in v.items()}
            if isinstance(v, Enum):
                return v.name
            return v

        return {k: extendvalue(v) for (k, v) in self.data.items()}

    def __dict__(self):
        return self.__to_dict__()

    def save(self, fn=_sim_set_file_name):
        save_json(fn, self.__to_dict__())
        return fn


class SimRunCore:

    def __init__(self, target: Some[str], ctrl: Some[SimSetPar] = None, log_path: Some[str] = None):
        self.execfile = projroot + "build/release/bin/" + target
        self.ctrl = ctrl
        self.log_path = log_path or target + '.log'

    def run(self, ctrl: Some[SimSetPar] = None, simset_fn: str = _sim_set_file_name):

        def simsetting():
            if ctrl:
                ctrl.save(simset_fn)
            elif self.ctrl:
                self.ctrl.save(simset_fn)

        def cleanup():
            pass

        simsetting()
        taskrun(self.execfile + ' ' + simset_fn, self.log_path)
        cleanup()


class SimRun(SimSetPar):
    exec = 'test_unit_all'
    post_action = None
    pre_action = None
    plt = MyPlot()

    def save_image(self, fn_in):
        self.plt.save_image(fn_in)

    def set_plot(self, kwargs):
        self.plt = MyPlot(**kwargs)
        return self

    def post_process_hook(self, func):
        self.post_action = func
        return self

    def pre_process_hook(self, func):
        self.pre_action = func
        return self

    def run(self, simset_fn: str = _sim_set_file_name, log_path: str = None):
        self.pre_proc_run()
        SimRunCore(self.exec, self, log_path).run(simset_fn=simset_fn)
        return self.post_proc_run()

    def pre_proc_run(self):
        out = self.pre_process()
        if (self.pre_action):
            return self.pre_action()
        return out

    def post_process(self):
        return None

    def pre_process(self):
        return None

    def post_proc_run(self):
        out = self.post_process()
        if self.post_action:
            return self.post_action()
        return out


SimSet = SimRun


class SimResult:

    def __init__(self, data):
        if isinstance(data, str):
            self.data = read_json(data)
        if isinstance(data, list):
            self.data = data

    def filter(self, setting):
        if isinstance(setting, SimSetPar):
            setting = setting.__to_dict__()

        def hit(simset):
            return all([simset[k] == v for (k, v) in setting.items() if k in simset.keys()])

        d = [v for v in self.data if hit(v['simset'])]
        return SimResult(d)

    def get_result(self, *keys):
        if len(keys) == 0:
            return [v['result']['stat'] for v in self.data]
        if len(keys) == 1:
            return [v['result']['stat'][keys[0]] for v in self.data]
        else:
            return [{k: v['result']['stat'][k] for k in keys} for v in self.data]

    def get_setting(self, *keys):
        if len(keys) == 0:
            return [v['simset'] for v in self.data]
        if len(keys) == 1:
            return [v['simset'][keys[0]] for v in self.data]
        else:
            return [{k: v['simset'][k] for k in keys} for v in self.data]

    def plot(self, key_result='success_rate', key_setting='snr_dB', semilogy=True, func=lambda x: 1 - x):

        result = self.get_result(key_result)
        if func is not None:
            result = [func(x) for x in result]
        if semilogy:
            plt.semilogy(self.get_setting(key_setting), result)
        else:
            plt.plot(self.get_setting(key_setting), result)

    def show(self):
        print(pd.DataFrame(self.data))


class EnT(Enum):
    off, on, defv = auto(), auto(), auto()


@typechecked
class QtTable(SimSetPar):

    def __init__(self, **kwargs):
        self.init(kwargs)


@typechecked
class Init(SimSetPar):

    def __init__(self, name: Some[str] = None, fx: Some[EnT] = None, dump_en: Some[bool] = None, sub_dump: Some[EnT] = None, qtable: Some[QtTable] = None):
        self.init(locals())


if __name__ == "__main__":

    class AA(Enum):
        A, B, C = auto(), auto(), auto()

    @typechecked
    class A(SimSet):

        def __init__(self, a: Some[int] = None, b: Some[int] = None, c: Some[int] = None):
            self.init(locals())

    @typechecked
    class B(SimSet):

        def __init__(self, a: Some[List[A]] = None, b: Some[int] = None, c: Some[float] = 2, d: Some[AA] = AA.A):
            self.init(locals())

    a = A(a=5, b=None, c=2)
    a.save("a.json")
    b = B(a=[a], b=3, c=None, d=AA.B)
    b.save("b.json")
