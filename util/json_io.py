from cmath import e
import json
import os
import numpy as np

from enum import Enum
from pathlib import Path

# import signal_parser

# Sig = signal_parser.Sig


def hash_dict(d):
    out = ('{}'.format(d))
    out = out.replace(':', '=').replace(',', '_')
    # out = out.translate({ord(i): '_' for i in ':{},'})
    out = out.translate({ord(i): None for i in ' \''})
    out = out[1:-1]  # remove first _ and last _
    return out


def read_json(fn):
    if os.path.isfile(fn):
        with open(fn) as data_file:
            return json.loads(data_file.read())
    else:
        return None


def create_path_if_not_exist(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(fn, x, indent=4, sort_keys=False):
    if not fn.endswith('.json'):
        fn = fn + '.json'

    if fn:
        if '/' in fn:  # if fn is a path
            create_path_if_not_exist(fn[0:fn.rfind('/') + 1])

    def replace_cpx(x):
        if isinstance(x, list):
            return [replace_cpx(y) for y in x]
        if isinstance(x, complex):
            return '({},{})'.format(x.real, x.imag)
        return x

    def make_seriesable(x):
        if isinstance(x, np.ndarray):
            if len(x.shape) == 1:
                return list(x)
            else:
                return [make_seriesable(y) for y in x]
        if isinstance(x, list):
            return replace_cpx(x)
        if isinstance(x, dict):
            return {k: make_seriesable(v) for k, v in x.items()}
        if isinstance(x, Enum):
            return x.name
        return x

    with open(fn, 'w') as file:
        file.write(json.dumps(make_seriesable(x), indent=indent, sort_keys=sort_keys, default=lambda o: '<not serializable>', ensure_ascii=True))


def read_vec(fn):

    def cpxstr2cpx(s):
        s1 = s.replace('(', '').replace(')', '').split(',')
        return complex(float(s1[0]), float(s1[1]))

    if not fn.endswith('json'):
        fn = fn + '.json'
    data = read_json(fn)
    try:
        return np.array([cpxstr2cpx(s) for s in data])
    except:
        return np.array(data)


def read_vecs(fn):

    def list2vecs(data):

        def cpxstr2cpx(s):
            s1 = s.replace('(', '').replace(')', '').split(',')
            return complex(float(s1[0]), float(s1[1]))

        if len(data) and isinstance(data[0], list):
            return np.array([list2vecs(x) for x in data])
        try:
            return np.array([cpxstr2cpx(x) for x in data])
        except:
            return np.array(data)

    data = read_json(fn)
    if isinstance(data, list):
        return list2vecs(data)
    return {k: list2vecs(v) for k, v in data.items()}


# class Sig:

#     def __init__(self, fn):
#         data = read_json(fn)
#         if not isinstance(data, dict):
#             self.sig_type = False
#         elif "name" not in data.keys():
#             self.sig_type = False
#         else:
#             self.sig_type = True

#         if self.sig_type:
#             self.timestamp = data['timestamp']
#             self.timex = [x * data['clock_fs'] for x in self.timestamp]

#             self.dim, self.name, self.fx = data['dim'], data['name'], data['fx']
#             if self.dim['dim'] == 0:
#                 self.read_1d(data)
#             if self.dim['dim'] != 0:
#                 print("to support sig read for dim>0")
#         else:
#             self.timestamp, self.timex, self.dim, self.name, self.fx = None, None, None, None, None
#             self.wave = read_vecs(fn)

#     def read_1d(self, data):
#         if isinstance(data['buf'], dict):
#             self.wave = np.array([complex(re, im) for (re, im) in zip(data['buf']['re'], data['buf']['im'])])

#         else:
#             self.wave = np.array(data['buf'])

# class SigWaveform:

#     def __init__(self, fn):
#         self.data = read_json(fn)

if __name__ == '__main__':

    # x = Signal().load_json('/workspaces/Chisel3DigDev/bbsysdev/source/pylib/test/sat_com/baseband/rx/int/waveform/rx_test.ce[0].golay_mf.gunit<0>.out.a.json')
    # y = Signal().load_json('/workspaces/Chisel3DigDev/bbsysdev/source/pylib/test/sat_com/baseband/rx/int/waveform/rx_test.dfe.dmixer40.inc.phase_delta.json')
    # z = Signal().load_json('/workspaces/Chisel3DigDev/bbsysdev/source/pylib/test/sat_com/baseband/rx/int/waveform/rx_test.dfe.dmixer40.ind.valid.json')
    # w = Signal().load_json('/workspaces/Chisel3DigDev/bbsysdev/source/pylib/test/sat_com/baseband/rx/int/waveform/rx_test.ce[0].out.ce.json')
    # w = Sig('/workspaces/Chisel3DigDev/bbsysdev/source/pylib/test/sat_com/baseband/rx/int/waveform/rx_test.ce[0].out.ce.json')
    # print(x.wave)
    # print(y.wave)
    # print(z.wave)
    print(w.wave[4])