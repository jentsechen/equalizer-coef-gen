import matplotlib.pyplot as plt
import json
import numpy as np
from os.path import isfile
import inspect
import time
from collections import namedtuple
import unittest
from glob import glob
import os
from os.path import join
from pathlib import Path
import shutil

KiB = 1024
MiB = 1024 * KiB
GiB = 1024 * MiB

show = lambda **kwargs: print(["{}={:.3g}".format(k, v) for k, v in kwargs.items()])


def ld2dl(ld):
    return {k: [dic[k] for dic in ld] for k in ld[0]}


def dl2ld(dl):
    return [dict(zip(dl, t)) for t in zip(*dl.values())]


def dl_mean(dl):
    return {k: np.mean(v) for (k, v) in dl.items()}


def ld_mean(ld):
    return dl_mean(ld2dl(ld))


def dict2str(d, p="#"):
    kvl = [k + "=" + str(v) for (k, v) in d.items()]
    kvl.sort(key=str.lower)
    return p + (p + "_" + p).join(kvl) + p  # + "#_#".join(kvl) + "#"


def submit_subprocess(python_script, run_sel, run_mode='nonblocking', func_preproc=None, **args):
    import subprocess
    import time
    if not python_script.endswith('.py'):
        python_script = python_script + '.py'

    cmd = 'python {} {}'.format(python_script, run_sel)
    time.sleep(2)
    for k, v in args.items():
        if k not in ['sim_select']:
            if isinstance(v, list):
                v = ','.join([str(a) for a in v])
            elif isinstance(v, str):
                v = '"' + v + '"'
            elif isinstance(v, bool):
                v = 'True' if v else 'False'
            else:
                pass

            cmd += ' -{}={}'.format(k, v)

    print(cmd)

    if func_preproc:
        func_preproc()

    if run_mode == 'blocking':
        subprocess.run(cmd, shell=True)
    elif run_mode == 'nonblocking':
        subprocess.Popen(cmd, shell=True)
    else:
        pass


def add_path(folder):
    Path(folder).mkdir(parents=True, exist_ok=True)


def mkdir(p):
    if not os.path.exists(p):
        os.makedirs(p)


def move_files(filetype, dst, prefix="", copy=False):

    def oper(a, b):
        if copy:
            shutil.copyfile(a, b)
        else:
            os.rename(a, b)

    def stripPrefix(x, p):
        return x[len(p):] if x.startswith(p) else x

    mkdir(dst)
    prefix = os.path.normpath(prefix)
    for a in glob(filetype):
        a = os.path.normpath(a)
        if os.path.isfile(a):
            c = os.path.normpath(stripPrefix(a, prefix)).split(os.sep)
            b = os.sep.join(c[(1 if len(c) > 1 else 0):])
            dist_fn = os.path.normpath(join(dst, b))
            mkdir(os.path.dirname(dist_fn))
            if os.path.isfile(dist_fn):
                os.remove(dist_fn)
            oper(a, dist_fn)


def copy_files(filetype, dst, prefix=""):
    move_files(filetype, dst, prefix, True)


def dict2stuct_str(x, prefix=""):

    def strs(x):
        return '\'%s\'' % x if isinstance(x, str) else "{}".format(x)

    def formatx(k, v):
        return prefix + '.%s = %s;\n' % (k, strs(v))

    out = []
    for k, v in x.items():
        if isinstance(v, dict):
            out += dict2stuct_str(v, prefix + '.' + k)
        else:
            out += formatx(k, v)
    return ''.join(out)


def cal_cdf(data, complementary=True, fn=None, num_bins=250):
    if isinstance(data, list):
        data = np.asarray(data)
    counts, bin_edges = np.histogram(data, bins=num_bins)
    counts = counts.astype(float) / data.size
    cdf = np.cumsum(counts)
    if complementary is True:
        y = 1 - cdf
        ylabel = 'CCDF'
    else:
        y = cdf
        ylabel = 'CDF'
    x = bin_edges[1:]
    if isinstance(fn, str):
        plt.plot(x, y)
        plt.grid(True)
        plt.ylabel(ylabel)
        plt.savefig(fn + '.png')
        plt.close()

    return x, y


def cal_hist(data, fn=None, num_bins=100):
    if isinstance(data, list):
        data = np.asarray(data)
    counts, bin_edges = np.histogram(data, bins=num_bins)
    counts = counts.astype(float) / data.size
    y = counts
    x = bin_edges[1:]
    if isinstance(fn, str):
        plt.plot(x, y)
        plt.grid(True)
        plt.ylabel('hist')
        plt.savefig(fn + '.png')
        plt.close()

    return x, y


class alpha_filt():

    def __init__(self, alpha, N=10):
        self.buf = []
        self.alpha = alpha
        self.N = N

    def avg(self, input):
        if len(self.buf) < self.N:
            self.buf.append(input)
            self._avg = sum(self.buf) / len(self.buf)
        else:
            self._avg = self._avg * (1 - self.alpha) + self.alpha * input

        return self._avg

    def reset(self):
        self.buf = []
        self._avg = 0


def debug_show(msg):
    print(msg)
    input('press any key')


def dict2class(dictionary, name='dict_class', exclude_list=[]):
    temp = dictionary.copy()
    for key, value in temp.items():
        if isinstance(value, dict) and key not in exclude_list:
            temp[key] = dict2class(value, key, exclude_list)
    return namedtuple(name, temp.keys())(**temp)


class Profiler:

    def __init__(self, class_name, en=False, discard_first_k=0):
        #         if not class_name:
        #             self.class_name = str(inspect.stack()[1][0].f_locals["self"].__class__.__name__)
        self.class_name = class_name
        self.en = en
        self.ctime_str = time.strftime('%y-%m-%d_%H_%M_%S')
        self.time = {}
        self.discard_first_k = discard_first_k

    def report(self):

        def avg(a):
            return float(np.mean(np.array(a)))

        def std(a):
            return float(np.std(np.array(a)))

        def inv_avg(a):
            return 1.0 / avg(a)

        r = {'class': self.class_name, 'init_time': self.ctime_str, 'def_out_fn': self._default_filename(), 'time': {}}
        for func_name, dat in zip(self.time.keys(), self.time.values()):
            r['time'][func_name] = {}
            for label, time_info in zip(dat.keys(), dat.values()):
                td = time_info['time_diff'][self.discard_first_k:]
                r['time'][func_name][label] = {
                    'call': len(td) if td else None,
                    'avg': avg(td) if td else None,
                    'std': std(td) if td else None,
                    'max': max(td) if td else None,
                    'min': min(td) if td else None,
                    'fps': inv_avg(td) if td else None,
                    'total': sum(td) if td else None
                }
        return r

    def __del__(self):
        if self.en:
            self.save(self._default_filename())

    def _default_filename(self):
        return '{}_{}_prof_rpt.log'.format(self.class_name, self.ctime_str)

    def save(self, fn=None):
        if self.en:
            if fn is None:
                fn = self._default_filename()
            with open(fn, 'w') as f:
                f.write(json.dumps(self.report(), indent=4, ensure_ascii=False))

    def _symmetric_check(self, func_name, label, filename, lineno):
        a = self.time[func_name][label]
        if not (len(a['start']) == len(a['time_diff'])):
            print('profile symetric bug @{}:{} : start-end mismatch'.format(filename, lineno))

    def _get_caller_info(self, stack_in):
        caller = inspect.getframeinfo(stack_in[1][0])  # filename,lineno,code_context,function,index
        return caller.function, caller.filename, caller.lineno

    def start(self, label=None):
        if self.en:
            if label is None:
                label = 'body'
            func_name, filename, lineno = self._get_caller_info(inspect.stack())
            try:
                self._symmetric_check(func_name, label, filename, lineno)
            except:
                self._init_dict(func_name, label)
                self._symmetric_check(func_name, label, filename, lineno)

            self.time[func_name][label]['start'].append(time.time())

    def end(self, label=None):
        if self.en:
            ctime = time.time()
            if label is None:
                label = 'body'
            func_name, filename, lineno = self._get_caller_info(inspect.stack())

            try:
                self.time[func_name][label]['time_diff'].append(ctime - self.time[func_name][label]['start'][-1])
            except:
                self._init_dict(func_name, label)
                self.time[func_name][label]['time_diff'].append(ctime - self.time[func_name][label]['start'][-1])

            self._symmetric_check(func_name, label, filename, lineno)

    def _init_dict(self, caller_func_name, label):
        try:
            self.time[caller_func_name][label] = {'start': [], 'time_diff': []}
        except:
            self.time[caller_func_name] = {}
            self.time[caller_func_name][label] = {'start': [], 'time_diff': []}

    def get_class(self):
        return self.class_name

    def get_time(self):
        return self.time


class PAR:

    def __init__(self, par=None):
        self._par = {}
        self.non_class_convert_list = []
        self.key_list = []
        if not (par == None):
            self._par = par if isinstance(par, dict) else read_json(par)

    def set(self, key, value, force=False, type=None, not_convert_to_class=False):
        self.key_list.append(key)
        if not_convert_to_class == True:
            self.non_class_convert_list.append(key)

        if force:
            self._par[key] = value
        elif key.replace('.', '_') in self._par.keys():
            self._par[key] = self._par.pop(key.replace('.', '_'))
            if self._par[key] is None:
                self._par[key] = value
            else:
                self._par[key] = self._par[key]
        elif key in self._par.keys():
            if self._par[key] is None:
                self._par[key] = value
            else:
                self._par[key] = self._par[key]
        else:
            self._par[key] = value

        if type is not None:
            if not type == dict:
                assert not isinstance(self._par[key], dict), '{}={} is dict, should not set to type={}'.format(key, self._par[key], type)
            if isinstance(type, list):
                assert self._par[key] in type, '{}={} not in {}'.format(key, self._par[key], type)
            if type in [str, int, float]:
                if isinstance(self._par[key], list):
                    self._par[key] = [type(a) for a in self._par[key]]
                else:
                    self._par[key] = type(self._par[key])

        return self._par[key]

    def get(self, key=None):
        if not (key == None):
            if key in self.key_list:
                return self._par[key]
            else:
                return None
        else:
            return {k: v for k, v in self._par.items() if k in self.key_list}

    def get_dict(self, key=None):
        return dict_wrap(self.get(key))

    def get_class(self, key=None):
        return dict2class(self.get_dict(key), exclude_list=self.non_class_convert_list)

    def save(self, fn):
        save_json(fn, self.get(), indent=4, sort_keys=True)


def dict_wrap(d_in):
    d_out = {}
    for k, v in d_in.items():
        d = d_out
        k_list = k.split('.')
        for i, ksub in enumerate(k_list):
            if ksub in d.keys():
                d = d[ksub]
                assert isinstance(d, dict), 'dict_extend error: key="{}" naming comflict'.format(ksub)
            else:
                if i == len(k_list) - 1:
                    d.update({ksub: v})  # assign value
                else:
                    d.update({ksub: {}})  # placeholder
                d = d[ksub]
    return d_out


def save_json(fn, x, indent=4, sort_keys=False):
    if not fn.endswith('.json'):
        fn = fn + '.json'
    with open(fn, 'w') as file:
        file.write(json.dumps(x, indent=indent, sort_keys=sort_keys, ensure_ascii=True))


def read_json(fn):
    if isfile(fn):
        with open(fn) as data_file:
            return json.loads(data_file.read())
    else:
        return None


def func_name():
    return inspect.stack()[1][3]


class LOG:

    def __init__(self, log_data_en, log_time_en=False):
        self.class_name = inspect.stack()[1][0].f_locals["self"].__class__
        self.log_en = log_data_en
        self.log_time_en = log_time_en
        self.ctime_str = time.strftime('%y-%m-%d_%H_%M_%S_%f')
        self.save_sn = 0
        self.reset_buffer()

    def __del__(self):
        self.save('log_{}_{}_end'.format(self.class_name, self.ctime_str))

    def reset_buffer(self):
        self.data = {'class': self.class_name, 'ctime': self.ctime_str}
        self.time = {}

    def save(self, fn=None):
        if self.log_en:
            if fn is None:
                fn = 'log_{}_{}_sn{}'.format(self.class_name, self.ctime_str, self.save_sn)
            save_json(fn, self.data, indent=4)
        if self.log_time_en:
            if fn is None:
                fn = 'log_{}_{}_sn{}'.format(self.class_name, self.ctime_str, self.save_sn)
            fn = 'time_' + fn
            save_json(fn, self.data, indent=4)
        self.save_sn += 1

    def add(self, key=None, val=None):
        if self.log_time_en:
            caller = inspect.stack()[1][0].f_code.co_name
            self.time[caller][('self' if (key is None) else key)].append(time.time())
        if self.log_en:
            caller = inspect.stack()[1][0].f_code.co_name
            self._init_data_if_not(caller, key)
            self.data[caller][key].append(val)

    def _init_data_if_not(self, caller, key):
        if not (caller in self.data.keys()):
            self.data[caller] = {}
            self.data[caller][key] = []
        elif not key in self.data[caller].keys():
            self.data[caller][key] = []

    def _init_time_if_not(self, caller, key):
        if not (caller in self.data.keys()):
            self.data[caller] = {}
            self.data[caller]['self'] = []
        if not (key is None):
            if not (key in self.data[caller].keys()):
                self.data[caller][key] = []

    def get_data(self):
        return self.data

    def get_class(self):
        return self.class_name

    def get_time(self):
        return self.time


class TEST(unittest.TestCase):

    def test_cal_cdf(self):
        data = np.random.normal(size=(1, 1000))
        cdf, xscale = cal_cdf(data, fn='test_cdf', num_bins=200)

    def test_dict2class(self):
        d = {'a': 1, 'b': 2, 'c': 3, 'dd': {'a1': 5, 'a2': [2, 3, 4], 'a3': ['x', 'y', 'z']}}
        dc = dict2class(d)

        print(d)
        print(dc)

        self.assertEqual(dc.a, d['a'])
        self.assertEqual(dc.b, d['b'])
        self.assertEqual(dc.c, d['c'])
        self.assertEqual(dc.dd.a1, d['dd']['a1'])
        self.assertEqual(dc.dd.a2[1], d['dd']['a2'][1])
        self.assertEqual(dc.dd.a3[-1], d['dd']['a3'][-1])

    def test_dict_extend(self):
        d_in = {'aa.a': 'aa', 'aa.1': 'bb', 'bb': 7, 'cc.y.z': 9, 'cc.y.w': 3, 'cc.x.w': 4, 'cc.u': 5}
        print(d_in)
        print(dict_wrap(d_in))

        d_in1 = {'a_x': 5, 'bX': 3, 'c': 2}
        print(d_in1)
        print(dict_wrap(d_in1))

    def test_alpha_avg(self):

        a = alpha_filt(0.05, 10)
        for x in range(10):
            print(a.avg(x))

        a.reset()
        for x in range(10):
            print(a.avg(x))


from tabulate import tabulate
import shutil


def table_print(dataSet: list) -> None:

    def get_max_col_width(headers) -> list:
        numCols = len(headers)
        shellColumns = shutil.get_terminal_size((80, 20)).columns
        maxColSingle = shellColumns / numCols
        return [maxColSingle] * numCols

    headers = dataSet[0].keys()
    rows = [x.values() for x in dataSet]
    maxColWidths = get_max_col_width(headers)
    tabulated = tabulate(rows, headers=headers, tablefmt="simple_grid", maxcolwidths=maxColWidths)
    print(tabulated)


if __name__ == '__main__':
    #     TEST_PROFILER().test_run()
    unittest.main()
