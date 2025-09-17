def append_search_path(l: list = []):
    from sys import path
    projpath = (__file__[0:(__file__.find('ws'))] if 'ws' in __file__ else
                __file__[0:(__file__.find('source'))]) + 'source'
    path += [projpath] + [projpath + '/' + x for x in l]


append_search_path(['pylib'])

import time


def get_time_stamp():
    return time.strftime("y%y_m%m_d%d_%H_%M_%S", time.localtime())


def runInParallel(*fns):
    from multiprocessing import Process
    proc = []
    for fn in fns:
        p = Process(target=fn)
        p.start()
        proc.append(p)
        time.sleep(1)
    for p in proc:
        p.join()

def runInParallel_chl(*fns):
    from multiprocessing import Process
    proc = []
    for fn in fns:
        p = Process(target=fn)
        p.start()
        proc.append(p)
        time.sleep(0.05)
    for p in proc:
        p.join()