import matplotlib.pyplot as plt
from .utility_lib import PAR
import numpy as np


def plot_fig(y):
    p = PLOT()
    p.new_fig()
    p.add(y)
    p.show()


def plot_figure(x, y, **args):
    try:
        plt.plot(x, y, args['marker'])
    except:
        plt.plot(x, y)

    if 'xlabel' in args.keys() and args['xlabel']:
        plt.xlabel(args['xlabel'])
    if 'ylabel' in args.keys() and args['ylabel']:
        plt.ylabel(args['ylabel'])
    if 'title' in args.keys() and args['title']:
        plt.title(args['title'])
    if 'text' in args.keys() and args['text']:
        plt.text(**args['text'])
    if 'axis' in args.keys() and args['axis']:
        plt.axis(args['axis'])
    if 'grid' in args.keys() and args['grid']:
        plt.grid(args['grid'])
    if 'yscale' in args.keys() and args['yscale']:
        plt.yscale(args['yscale'])
    if 'fn' in args.keys() and args['fn']:
        if 'dpi' in args.keys() and args['dpi']:
            plt.savefig(args['fn'], dpi=args['dpi'])
        else:
            plt.savefig(args['fn'])
    if 'show' in args.keys() and args['show']:
        plt.show()
    if 'close' in args.keys() and args['close']:
        plt.close()


class PLOT():

    def __init__(self, **args):
        self.def_fig_unit = {'lines': [], 'par': self.set_par(args)}
        self.reset()
        self.plot_figure = lambda x, y, **args: plot_figure(x, y, **args)

    def set_par(self, args):
        par = PAR(args)
        par.set('xlabel', None)
        par.set('ylabel', None)
        par.set('title', None)
        par.set('axis', None)
        par.set('xlim', None)
        par.set('ylim', None)
        par.set('xticks', None)
        par.set('grid', False)
        par.set('yscale', 'linear', type=['linear', 'log', 'logit', 'symlog', 'semilogy'])
        par.set('legend', False)
        par.set('density', False)
        return par.get()

    def new_fig(self, fid=None, fig_type='plot', title=None):
        assert (fig_type in ['plot', 'hist', 'scatter'])
        if fid is None:
            if self.curr_id is None:
                self.curr_id = 0
            else:
                self.curr_id = max([int(a) for a in self.fig.keys()]) + 1
        else:
            self.curr_id = fid
        self.fig[self.curr_id] = {k: v.copy() for k, v in self.def_fig_unit.items()}  # to prevent some dict reference issue
        self.fig[self.curr_id]['type'] = fig_type
        if title:
            self.fig[self.curr_id]['par']['title'] = title
        return self.curr_id

    def get_id(self, fid=None):
        if fid is None:
            if self.curr_id is None:
                self.curr_id = self.new_fig(0)
        else:
            if fid not in self.fig.keys():
                self.curr_id = self.new_fig(fid)
            else:
                self.curr_id = fid
        return self.curr_id

    def add(self, *args, **kwargs):

        if isinstance(args[0], int):
            fid, args = args[0], args[1:len(args)]
        else:
            fid = None
        fid = self.get_id(fid)
        lines = self.fig[fid]['lines']
        par = PAR(kwargs)
        par.set('label', 'data' + str(len(lines)))
        par.set('linestyle', 'solid')
        par.set('linewidth', 2)
        par.set('markersize', 4)
        lines.append({'args': args, 'par': par.get()})
        return fid

    def setp(self, fid=None, **args):
        fid = self.get_id(fid)
        par = self.fig[fid]['par']
        par.update(args)
        self.fig[fid]['par'] = self.set_par(par)
        return fid

    def _plot(self, fid=None):
        fid = self.get_id(fid)
        lines = self.fig[fid]['lines']
        par = self.fig[fid]['par']

        plt.figure(fid)
        for data in lines:
            if self.fig[fid]['type'] == 'hist':
                plt.hist(*data['args'], label=data['par']['label'], density=par['density'])
            elif self.fig[fid]['type'] == 'scatter':
                plt.scatter(np.real(*data['args']), np.imag(*data['args']))
            elif par['yscale'] == 'semilogy':
                plt.semilogy(*data['args'], **data['par'])
            else:
                plt.plot(*data['args'], **data['par'])

        if par['xlabel'] is not None:
            plt.xlabel(par['xlabel'])
        if par['ylabel'] is not None:
            plt.ylabel(par['ylabel'])
        if par['title'] is not None:
            plt.title(par['title'])
        if par['axis'] is not None:
            plt.axis(par['axis'])
        if par['xticks'] is not None:
            plt.xticks(np.arange(len(par['xticks'])), par['xticks'])
        if par['xlim'] is not None:
            plt.xlim(par['xlim'][0], par['xlim'][1])
        if par['ylim'] is not None:
            plt.ylim(par['ylim'][0], par['ylim'][1])
        if par['grid']:
            plt.grid()
        if par['legend']:
            plt.legend()

        if not par['yscale'] == 'semilogy':
            plt.yscale(par['yscale'])
        return fid

    def show(self, block=False, all=True, fid=None):
        if all == True:
            for fid in self.fig.keys():
                self._plot(fid)
            plt.show(block=block)
        else:
            self._plot(fid)
            plt.show()
            plt.close()
            fid = self.get_id(fid)
        if not block:
            input("press any key to continue...")

    def reset(self):
        self.fig = {}
        self.curr_id = None

    def save(self, *args, **kwargs):
        if len(args) == 1:
            fid, fn = None, args[0]
        else:
            fid, fn = args[0], args[1]
        self._plot(fid)
        plt.savefig(fn, **kwargs)
        plt.close()


if __name__ == '__main__':
    import numpy as np
    from sp import spectrum_dB
    t = np.arange(0.0, 1.0, 0.1)
    t1 = np.arange(0.0, 1.0, 0.2)
    s1 = np.sin(20 * 2 * np.pi * t)
    s2 = np.sin(30 * 2 * np.pi * t1)
    s3 = np.sin(40 * 2 * np.pi * t)
    s4 = np.sin(80 * 2 * np.pi * t)

    p = PLOT()
    fig1 = p.new_fig()
    fig2 = p.new_fig()
    p.add(fig1, t, s1, label='x1')
    p.add(fig1, t1, s2, label='x2')
    p.add(fig1, t, s3, label='x3')
    p.setp(xlabel='time', ylabel='xxx', title='sin', grid=True, legend=True)
    # fig2=p.new_fig()
    # p.add(s4,'-c',label='x4')
    # p.add(s1,'-r',label='x1')
    # p.setp(xlabel='time',ylabel='zzz',title='sss',grid=True,legend=False)

    p.save(fig1, 'test1.png')
    # p.save(fig2,'test2.png')

    # f,s=spectrum_dB(s1,s2,s3,s4,fs=320,nfft=512,show=False,fn='test3.png',label=['s1','s2','s3','s4'])
