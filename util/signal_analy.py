def append_search_path(l: list = []):
    from sys import path
    projpath = (__file__[0:(__file__.find('ws'))] if 'ws' in __file__ else __file__[0:(__file__.find('source'))]) + 'source'
    path += [projpath] + [projpath + '/' + x for x in l]


append_search_path(['pylib'])
from pathlib import Path
from scipy.signal import welch
import matplotlib.pyplot as plt
import numpy as np
from numpy import fft
from util.json_io import read_vec
from util.signal_parser import Sig

image_dpi = 600


def papr_analysis(waveform, label: str = "", range=[4.5, 8]):
    waveform_norm = np.abs(waveform)**2
    waveform_norm = [x for x in waveform_norm if x > 0]
    average_power = np.average(waveform_norm)
    waveform_norm_dB = 10 * np.log10(waveform_norm / average_power)
    hist, bin_edges = np.histogram(waveform_norm_dB, bins=100, range=range)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    pdf = hist / np.sum(hist)
    cdf = np.cumsum(pdf)
    ccdf = 1 - cdf
    plt.figure(1)
    plt.plot(bin_centers, pdf, label='pdf')
    plt.xlim(range)
    plt.ylim([1e-3, 1])
    plt.grid('both')
    plt.savefig("{}_papr_pdf.png".format(label))
    plt.close()
    plt.figure(2)
    plt.plot(bin_centers, cdf, label='cdf')
    plt.hlines(0.99, xmax=range[1], xmin=range[0], linewidth=2, color='r')
    plt.grid('both')
    plt.xlim(range)
    plt.ylim([1e-3, 1])
    plt.savefig("{}_papr_cdf.png".format(label))
    plt.close()
    plt.figure(3)
    plt.semilogy(bin_centers, ccdf, label='ccdf')
    plt.hlines(0.01, xmax=range[1], xmin=range[0], linewidth=2, color='r')
    plt.xlim(range)
    plt.ylim([1e-3, 1])
    plt.grid('both')
    plt.savefig("{}_papr_ccdf.png".format(label))
    plt.close()


def spectrum(waveform, fs=1.0, nfft=256, show=False, fn=None, label=[], title='spectrum', xlabel='Freq', linewidth=3, legend=True):

    if isinstance(waveform, Sig):
        wave = waveform.wave
    else:
        wave = waveform
    freq, pxx = welch(wave, fs=fs, window='hanning', nperseg=nfft, noverlap=None, nfft=nfft, scaling='density')
    if show or (fn is not None):
        plt.plot(fft.fftshift(freq), fft.fftshift(10 * np.log10(pxx)), label=label, linewidth=linewidth)
        plt.xlabel(xlabel)
        plt.title(title)
        # if legend:
        #     plt.legend()
        if fn is not None:
            plt.savefig(fn)
        if show:
            plt.show(block=False)
            input("Press [enter] to continue.")
    return freq, pxx


def real(waveform):
    return [x.real for x in waveform]


def imag(waveform):
    return [x.imag for x in waveform]


# def plot_cons(waveform, show=False, fn=None, title=None, dpi=image_dpi):

#     if fn:
#         if '/' in fn:  # if fn is a path
#             create_path_if_not_exist(fn[0:fn.rfind('/') + 1])

#     plt.scatter(real(waveform), imag(waveform), s=1)
#     if title: plt.title(title)
#     if fn is not None:
#         plt.gcf().set_size_inches(8, 6)
#         plt.savefig(fn, dpi=dpi)
#     if show:
#         plt.show(block=False)
#         input("Press [enter] to continue.")
#     plt.close()


def plot_vecc(waveform, show=False, fn=None, marker=None, linewidth=1, timex=None, title=None, scatter=False, time_window=None, dpi=image_dpi, **kwargs):
    if fn:
        if '/' in fn:  # if fn is a path
            create_path_if_not_exist(fn[0:fn.rfind('/') + 1])
    time_window = time_window or range(len(waveform))
    waveform = [waveform[x] for x in time_window if x < len(waveform)]
    if not scatter:
        subplt_re = plt.subplot(311)
        timex = timex or range(len(waveform))
        timex = [timex[x] for x in time_window if x < len(timex)]
        subplt_re.plot(timex, real(waveform), label='real', marker=marker, linewidth=linewidth)
        subplt_im = plt.subplot(312)
        subplt_im.plot(timex, imag(waveform), label='imag', marker=marker, linewidth=linewidth)
        subplt_abs = plt.subplot(313)
        subplt_abs.plot(timex, np.abs(waveform), label='abs', marker=marker, linewidth=linewidth)
    else:
        plt.scatter(real(waveform), imag(waveform))
    if title:
        plt.title(title)
    if fn is not None:
        plt.gcf().set_size_inches(8, 6)
        plt.savefig(fn, dpi=dpi)
    if show:
        plt.show(block=False)
        input("Press [enter] to continue.")
    plt.close()


def plot_vecf(waveform, show=False, fn=None, marker=None, linewidth=1, timex=None, title=None, time_window=None, dpi=image_dpi, **kwargs):
    if fn:
        if '/' in fn:  # if fn is a path
            create_path_if_not_exist(fn[0:fn.rfind('/') + 1])
    time_window = time_window or range(len(waveform))
    timex = timex or range(len(waveform))
    subplt_re = plt.subplot(111)
    subplt_re.plot([timex[t] for t in time_window], [waveform[t] for t in time_window], label='response', marker=marker, linewidth=linewidth)
    # subplt_abs = plt.subplot(212)
    # subplt_abs.plot(np.abs(waveform),
    #                 label='abs',
    #                 marker=marker,
    #                 linewidth=linewidth)
    if title:
        plt.title(title)
    if fn is not None:
        plt.gcf().set_size_inches(8, 6)
        plt.savefig(fn, dpi=dpi)

    if show:
        plt.show(block=False)
        input("Press [enter] to continue.")
    plt.close()


def create_path_if_not_exist(path):
    Path(path).mkdir(parents=True, exist_ok=True)


class MyPlot:

    def __init__(self, show=False, save_en=True, marker=None, linewidth=1, scatter=None, time_window=None, waveform_folder='waveform', dpi=None):
        self.par = locals()
        self.par.pop('self')

    def save_image(self, fn_in):
        try:
            if self.par['waveform_folder'] != 'waveform':
                imagefn = self.par['waveform_folder'] + '/' + fn_in
            else:
                imagefn = fn_in
            try:
                self.plot(Sig("{}/{}.json".format(self.par['waveform_folder'], fn_in)), fn="image/{}.png".format(imagefn))
            except:
                waveform = read_vec("{}/{}.json".format(self.par['waveform_folder'], fn_in))
                print("{}/{}.json".format(self.par['waveform_folder'], fn_in))
                self.plot(waveform, fn="image/{}.png".format(imagefn))
        except:
            print("{} not found".format(fn_in))

    def plot(self, waveform, show=None, fn=None, marker=None, linewidth=None, scatter=None, time_window=None, dpi=None):
        plot(waveform,
             show=show or self.par['show'],
             fn=fn if self.par['save_en'] else None,
             marker=marker or self.par['marker'],
             linewidth=linewidth or self.par['linewidth'],
             scatter=scatter or self.par['scatter'],
             time_window=time_window or self.par['time_window'],
             dpi=dpi or self.par['dpi'])


def plot(waveform, show=False, fn=None, marker=None, linewidth=1, dpi=image_dpi, scatter=None, time_window=None, default_folder='image'):

    par = locals()
    par.pop('waveform')

    if fn:
        if '/' in fn:  # if fn is a path
            create_path_if_not_exist(fn[0:fn.rfind('/') + 1])
        else:
            create_path_if_not_exist(default_folder)
            fn = default_folder + '/' + fn

    if isinstance(waveform, Sig):
        par.pop("fn")

        if waveform.wave.dtype == np.complex:
            plot_vecc(waveform.wave, fn=fn or ('image/' + waveform.name + '.png' if not scatter else 'image/' + waveform.name + '_cons.png'), timex=waveform.timex, **par)
        else:
            plot_vecf(waveform.wave, fn=fn or 'image/' + waveform.name + '.png', timex=waveform.timex, **par)
    elif isinstance(waveform, str):
        par.pop('fn')
        filename = waveform[0:waveform.find('.json')] + '.png' if '.json' in waveform else waveform + '.png'
        plot(read_vec(waveform), fn=fn or filename, **par)
    elif isinstance(waveform, np.ndarray) and waveform.ndim == 2:
        par.pop('fn')
        for (i, w) in enumerate(waveform):
            fnx = fn[0:fn.find('.png')] + '_' + str(i) + '.png' if fn else None
            plot(w, fn=fnx, **par)
    elif waveform.dtype == np.complex:
        plot_vecc(waveform, **par)
    else:
        plot_vecf(waveform, **par)
