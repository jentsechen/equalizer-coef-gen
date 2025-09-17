def append_search_path(l: list = []):
    from sys import path
    projpath = (__file__[0:(__file__.find('ws'))] if 'ws' in __file__ else __file__[0:(__file__.find('source'))]) + 'source'
    path += [projpath] + [projpath + '/' + x for x in l]


append_search_path(['pylib', 'pylib/test/', 'pylib/util'])

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os
from util.json_io import read_vec, save_json, read_json

file_path = '/'.join(__file__.split('/')[:-1])
dump_path = f'{file_path}/dump'
image_path = f'{file_path}/image'
if not os.path.exists(dump_path):
    os.mkdir(dump_path)
if not os.path.exists(image_path):
    os.mkdir(image_path)


class AnalysisData:

    def __init__(self, bittrue_folder):
        self.bittrue_folder = bittrue_folder

    def plot(self, names: list):

        def read_plot(fn):
            data = read_json(fn)
            fig.add_traces(go.Scatter(x=data['timestamp'], y=data['buf'], name=name, line_shape='hv', mode='lines+markers'))

        fig = go.Figure()
        for name in names:
            read_plot(self.bittrue_folder + name + '.json')
        fig.update_xaxes(tickfont_size=20)
        fig.update_yaxes(tickfont_size=20)
        fig.update_xaxes(title_font_size=20)
        fig.update_yaxes(title_font_size=20)
        fig.update_layout(legend_font_size=20)
        fig.update_layout(hovermode='x unified', hoverlabel_font_size=20)
        fig.write_html(f'{image_path}/results.html')

    def subplot(self, names: list):

        def read_plot(fn, row):
            data = read_json(fn)
            fig.add_trace(go.Scatter(x=data['timestamp'], y=data['buf'], name=name, line_shape='hv', mode='lines+markers'), row=row, col=1)

        fig = make_subplots(rows=len(names), cols=1, shared_xaxes=True)
        for idx, name in enumerate(names):
            read_plot(self.bittrue_folder + name + '.json', idx + 1)

        fig.update_xaxes(tickfont_size=20)
        fig.update_yaxes(tickfont_size=20)
        fig.update_xaxes(title_font_size=20)
        fig.update_yaxes(title_font_size=20)
        fig.update_layout(legend_font_size=20)
        fig.update_layout(hovermode='x unified', hoverlabel_font_size=20)
        fig.write_html(f'{image_path}/results.html')


def gen_rand_bits(len=None):
    return np.random.randint(0, 2, len)


def wgn(noise_std, len):
    return np.random.randn(len) * noise_std


def cwgn(noise_std, len):
    return (wgn(noise_std / np.sqrt(2), len) + 1j * wgn(noise_std / np.sqrt(2), len))


def awgn(data, snr_dB=None):
    if snr_dB is None:
        return data
    else:
        noise_std = np.sqrt(10**(-snr_dB / 10) * np.mean(np.abs(data)))
        return data + wgn(noise_std, len(data))


def cawgn(data, snr_dB=None):

    def gen_noise():
        return wgn(noise_std / np.sqrt(2.0), len(data))

    if snr_dB is None:
        return data
    else:
        noise_std = np.sqrt(10**(-snr_dB / 10) * np.mean(np.abs(data)))
        return data + gen_noise() + 1j * gen_noise()


def to_SNR(EbN0_dB: list, cr: float = 1, n_bits: int = 1):
    if isinstance(EbN0_dB, list):
        return [e + 10 * np.log10(2 * cr * n_bits) for e in EbN0_dB]
    else:
        return EbN0_dB + 10 * np.log10(2 * cr * n_bits)


def collect_report(folder, fn):
    import os
    files = os.listdir(folder)
    results = []
    for file in files:
        results.append(read_json(fn=folder + '/' + file))
    snr_dB = [result['simset']['snr_dB'] for result in results]
    idx = np.argsort(snr_dB)
    sorted_results = [results[i] for i in idx]
    save_json(fn, sorted_results)


def get_snr_fer(fn, sfer='fer', criterion='standard', iter_th=None):

    def iter_to_fer(iters, th):
        return sum([1 for iter in iters if iter > th or iter == max_iter_num]) / len(iters)

    data = read_json(fn)
    snr_dB = [d['simset']['snr_dB'] for d in data]

    if criterion == 'standard':
        fer = [d[sfer] for d in data]
    elif criterion == 'ldpc':
        max_iter_num = data[0]['simset']['max_iter_num']
        fer = [iter_to_fer(d['iter_num_coll'], iter_th) for d in data]
    elif criterion == 'ldpc_info':
        max_iter_num = 1000
        fer = [iter_to_fer(d['golden_iter_num_coll'], iter_th) for d in data]
    elif criterion == 'bch':
        max_iter_num = 1000
        fer = [iter_to_fer(d['bch_iter_num_coll'], iter_th) for d in data]
    fer = [r for r in fer if r != 0]
    return snr_dB, fer


def plot_fer(snr_dB, fer, fn):
    fig = PlotlyFig(general_size=40)
    fig.semilog(x=snr_dB, y=fer, name='FER', mode='lines+markers')
    fig.set_line(line_width=8, marker_size=8)
    fig.set_title(x_title='SNR (dB)', y_title='FER')
    fig.write_html(fn)


class PlotlyFig:

    # def __init__(self, subplot=False, rows=1, cols=1, general_size=30) -> None:
    def __init__(self, **kwargs) -> None:
        subplot = kwargs.get('subplot', False)
        rows = kwargs.get('rows', 1)
        cols = kwargs.get('cols', 1)
        general_size = kwargs.get('general_size', 39)
        subplot_titles = kwargs.get('subplot_titles', [''])
        self.fig = go.Figure() if not subplot else make_subplots(rows=rows, cols=cols, subplot_titles=subplot_titles)
        self.general_size = general_size

    def subplot(self, **kwargs):
        x = kwargs.get('x', 0)
        y = kwargs.get('y', 0)
        row = kwargs.get('row', 1)
        col = kwargs.get('col', 1)
        mode = kwargs.get('mode', 'lines')
        xaxis_range = kwargs.get('xaxis_range', None)
        yaxis_range = kwargs.get('yaxis_range', None)
        name = kwargs.get('name', '')
        self.fig.add_trace(go.Scatter(x=x, y=y, mode=mode, name=name), row=row, col=col)
        self.set_range(xaxis_range=xaxis_range, yaxis_range=yaxis_range)
        self.update_axes()
        self.update_layout()

    def plot(self, **kwargs):
        x = kwargs.get('x', 0)
        y = kwargs.get('y', 0)
        mode = kwargs.get('mode', 'lines')
        xaxis_range = kwargs.get('xaxis_range', None)
        yaxis_range = kwargs.get('yaxis_range', None)
        name = kwargs.get('name', '')
        self.fig.add_traces(go.Scatter(x=x, y=y, mode=mode, name=name))
        self.set_range(xaxis_range=xaxis_range, yaxis_range=yaxis_range)
        self.update_axes()
        self.update_layout()

    def semilog(self, **kwargs):
        x = kwargs.get('x', 0)
        y = kwargs.get('y', 0)
        mode = kwargs.get('mode', 'lines')
        xaxis_range = kwargs.get('xaxis_range', None)
        yaxis_range = kwargs.get('yaxis_range', None)
        name = kwargs.get('name', '')
        line = kwargs.get('line', {'dash': 'solid'})
        self.fig.add_traces(go.Scatter(x=x, y=y, mode=mode, name=name, line=line))
        self.fig.update_yaxes(type='log')
        self.fig.update_yaxes(dtick=1)
        self.fig.update_layout(yaxis_exponentformat='power')
        self.set_range(xaxis_range=xaxis_range, yaxis_range=yaxis_range)
        self.update_axes()
        self.update_layout()

    def update_axes(self):
        self.fig.update_layout(title_font_size=self.general_size)
        self.fig.update_xaxes(tickfont_size=self.general_size)
        self.fig.update_yaxes(tickfont_size=self.general_size)
        self.fig.update_xaxes(title_font_size=self.general_size)
        self.fig.update_yaxes(title_font_size=self.general_size)
        self.fig.update_xaxes(gridcolor='rgb(150, 150, 150)')
        self.fig.update_xaxes(linecolor='black')
        self.fig.update_yaxes(gridcolor='rgb(150, 150, 150)')
        self.fig.update_yaxes(linecolor='black')
        self.fig.update_xaxes(zeroline=False)
        self.fig.update_yaxes(zeroline=False)
        self.fig.update_xaxes(minor_showgrid=True)
        self.fig.update_yaxes(minor_showgrid=True)
        self.fig.update_yaxes(minor_ticks='inside', minor_ticklen=5)
        self.fig.update_xaxes(gridwidth=3)
        self.fig.update_yaxes(gridwidth=3)
        # self.fig.update_yaxes(dtick=1)
        # self.fig.update_yaxes(minor_dtick=2)

    def update_layout(self):
        self.fig.update_layout(go.Layout(paper_bgcolor='rgb(255, 255, 255)'))
        self.fig.update_layout(go.Layout(plot_bgcolor='rgb(230, 230, 230)'))
        self.fig.update_layout(legend_font_size=self.general_size)

    def set_range(self, xaxis_range=None, yaxis_range=None):
        if xaxis_range is not None:
            self.fig.update_layout(xaxis_range=xaxis_range)
        if yaxis_range is not None:
            self.fig.update_layout(yaxis_range=yaxis_range)

    def set_line(self, line_width=3, marker_size=3):
        self.fig.update_traces(line_width=line_width, selector=dict(type='scatter'))
        self.fig.update_traces(marker_size=marker_size, selector=dict(type='scatter'))

    def set_title(self, title=None, x_title=None, y_title=None):
        if title is not None:
            self.fig.update_layout(title_text=title)
        if x_title is not None:
            self.fig.update_xaxes(title_text=x_title)
        if y_title is not None:
            self.fig.update_yaxes(title_text=y_title)

    def write_html(self, filename):
        self.fig.write_html(filename)


def test_plot():
    fig = PlotlyFig()
    t = np.linspace(0, 2, 100)
    c = np.cos(2 * np.pi * t)
    s = np.sin(2 * np.pi * t)
    fig.plot(x=t, y=c, mode='lines+markers')
    fig.plot(x=t, y=s)
    fig.write_html(f'{image_path}/test_plot.html')


def test_semilog():
    fig = PlotlyFig()
    x = [0, 1, 2, 3, 4]
    y = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    fig.semilog(x=x, y=y, yaxis_range=[-8, 0])
    fig.write_html(f'{image_path}/test_semilog.html')


def test_subplot():
    fig = PlotlyFig(subplot=True, rows=2, cols=1)
    t = np.linspace(0, 2, 100)
    c = np.cos(2 * np.pi * t)
    s = np.sin(2 * np.pi * t)
    fig.subplot(x=t, y=c, mode='lines+markers', row=1, col=1)
    fig.subplot(x=t, y=s, row=2, col=1)
    fig.set_line(line_width=3)
    fig.write_html(f'{image_path}/test_subplot.html')


if __name__ == '__main__':
    test_plot()
    test_semilog()
    test_subplot()
